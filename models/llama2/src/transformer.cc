// Llama-2 transformer forward. Owns the model graph; every hot loop
// delegates to IComputeBackend. Math spec: SPEC/05_TRANSFORMER_LLAMA2.md.

#include "inference/llama2/transformer.h"

#include <cstring>

namespace inference {
namespace llama2 {

size_t activation_floats(const core::ModelConfig& c) {
  size_t n = 0;
  n += c.dim;          // x
  n += c.dim;          // xb
  n += c.dim;          // xb2
  n += c.hidden_dim;   // hb
  n += c.hidden_dim;   // hb2
  n += c.dim;          // q
  n += c.kv_dim();     // k
  n += c.kv_dim();     // v
  n += (size_t)c.n_heads * c.seq_len;  // att
  n += c.vocab_size;                   // logits
  return n;
}

size_t kv_floats(const core::ModelConfig& c) {
  return (size_t)2 * c.n_layers * c.seq_len * c.kv_dim();
}

void forward(const Weights& w, RunState& s, int token, int pos,
             core::IComputeBackend& backend) {
  const core::ModelConfig& p = w.config();
  const int dim = p.dim, hidden = p.hidden_dim, heads = p.n_heads,
            kv_heads = p.n_kv_heads, kv_dim = p.kv_dim(),
            head_dim = p.head_dim();

  const float* emb = w.token_embedding() + (size_t)token * dim;
  std::memcpy(s.x, emb, (size_t)dim * sizeof(float));

  for (int l = 0; l < p.n_layers; ++l) {
    // --- attention block ---
    backend.rmsnorm(s.x, w.attn_rms(l), s.xb, dim, 1e-5f);
    backend.matvec(w.Wq(l), s.xb, s.q, dim, dim);
    backend.matvec(w.Wk(l), s.xb, s.k, kv_dim, dim);
    backend.matvec(w.Wv(l), s.xb, s.v, kv_dim, dim);
    backend.rope(s.q, s.k, head_dim, heads, kv_heads, pos, p.rope_theta);
    float* kc = s.key_cache + ((size_t)l * p.seq_len + pos) * kv_dim;
    float* vc = s.value_cache + ((size_t)l * p.seq_len + pos) * kv_dim;
    std::memcpy(kc, s.k, (size_t)kv_dim * sizeof(float));
    std::memcpy(vc, s.v, (size_t)kv_dim * sizeof(float));
    const float* layer_k = s.key_cache + (size_t)l * p.seq_len * kv_dim;
    const float* layer_v = s.value_cache + (size_t)l * p.seq_len * kv_dim;
    backend.attention(s.q, layer_k, layer_v, s.xb, s.att, heads, kv_heads,
                      head_dim, pos);
    backend.matvec(w.Wo(l), s.xb, s.xb2, dim, dim);
    for (int i = 0; i < dim; ++i) s.x[i] += s.xb2[i];  // residual

    // --- SwiGLU FFN block: W2( SiLU(W1 x) * (W3 x) ) ---
    backend.rmsnorm(s.x, w.ffn_rms(l), s.xb, dim, 1e-5f);
    backend.matvec(w.W1(l), s.xb, s.hb, hidden, dim);
    backend.matvec(w.W3(l), s.xb, s.hb2, hidden, dim);
    backend.silu(s.hb, hidden);
    backend.elem_mul(s.hb, s.hb, s.hb2, hidden);  // gate * up (in place)
    backend.matvec(w.W2(l), s.hb, s.xb, dim, hidden);
    for (int i = 0; i < dim; ++i) s.x[i] += s.xb[i];  // residual
  }
  backend.rmsnorm(s.x, w.final_rms(), s.x, dim, 1e-5f);
  backend.matvec(w.Wcls(), s.x, s.logits, p.vocab_size, dim);
}

}  // namespace llama2
}  // namespace inference
