// Gemma2 transformer forward. Mirrors HF Gemma2DecoderLayer exactly:
//   x += post_attn(attn(input_norm(x))); x += post_ffn(mlp(pre_ffn(x)))
// Every hot loop delegates to IComputeBackend. Spec: SPEC/10_GEMMA2.md.

#include "inference/gemma2/transformer.h"

#include <cmath>
#include <cstring>

namespace inference {
namespace gemma2 {

size_t activation_floats(const ModelConfig& c) {
  size_t n = 0;
  n += c.base.dim;          // x
  n += c.base.dim;          // xb
  n += c.base.dim;          // xb2
  n += c.base.hidden_dim;   // hb
  n += c.base.hidden_dim;   // hb2
  n += c.q_dim();           // q
  n += c.kv_dim();          // k
  n += c.kv_dim();          // v
  n += c.q_dim();           // att_out
  n += (size_t)c.base.n_heads * c.base.seq_len;  // att
  n += c.base.vocab_size;                        // logits
  return n;
}

size_t kv_floats(const ModelConfig& c) {
  return (size_t)2 * c.base.n_layers * c.base.seq_len * c.kv_dim();
}

void forward(const Weights& w, RunState& s, int token, int pos,
             core::IComputeBackend& backend) {
  const ModelConfig& p = w.config();
  const int dim = p.base.dim, hidden = p.base.hidden_dim;
  const int heads = p.base.n_heads, kv_heads = p.base.n_kv_heads;
  const int head_dim = p.head_dim, q_dim = p.q_dim(), kv_dim = p.kv_dim();

  // Gemma2TextScaledWordEmbedding: emb * sqrt(dim).
  const float* emb = w.token_embedding() + (size_t)token * dim;
  std::memcpy(s.x, emb, (size_t)dim * sizeof(float));
  const float esc = std::sqrt((float)dim);
  for (int i = 0; i < dim; ++i) s.x[i] *= esc;

  for (int l = 0; l < p.base.n_layers; ++l) {
    const int window = p.sliding_layer(l) ? p.sliding_window : 0;

    // --- attention block: x += post_attn(attn(input_norm(x))) ---
    backend.rmsnorm(s.x, w.input_rms(l), s.xb, dim, p.rms_eps);
    backend.matvec(w.Wq(l), s.xb, s.q, q_dim, dim);
    backend.matvec(w.Wk(l), s.xb, s.k, kv_dim, dim);
    backend.matvec(w.Wv(l), s.xb, s.v, kv_dim, dim);
    backend.rope(s.q, s.k, head_dim, heads, kv_heads, pos, p.base.rope_theta);
    float* kc = s.key_cache + ((size_t)l * p.base.seq_len + pos) * kv_dim;
    float* vc = s.value_cache + ((size_t)l * p.base.seq_len + pos) * kv_dim;
    std::memcpy(kc, s.k, (size_t)kv_dim * sizeof(float));
    std::memcpy(vc, s.v, (size_t)kv_dim * sizeof(float));
    const float* layer_k = s.key_cache + (size_t)l * p.base.seq_len * kv_dim;
    const float* layer_v = s.value_cache + (size_t)l * p.base.seq_len * kv_dim;
    backend.attention_softcap(s.q, layer_k, layer_v, s.att_out, s.att, heads,
                              kv_heads, head_dim, pos, p.attn_scale,
                              p.attn_softcap, window);
    backend.matvec(w.Wo(l), s.att_out, s.xb, dim, q_dim);
    backend.rmsnorm(s.xb, w.post_attn_rms(l), s.xb2, dim, p.rms_eps);
    for (int i = 0; i < dim; ++i) s.x[i] += s.xb2[i];  // residual

    // --- GeGLU FFN block: x += post_ffn(down(gelu(gate) * up)) ---
    backend.rmsnorm(s.x, w.pre_ffn_rms(l), s.xb, dim, p.rms_eps);
    backend.matvec(w.Wgate(l), s.xb, s.hb, hidden, dim);
    backend.matvec(w.Wup(l), s.xb, s.hb2, hidden, dim);
    backend.gelu_tanh(s.hb, hidden);
    backend.elem_mul(s.hb, s.hb, s.hb2, hidden);  // gate * up (in place)
    backend.matvec(w.Wdown(l), s.hb, s.xb, dim, hidden);
    backend.rmsnorm(s.xb, w.post_ffn_rms(l), s.xb2, dim, p.rms_eps);
    for (int i = 0; i < dim; ++i) s.x[i] += s.xb2[i];  // residual
  }
  backend.rmsnorm(s.x, w.final_rms(), s.x, dim, p.rms_eps);
  backend.matvec(w.Wcls(), s.x, s.logits, p.base.vocab_size, dim);
  backend.tanh_softcap(s.logits, p.base.vocab_size, p.final_softcap);
}

}  // namespace gemma2
}  // namespace inference
