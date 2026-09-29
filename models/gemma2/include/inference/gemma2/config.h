#pragma once
// Gemma2 model config: shared dense-decoder base + Gemma-specific knobs.
// Field mapping to HF Gemma2Config is documented per member.

#include "inference/core/config.h"

namespace inference {
namespace gemma2 {

struct ModelConfig {
  core::ModelConfig base;  // dim=hidden_size, hidden=intermediate_size,
                           // layers, heads, kv_heads, vocab, seq, rope_theta
  int head_dim = 0;        // HF head_dim (explicit, 256 — NOT dim/heads)
  int sliding_window = 0;  // HF sliding_window; <=0 = full attention on all
                           // layers. Sliding applies to layer l iff
                           // ((l+1)%2==1) — HF layer_types pattern.
  float attn_scale = 0.0f;  // HF query_pre_attn_scalar^-0.5 (NOT 1/sqrt(hd))
  float attn_softcap = 0.0f;   // HF attn_logit_softcapping; <=0 disables
  float final_softcap = 0.0f;  // HF final_logit_softcapping; <=0 disables
  float rms_eps = 1e-6f;       // HF rms_norm_eps

  int q_dim() const { return base.n_heads * head_dim; }
  int kv_dim() const { return base.n_kv_heads * head_dim; }
  bool sliding_layer(int l) const {
    return sliding_window > 0 && ((l + 1) % 2 == 1);
  }
  bool valid() const {
    return base.valid() && head_dim > 0 && q_dim() > 0 && kv_dim() > 0 &&
           attn_scale > 0.0f;
  }
};

}  // namespace gemma2
}  // namespace inference
