#pragma once
// Gemma2 transformer graph. Math spec: SPEC/10_GEMMA2.md (verified against
// HF modeling_gemma2.py). Model-specific: models/gemma2.

#include "inference/core/backend_plugin.h"
#include "inference/gemma2/config.h"
#include "inference/gemma2/weights.h"

namespace inference {
namespace gemma2 {

// All activation buffers, pre-allocated by Engine (no malloc here).
struct RunState {
  float* x = nullptr;        // [dim]
  float* xb = nullptr;       // [dim] block input / Wo output
  float* xb2 = nullptr;      // [dim] post-norm output (residual addend)
  float* hb = nullptr;       // [hidden] FFN gate (post-GeGLU, pre-multiply)
  float* hb2 = nullptr;      // [hidden] FFN up-projection
  float* q = nullptr;        // [q_dim = heads*head_dim]
  float* k = nullptr;        // [kv_dim]
  float* v = nullptr;        // [kv_dim]
  float* att_out = nullptr;  // [q_dim] attention output (pre-Wo)
  float* att = nullptr;      // [n_heads, seq_len] attention scores
  float* logits = nullptr;   // [vocab_size]
  float* key_cache = nullptr;    // [n_layers, seq_len, kv_dim]
  float* value_cache = nullptr;  // [n_layers, seq_len, kv_dim]
};

size_t activation_floats(const ModelConfig& c);
size_t kv_floats(const ModelConfig& c);

// One forward step: token -> logits. pos = current position.
void forward(const Weights& w, RunState& s, int token, int pos,
             core::IComputeBackend& backend);

}  // namespace gemma2
}  // namespace inference
