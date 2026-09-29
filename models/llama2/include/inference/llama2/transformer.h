#pragma once
// Llama-2 transformer graph. Math spec: SPEC/05_TRANSFORMER_LLAMA2.md.
// Model-specific: lives under models/llama2 (see SPEC/09_MODEL_LAYOUT.md).

#include "inference/core/backend_plugin.h"
#include "inference/core/config.h"
#include "inference/llama2/weights.h"

namespace inference {
namespace llama2 {

// All activation buffers, pre-allocated by Engine (no malloc here).
struct RunState {
  float* x = nullptr;       // [dim]
  float* xb = nullptr;      // [dim] attention output / FFN input
  float* xb2 = nullptr;     // [dim] attention output projection
  float* hb = nullptr;      // [hidden_dim] FFN gate (post-SiLU, pre-multiply)
  float* hb2 = nullptr;     // [hidden_dim] FFN up-projection
  float* q = nullptr;       // [dim] query
  float* k = nullptr;       // [kv_dim] key (this position)
  float* v = nullptr;       // [kv_dim] value (this position)
  float* att = nullptr;     // [n_heads, seq_len] attention scores
  float* logits = nullptr;  // [vocab_size]
  float* key_cache = nullptr;    // [n_layers, seq_len, kv_dim]
  float* value_cache = nullptr;  // [n_layers, seq_len, kv_dim]
};

// Scratch sizes (in floats) for Engine allocation.
size_t activation_floats(const core::ModelConfig& c);
size_t kv_floats(const core::ModelConfig& c);

// One forward step: token -> logits. pos = current position.
void forward(const Weights& w, RunState& s, int token, int pos,
             core::IComputeBackend& backend);

}  // namespace llama2
}  // namespace inference
