#pragma once
// Llama-2 weight store: RAII mmap'd checkpoint + typed views.
// Format: Karpathy llama2.bin, GROUPED BY TENSOR across layers
// (all rms_att, then all Wq, ... — see SPEC/06_MEMORY_WEIGHTS.md).
// Model-specific: other architectures get their own loader under
// models/<name>/ (see SPEC/09_MODEL_LAYOUT.md).

#include <cstddef>
#include <string>

#include "inference/core/config.h"

namespace inference {
namespace llama2 {

class Weights {
 public:
  Weights() = default;
  ~Weights();
  Weights(const Weights&) = delete;
  Weights& operator=(const Weights&) = delete;
  Weights(Weights&&) noexcept;
  Weights& operator=(Weights&&) noexcept;

  // Memory-maps checkpoint, parses header, binds views. Throws on error.
  // If vocab_size/seq_len are 0, values from file header are used.
  void load(const std::string& path, int vocab_override = 0,
            int seq_override = 0);

  const core::ModelConfig& config() const { return config_; }
  // Raw float pool (all weights concatenated, for size checks).
  const float* data() const { return data_; }
  size_t num_floats() const { return num_floats_; }

  // --- typed views (row-major) ---
  const float* token_embedding() const { return data_; }  // [vocab, dim]
  const float* attn_rms(int l) const { return g_attn_rms_ + (size_t)l * dim_; }
  const float* Wq(int l) const { return g_wq_ + (size_t)l * dim_ * dim_; }
  const float* Wk(int l) const { return g_wk_ + (size_t)l * dim_ * kv_dim_; }
  const float* Wv(int l) const { return g_wv_ + (size_t)l * dim_ * kv_dim_; }
  const float* Wo(int l) const { return g_wo_ + (size_t)l * dim_ * dim_; }
  const float* ffn_rms(int l) const { return g_ffn_rms_ + (size_t)l * dim_; }
  const float* W1(int l) const { return g_w1_ + (size_t)l * hidden_ * dim_; }
  const float* W2(int l) const { return g_w2_ + (size_t)l * dim_ * hidden_; }
  const float* W3(int l) const { return g_w3_ + (size_t)l * hidden_ * dim_; }
  const float* final_rms() const { return final_rms_; }
  const float* Wcls() const {
    return config_.shared_classifier ? token_embedding() : wcls_;
  }

 private:
  core::ModelConfig config_;
  int dim_ = 0, hidden_ = 0, kv_dim_ = 0;
  // mmap handle
  void* map_base_ = nullptr;  // includes header
  size_t map_bytes_ = 0;
  int mmap_fd_ = -1;
  // float pool after header
  const float* data_ = nullptr;
  size_t num_floats_ = 0;
  // grouped tensor bases (see load())
  const float *g_attn_rms_ = nullptr, *g_wq_ = nullptr, *g_wk_ = nullptr,
              *g_wv_ = nullptr, *g_wo_ = nullptr, *g_ffn_rms_ = nullptr,
              *g_w1_ = nullptr, *g_w2_ = nullptr, *g_w3_ = nullptr;
  const float* final_rms_ = nullptr;
  const float* wcls_ = nullptr;
};

}  // namespace llama2
}  // namespace inference
