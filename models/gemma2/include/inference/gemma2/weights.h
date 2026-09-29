#pragma once
// Gemma2 weight store: RAII mmap'd GGM2 checkpoint + typed views.
// Format: SPEC/10_GEMMA2.md. Norm weights are stored FOLDED as (1+w)
// (Gemma2RMSNorm semantics) in owned buffers, so the shared rmsnorm kernel
// serves both model families with zero ABI change.

#include <cstddef>
#include <string>
#include <vector>

#include "inference/gemma2/config.h"

namespace inference {
namespace gemma2 {

class Weights {
 public:
  Weights() = default;
  ~Weights();
  Weights(const Weights&) = delete;
  Weights& operator=(const Weights&) = delete;
  Weights(Weights&&) noexcept;
  Weights& operator=(Weights&&) noexcept;

  void load(const std::string& path);  // throws on error

  const ModelConfig& config() const { return config_; }

  // --- typed views (row-major; norm views are folded (1+w)) ---
  const float* token_embedding() const { return data_; }  // [vocab, dim]
  const float* input_rms(int l) const { return g_in_ + (size_t)l * dim_; }
  const float* Wq(int l) const { return g_wq_ + (size_t)l * q_dim_ * dim_; }
  const float* Wk(int l) const { return g_wk_ + (size_t)l * kv_dim_ * dim_; }
  const float* Wv(int l) const { return g_wv_ + (size_t)l * kv_dim_ * dim_; }
  const float* Wo(int l) const { return g_wo_ + (size_t)l * dim_ * q_dim_; }
  const float* post_attn_rms(int l) const {
    return g_post_attn_ + (size_t)l * dim_;
  }
  const float* pre_ffn_rms(int l) const {
    return g_pre_ffn_ + (size_t)l * dim_;
  }
  const float* Wgate(int l) const {
    return g_wgate_ + (size_t)l * hidden_ * dim_;
  }
  const float* Wdown(int l) const {
    return g_wdown_ + (size_t)l * dim_ * hidden_;
  }
  const float* Wup(int l) const { return g_wup_ + (size_t)l * hidden_ * dim_; }
  const float* post_ffn_rms(int l) const {
    return g_post_ffn_ + (size_t)l * dim_;
  }
  const float* final_rms() const { return final_rms_; }
  const float* Wcls() const {
    return config_.base.shared_classifier ? token_embedding() : wcls_;
  }

 private:
  // (Re)derive norm views from folded_/dims. Takes n_layers explicitly
  // because config_ isn't populated yet when load() binds (SPEC/10 lesson:
  // binding from an empty config silently aliased every norm to layer 0).
  void bind_folded_views(int n_layers);

  ModelConfig config_;
  int dim_ = 0, hidden_ = 0, q_dim_ = 0, kv_dim_ = 0;
  void* map_base_ = nullptr;
  size_t map_bytes_ = 0;
  int mmap_fd_ = -1;
  const float* data_ = nullptr;
  size_t num_floats_ = 0;
  // mmap'd (unfolded) tensor groups
  const float *g_wq_ = nullptr, *g_wk_ = nullptr, *g_wv_ = nullptr,
              *g_wo_ = nullptr, *g_wgate_ = nullptr, *g_wdown_ = nullptr,
              *g_wup_ = nullptr;
  // owned folded (1+w) norms: [input|post_attn|pre_ffn|post_ffn] ×L + final
  std::vector<float> folded_;
  const float *g_in_ = nullptr, *g_post_attn_ = nullptr,
              *g_pre_ffn_ = nullptr, *g_post_ffn_ = nullptr;
  const float* final_rms_ = nullptr;
  const float* wcls_ = nullptr;
};

}  // namespace gemma2
}  // namespace inference
