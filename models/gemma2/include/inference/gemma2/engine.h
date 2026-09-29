#pragma once
// Gemma2 orchestrator: owns weights, KV cache, tokenizer, backend.
// Same shape as llama2::LlamaEngine (see SPEC/09); model-specific wiring.

#include <functional>
#include <string>
#include <vector>

#include "inference/core/plugin_loader.h"
#include "inference/gemma2/transformer.h"
#include "inference/gemma2/weights.h"
#include "inference/sampling/sampler.h"
#include "inference/tokenizer/tokenizer.h"

namespace inference {
namespace gemma2 {

struct EngineConfig {
  std::string checkpoint;    // .ggm2 path
  std::string tokenizer;     // tokenizer.bin path
  std::string backend_path;  // .dylib/.so/.dll ; empty => built-in reference
  sampling::SampleConfig sample{1.0f, 0.9f, 0, 42};
};

class GemmaEngine {
 public:
  GemmaEngine();
  ~GemmaEngine();
  GemmaEngine(const GemmaEngine&) = delete;
  GemmaEngine& operator=(const GemmaEngine&) = delete;

  void load(const EngineConfig& cfg);  // throws on error
  const std::string& backend_name() const { return backend_name_; }

  // Prefill prompt tokens, then decode up to max_steps.
  // on_token(id, decoded_piece) called per generated token; return false to stop.
  std::vector<int> generate(
      const std::string& prompt, int max_steps,
      const std::function<bool(int, const std::string&)>& on_token = nullptr);

  const ModelConfig& model_config() const { return weights_.config(); }

  // Prompt length in tokens (for benchmark metadata, capacity planning).
  int count_prompt_tokens(const std::string& prompt) const {
    return (int)tokenizer_.encode(prompt, true).size();
  }

 private:
  Weights weights_;
  tokenizer::Tokenizer tokenizer_;
  RunState state_{};
  core::BackendPtr backend_{nullptr, nullptr};
  std::string backend_name_;
  sampling::SampleConfig sample_cfg_;
  std::vector<float> sample_scratch_;  // [vocab_size], reused every step
  uint64_t rng_state_ = 42;
  float* act_mem_ = nullptr;  // 64B-aligned activations
  float* kv_mem_ = nullptr;   // 64B-aligned KV cache

  void alloc_state();
  void free_state();
  const float* forward_step(int token, int pos);
};

}  // namespace gemma2
}  // namespace inference
