#pragma once
// Sampler: greedy / temperature / top-k / top-p. See SPEC/07.

#include <cstdint>
#include <vector>

namespace inference {
namespace sampling {

struct SampleConfig {
  float temperature = 1.0f;  // 0 => greedy
  float top_p = 0.9f;        // 1.0 => disabled
  int top_k = 0;             // 0 => disabled
  uint64_t seed = 42;
};

// scratch must hold >= vocab floats. rng_state updated in place.
int sample(const float* logits, int vocab, const SampleConfig& cfg,
           uint64_t* rng_state, float* scratch);

}  // namespace sampling
}  // namespace inference
