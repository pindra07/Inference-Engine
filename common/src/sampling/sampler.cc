// Sampler: greedy / temperature + top-k + nucleus(top-p), xoroshiro128+.

#include "inference/sampling/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace inference {
namespace sampling {
namespace {

uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

uint64_t next_rand(uint64_t* s) {  // xoroshiro128+
  uint64_t s0 = s[0], s1 = s[1];
  uint64_t r = s0 + s1;
  s1 ^= s0;
  s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
  s[1] = rotl(s1, 36);
  return r;
}

float randf(uint64_t* s) { return (next_rand(s) >> 11) * (1.0f / 9007199254740992.0f); }

}  // namespace

int sample(const float* logits, int vocab, const SampleConfig& cfg,
           uint64_t* rng_state, float* scratch) {
  if (cfg.temperature <= 0.0f || cfg.temperature != cfg.temperature) {
    int best = 0;
    for (int i = 1; i < vocab; ++i)
      if (logits[i] > logits[best]) best = i;
    return best;
  }
  for (int i = 0; i < vocab; ++i) scratch[i] = logits[i] / cfg.temperature;
  // softmax
  float mx = scratch[0];
  for (int i = 1; i < vocab; ++i) mx = std::max(mx, scratch[i]);
  double sum = 0;
  for (int i = 0; i < vocab; ++i) {
    scratch[i] = std::exp(scratch[i] - mx);
    sum += scratch[i];
  }
  float inv = (float)(1.0 / sum);
  for (int i = 0; i < vocab; ++i) scratch[i] *= inv;

  // top-k shortlist, highest probability first.
  std::vector<int> idx((size_t)vocab);
  std::iota(idx.begin(), idx.end(), 0);
  int k = cfg.top_k > 0 ? std::min(cfg.top_k, vocab) : vocab;
  auto cmp = [&](int a, int b) { return scratch[a] > scratch[b]; };
  if (k < vocab) {
    // OPT (SPEC/08): partition in O(vocab), then sort only the top-k window.
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(), cmp);
    std::sort(idx.begin(), idx.begin() + k, cmp);
  } else {
    std::sort(idx.begin(), idx.end(), cmp);
  }
  // top-p truncation over the top-k set
  int cutoff = k;
  if (cfg.top_p < 1.0f) {
    double cumsum = 0;
    cutoff = 0;
    for (int i = 0; i < k; ++i) {
      cumsum += scratch[idx[i]];
      ++cutoff;
      if ((float)cumsum >= cfg.top_p) break;
    }
    if (cutoff < 1) cutoff = 1;
  }
  double mass = 0;
  for (int i = 0; i < cutoff; ++i) mass += scratch[idx[i]];
  double r = randf(rng_state) * mass;
  double acc = 0;
  for (int i = 0; i < cutoff; ++i) {
    acc += scratch[idx[i]];
    if (r < acc) return idx[i];
  }
  return idx[cutoff - 1];
}

}  // namespace sampling
}  // namespace inference
