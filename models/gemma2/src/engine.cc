// GemmaEngine: load weights/tokenizer/backend, pre-allocate state,
// prefill + decode loop. Gemma special ids: bos=2, eos=1 (SPEC/10).

#include "inference/gemma2/engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "inference/core/reference_backend.h"

namespace inference {
namespace gemma2 {
namespace {

// Portable 64B-aligned zeroed allocation (SIMD-safe).
float* aligned_calloc(size_t n) {
  void* p = nullptr;
#if defined(_WIN32)
  p = _aligned_malloc(n * sizeof(float), 64);
#else
  if (posix_memalign(&p, 64, n * sizeof(float)) != 0) p = nullptr;
#endif
  if (p) std::memset(p, 0, n * sizeof(float));
  return (float*)p;
}
void aligned_free(float* p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

void delete_backend(core::IComputeBackend* p) { delete p; }

}  // namespace

GemmaEngine::GemmaEngine() = default;
GemmaEngine::~GemmaEngine() { free_state(); }

void GemmaEngine::load(const EngineConfig& cfg) {
  weights_.load(cfg.checkpoint);
  tokenizer_.load(cfg.tokenizer);
  tokenizer_.set_special_ids(/*bos=*/2, /*eos=*/1);
  tokenizer_.set_dummy_prefix(false);  // Gemma pieces carry their own spaces
  // Chat markers must survive encode() as single ids. Tokenizers built by
  // our converter always carry them; plain-text use works regardless.
  for (const char* m : {"<start_of_turn>", "<end_of_turn>"}) {
    try {
      tokenizer_.add_special(m);
    } catch (const std::exception&) {
      std::fprintf(stderr, "warning: tokenizer lacks '%s'; --chat needs it\n",
                   m);
    }
  }
  sample_cfg_ = cfg.sample;
  rng_state_ = cfg.sample.seed ? cfg.sample.seed : 42;
  rng_state_ ^= 0x9e3779b97f4a7c15ull;
  sample_scratch_.assign((size_t)weights_.config().base.vocab_size, 0.0f);

  if (!cfg.backend_path.empty()) {
    core::PluginLoader loader;
    backend_ = loader.load(cfg.backend_path);
  } else {
    backend_ = core::BackendPtr(new core::ReferenceBackend(), delete_backend);
  }
  backend_name_ = backend_->name();
  alloc_state();
}

void GemmaEngine::alloc_state() {
  free_state();
  const auto& c = weights_.config();
  act_mem_ = aligned_calloc(activation_floats(c));
  kv_mem_ = aligned_calloc(kv_floats(c));
  if (!act_mem_ || !kv_mem_) throw std::runtime_error("oom allocating run state");
  float* p = act_mem_;
  state_.x = p;
  p += c.base.dim;
  state_.xb = p;
  p += c.base.dim;
  state_.xb2 = p;
  p += c.base.dim;
  state_.hb = p;
  p += c.base.hidden_dim;
  state_.hb2 = p;
  p += c.base.hidden_dim;
  state_.q = p;
  p += c.q_dim();
  state_.k = p;
  p += c.kv_dim();
  state_.v = p;
  p += c.kv_dim();
  state_.att_out = p;
  p += c.q_dim();
  state_.att = p;
  p += (size_t)c.base.n_heads * c.base.seq_len;
  state_.logits = p;
  state_.key_cache = kv_mem_;
  state_.value_cache = kv_mem_ + kv_floats(c) / 2;
}

void GemmaEngine::free_state() {
  aligned_free(act_mem_);
  act_mem_ = nullptr;
  aligned_free(kv_mem_);
  kv_mem_ = nullptr;
  state_ = RunState{};
}

const float* GemmaEngine::forward_step(int token, int pos) {
  forward(weights_, state_, token, pos, *backend_);
  return state_.logits;
}

std::vector<int> GemmaEngine::generate(
    const std::string& prompt, int max_steps,
    const std::function<bool(int, const std::string&)>& on_token) {
  const auto& c = weights_.config();
  std::vector<int> prompt_ids = tokenizer_.encode(prompt, true);
  if ((int)prompt_ids.size() + max_steps > c.base.seq_len)
    max_steps = c.base.seq_len - (int)prompt_ids.size();
  if (max_steps <= 0) throw std::runtime_error("prompt longer than seq_len");

  std::vector<int> out;
  out.reserve((size_t)max_steps);
  uint64_t rng[2] = {rng_state_, rng_state_ ^ 0xbf58476d1ce4e5b9ull};

  int next = prompt_ids[0];
  int pos = 0;
  int prev = tokenizer_.bos_id();
  // Prefill: run every prompt token, sample only after the last one.
  for (size_t i = 0; i < prompt_ids.size(); ++i) {
    next = prompt_ids[i];
    const float* logits = forward_step(next, pos++);
    prev = next;
    if (i + 1 == prompt_ids.size()) {
      int id = sampling::sample(logits, c.base.vocab_size, sample_cfg_, rng,
                                sample_scratch_.data());
      out.push_back(id);
      if (on_token && !on_token(id, tokenizer_.decode(prev, id))) return out;
      prev = id;
      next = id;
    }
  }
  // Decode one token per step until max_steps or EOS.
  for (int s = 1; s < max_steps; ++s) {
    const float* logits = forward_step(next, pos++);
    int id = sampling::sample(logits, c.base.vocab_size, sample_cfg_, rng,
                              sample_scratch_.data());
    out.push_back(id);
    if (id == tokenizer_.eos_id()) break;
    if (on_token && !on_token(id, tokenizer_.decode(prev, id))) break;
    prev = id;
    next = id;
  }
  rng_state_ = rng[0];
  return out;
}

}  // namespace gemma2
}  // namespace inference
