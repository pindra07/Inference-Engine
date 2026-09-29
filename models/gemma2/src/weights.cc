// Gemma2 weight loader: GGM2 container (SPEC/10_GEMMA2.md).
// mmap on POSIX, malloc+fread fallback on Windows.

#include "inference/gemma2/weights.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace inference {
namespace gemma2 {
namespace {

struct RawHeader {
  char magic[4];      // "GGM2"
  int32_t version;    // 1
  int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, head_dim,
      vocab_size, seq_len, sliding_window;
  float rope_theta, rms_eps, attn_scale, attn_softcap, final_softcap;
};

}  // namespace

Weights::~Weights() {
#ifndef _WIN32
  if (map_base_ && map_base_ != MAP_FAILED) munmap(map_base_, map_bytes_);
  if (mmap_fd_ >= 0) ::close(mmap_fd_);
#else
  std::free(map_base_);
#endif
}

void Weights::bind_folded_views(int n_layers) {
  const int L = n_layers;
  // Fail fast: binding against a wrong-sized store (or an unpopulated
  // config) used to alias every norm view to layer 0 with zero diagnostics.
  if (L <= 0 || folded_.size() != (size_t)(4 * L + 1) * dim_)
    throw std::runtime_error("bind_folded_views: bad dims");
  float* f = folded_.data();
  g_in_ = f;
  f += (size_t)L * dim_;
  g_post_attn_ = f;
  f += (size_t)L * dim_;
  g_pre_ffn_ = f;
  f += (size_t)L * dim_;
  g_post_ffn_ = f;
  f += (size_t)L * dim_;
  final_rms_ = f;
}

Weights::Weights(Weights&& o) noexcept
    : config_(o.config_),
      dim_(o.dim_),
      hidden_(o.hidden_),
      q_dim_(o.q_dim_),
      kv_dim_(o.kv_dim_),
      map_base_(o.map_base_),
      map_bytes_(o.map_bytes_),
      mmap_fd_(o.mmap_fd_),
      data_(o.data_),
      num_floats_(o.num_floats_),
      g_wq_(o.g_wq_),
      g_wk_(o.g_wk_),
      g_wv_(o.g_wv_),
      g_wo_(o.g_wo_),
      g_wgate_(o.g_wgate_),
      g_wdown_(o.g_wdown_),
      g_wup_(o.g_wup_),
      folded_(std::move(o.folded_)),
      wcls_(o.wcls_) {
  o.map_base_ = nullptr;
  o.map_bytes_ = 0;
  o.mmap_fd_ = -1;
  o.data_ = nullptr;
  // config_ was moved above, so its dims are valid here.
  bind_folded_views(config_.base.n_layers);
}

Weights& Weights::operator=(Weights&& o) noexcept {
  if (this != &o) {
    this->~Weights();
    new (this) Weights(std::move(o));
  }
  return *this;
}

void Weights::load(const std::string& path) {
  RawHeader h{};
  const float* pool = nullptr;
  size_t n_floats = 0;

#ifndef _WIN32
  mmap_fd_ = ::open(path.c_str(), O_RDONLY);
  if (mmap_fd_ < 0) throw std::runtime_error("open failed: " + path);
  struct stat st{};
  if (fstat(mmap_fd_, &st) != 0) throw std::runtime_error("stat failed: " + path);
  map_bytes_ = (size_t)st.st_size;
  if (map_bytes_ < sizeof(RawHeader))
    throw std::runtime_error("checkpoint too small: " + path);
  map_base_ = mmap(nullptr, map_bytes_, PROT_READ, MAP_PRIVATE, mmap_fd_, 0);
  if (map_base_ == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
#ifdef MADV_WILLNEED
  madvise(map_base_, map_bytes_, MADV_WILLNEED);
#endif
  memcpy(&h, map_base_, sizeof(h));
  pool = reinterpret_cast<const float*>((const char*)map_base_ + sizeof(h));
  n_floats = (map_bytes_ - sizeof(h)) / sizeof(float);
#else
  FILE* f = nullptr;
  if (fopen_s(&f, path.c_str(), "rb") != 0 || !f)
    throw std::runtime_error("open failed: " + path);
  if (fread(&h, sizeof(h), 1, f) != 1) {
    fclose(f);
    throw std::runtime_error("header read failed: " + path);
  }
  fseek(f, 0, SEEK_END);
  long total = ftell(f);
  fseek(f, sizeof(h), SEEK_SET);
  n_floats = ((size_t)total - sizeof(h)) / sizeof(float);
  float* buf = (float*)std::malloc(n_floats * sizeof(float));
  if (!buf) {
    fclose(f);
    throw std::runtime_error("oom for weights");
  }
  if (fread(buf, sizeof(float), n_floats, f) != n_floats) {
    std::free(buf);
    fclose(f);
    throw std::runtime_error("weight read failed: " + path);
  }
  fclose(f);
  map_base_ = buf;
  map_bytes_ = (size_t)total;
  pool = buf;
#endif

  if (memcmp(h.magic, "GGM2", 4) != 0)
    throw std::runtime_error("bad magic (not a GGM2 file): " + path);
  if (h.version != 1)
    throw std::runtime_error("unsupported GGM2 version: " + path);

  ModelConfig c;
  c.base.dim = h.dim;
  c.base.hidden_dim = h.hidden_dim;
  c.base.n_layers = h.n_layers;
  c.base.n_heads = h.n_heads;
  c.base.n_kv_heads = h.n_kv_heads;
  c.base.vocab_size = h.vocab_size;
  c.base.seq_len = h.seq_len;
  c.base.rope_theta = h.rope_theta;
  c.head_dim = h.head_dim;
  c.sliding_window = h.sliding_window;
  c.attn_scale = h.attn_scale;
  c.attn_softcap = h.attn_softcap;
  c.final_softcap = h.final_softcap;
  c.rms_eps = h.rms_eps;
  c.base.shared_classifier = false;  // decided by trailing size below
  if (!c.valid()) throw std::runtime_error("invalid header in: " + path);

  const int L = c.base.n_layers;
  const int dim = c.base.dim, hidden = c.base.hidden_dim;
  const int q_dim = c.q_dim(), kv_dim = c.kv_dim();
  dim_ = dim;
  hidden_ = hidden;
  q_dim_ = q_dim;
  kv_dim_ = kv_dim;
  size_t off = 0;
  auto take = [&](size_t n, const char* what) -> const float* {
    if (off + n > n_floats)
      throw std::runtime_error(std::string("checkpoint truncated (") + what +
                               "): " + path);
    const float* p = pool + off;
    off += n;
    return p;
  };
  // Folded (1+w) norms live in owned storage; everything else views mmap.
  auto take_folded = [&](float* dst, size_t n, const char* what) {
    const float* src = take(n, what);
    for (size_t i = 0; i < n; ++i) dst[i] = 1.0f + src[i];
  };
  data_ = pool;
  num_floats_ = n_floats;
  folded_.assign((size_t)(4 * L + 1) * dim, 0.0f);
  bind_folded_views(L);

  take((size_t)c.base.vocab_size * dim, "token_embedding");
  float* f = folded_.data();
  for (int l = 0; l < L; ++l)  // group order must match exporter
    take_folded(f + (size_t)l * dim, dim, "input_rms");
  g_wq_ = take((size_t)L * q_dim * dim, "Wq");
  g_wk_ = take((size_t)L * kv_dim * dim, "Wk");
  g_wv_ = take((size_t)L * kv_dim * dim, "Wv");
  g_wo_ = take((size_t)L * dim * q_dim, "Wo");
  f += (size_t)L * dim;
  for (int l = 0; l < L; ++l) take_folded(f + (size_t)l * dim, dim, "post_attn");
  f += (size_t)L * dim;
  for (int l = 0; l < L; ++l) take_folded(f + (size_t)l * dim, dim, "pre_ffn");
  g_wgate_ = take((size_t)L * hidden * dim, "Wgate");
  g_wdown_ = take((size_t)L * dim * hidden, "Wdown");
  g_wup_ = take((size_t)L * hidden * dim, "Wup");
  f += (size_t)L * dim;
  for (int l = 0; l < L; ++l) take_folded(f + (size_t)l * dim, dim, "post_ffn");
  f += (size_t)L * dim;
  take_folded(f, dim, "final_rms");

  // Trailing wcls iff untied (Gemma ties by default).
  const size_t cls_n = (size_t)c.base.vocab_size * dim;
  const size_t rest = n_floats - off;
  wcls_ = nullptr;
  if (rest == 0) {
    c.base.shared_classifier = true;
  } else if (rest == cls_n) {
    c.base.shared_classifier = false;
    wcls_ = take(cls_n, "wcls");
  } else {
    throw std::runtime_error("checkpoint has unexpected trailing size (" +
                             std::to_string(rest) + " floats): " + path);
  }
  config_ = c;
}

}  // namespace gemma2
}  // namespace inference
