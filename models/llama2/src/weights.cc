// Portable weight loader: mmap on POSIX, malloc+fread fallback elsewhere.
// Format: Karpathy llama2.bin — int32[7] {dim, hidden, layers, heads,
// kv_heads, vocab_size, seq_len} then FP32 weights in fixed order.
// Negative vocab_size => shared classifier (use abs).

#include "inference/llama2/weights.h"

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
namespace llama2 {
namespace {

struct RawHeader {
  int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
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

Weights::Weights(Weights&& o) noexcept
    : config_(o.config_),
      dim_(o.dim_),
      hidden_(o.hidden_),
      kv_dim_(o.kv_dim_),
      map_base_(o.map_base_),
      map_bytes_(o.map_bytes_),
      mmap_fd_(o.mmap_fd_),
      data_(o.data_),
      num_floats_(o.num_floats_),
      g_attn_rms_(o.g_attn_rms_),
      g_wq_(o.g_wq_),
      g_wk_(o.g_wk_),
      g_wv_(o.g_wv_),
      g_wo_(o.g_wo_),
      g_ffn_rms_(o.g_ffn_rms_),
      g_w1_(o.g_w1_),
      g_w2_(o.g_w2_),
      g_w3_(o.g_w3_),
      final_rms_(o.final_rms_),
      wcls_(o.wcls_) {
  o.map_base_ = nullptr;
  o.map_bytes_ = 0;
  o.mmap_fd_ = -1;
  o.data_ = nullptr;
}

Weights& Weights::operator=(Weights&& o) noexcept {
  if (this != &o) {
    this->~Weights();
    new (this) Weights(std::move(o));
  }
  return *this;
}

void Weights::load(const std::string& path, int vocab_override,
                   int seq_override) {
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
  if (map_base_ == MAP_FAILED)
    throw std::runtime_error("mmap failed: " + path);
#ifdef MADV_WILLNEED
  // OPT (SPEC/08): hint readahead — shortens time-to-first-token on cold cache.
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

  core::ModelConfig c;
  c.dim = h.dim;
  c.hidden_dim = h.hidden_dim;
  c.n_layers = h.n_layers;
  c.n_heads = h.n_heads;
  c.n_kv_heads = h.n_kv_heads == 0 ? h.n_heads : h.n_kv_heads;
  c.shared_classifier = h.vocab_size < 0;
  c.vocab_size = h.vocab_size < 0 ? -h.vocab_size : h.vocab_size;
  c.seq_len = h.seq_len;
  if (vocab_override > 0) c.vocab_size = vocab_override;
  if (seq_override > 0) c.seq_len = seq_override;
  if (!c.valid()) throw std::runtime_error("invalid header in: " + path);

  // Bind views GROUPED BY TENSOR (matches llama2.c memory_map_weights):
  //   token_embedding, ALL attn_rms, ALL Wq, ALL Wk, ALL Wv, ALL Wo,
  //   ALL ffn_rms, ALL W1, ALL W2, ALL W3, final_rms,
  //   [legacy freq_cis_real/imag — skipped, RoPE computed on the fly]
  //   [wcls unless shared].
  // Layout variants (all Karpathy-compatible):
  // - New exports: no freq tables; negative vocab_size => shared classifier.
  // - Old exports (e.g. tinyllamas stories*.bin): freq_cis_real/imag
  //   [seq_len × head_dim/2] each are stored after final_rms, and wcls is
  //   omitted (shared) even though vocab_size is positive.
  // Trailing layout is detected from the file size.
  const int L = c.n_layers;
  const int dim = c.dim, hidden = c.hidden_dim, kv_dim = c.kv_dim();
  const int head_dim = c.head_dim();
  dim_ = dim;
  hidden_ = hidden;
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
  data_ = pool;
  num_floats_ = n_floats;
  take((size_t)c.vocab_size * dim, "token_embedding");
  g_attn_rms_ = take((size_t)L * dim, "attn_rms");
  g_wq_ = take((size_t)L * dim * dim, "Wq");
  g_wk_ = take((size_t)L * dim * kv_dim, "Wk");
  g_wv_ = take((size_t)L * dim * kv_dim, "Wv");
  g_wo_ = take((size_t)L * dim * dim, "Wo");
  g_ffn_rms_ = take((size_t)L * dim, "ffn_rms");
  g_w1_ = take((size_t)L * hidden * dim, "W1");
  g_w2_ = take((size_t)L * dim * hidden, "W2");
  g_w3_ = take((size_t)L * hidden * dim, "W3");
  final_rms_ = take(dim, "final_rms");

  const size_t freq_n = (size_t)c.seq_len * (head_dim / 2);  // per table
  const size_t cls_n = (size_t)c.vocab_size * dim;
  const size_t rest = n_floats - off;
  wcls_ = nullptr;
  if (rest == 0) {
    c.shared_classifier = true;  // shared, modern layout
  } else if (rest == cls_n) {
    c.shared_classifier = false;  // separate classifier, modern layout
    wcls_ = take(cls_n, "wcls");
  } else if (rest == 2 * freq_n) {
    c.shared_classifier = true;  // shared, legacy layout with freq tables
    take(freq_n, "freq_cis_real");  // skipped: RoPE computed on the fly
    take(freq_n, "freq_cis_imag");
  } else if (rest == 2 * freq_n + cls_n) {
    c.shared_classifier = false;  // unshared, legacy layout
    take(freq_n, "freq_cis_real");
    take(freq_n, "freq_cis_imag");
    wcls_ = take(cls_n, "wcls");
  } else {
    throw std::runtime_error("checkpoint has unexpected trailing size (" +
                             std::to_string(rest) + " floats): " + path);
  }
  config_ = c;
}

}  // namespace llama2
}  // namespace inference
