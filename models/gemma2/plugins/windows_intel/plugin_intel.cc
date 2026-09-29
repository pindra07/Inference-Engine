// windows_intel plugin (Gemma2): x86-64 (Intel/AMD) backend.
//
// Subclasses core::ReferenceBackend. Accelerated: matvec/rmsnorm/elem_mul
// (AVX2+FMA, same kernels as the llama2 intel plugin — small intentional
// duplication keeps each model's plugin self-contained, SPEC/09).
// Inherited (scalar, correct): attention_softcap/softmax/gelu_tanh/
// tanh_softcap/rope/silu. matvec is ~95% of decode runtime, so the stranded
// scalar attention costs little (see SPEC/10).
// Builds everywhere: without __AVX2__ every override delegates to the base.

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include "inference/core/reference_backend.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace inference {
namespace gemma2 {
namespace plugin {
namespace intel {

namespace detail {

inline int hardware_threads() {
  unsigned n = std::thread::hardware_concurrency();
  return n ? (int)n : 4;
}

void matvec_rows(const float* W, const float* x, float* out, int r0, int r1,
                 int cols) {
#if defined(__AVX2__)
  for (int r = r0; r < r1; ++r) {
    const float* w = W + (size_t)r * cols;
    __m256 acc = _mm256_setzero_ps();
    int c = 0;
    for (; c + 8 <= cols; c += 8)
      acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + c), _mm256_loadu_ps(x + c),
                            acc);
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] +
                tmp[7];
    for (; c < cols; ++c) sum += w[c] * x[c];
    out[r] = sum;
  }
#else
  for (int r = r0; r < r1; ++r) {
    const float* w = W + (size_t)r * cols;
    float acc = 0.0f;
    for (int c = 0; c < cols; ++c) acc += w[c] * x[c];
    out[r] = acc;
  }
#endif
}

}  // namespace detail

class WindowsIntelBackend final : public core::ReferenceBackend {
 public:
  std::string name() const override { return "gemma2/windows_intel"; }
  std::string device_info() const override {
#if defined(__AVX2__)
    return "x86-64 AVX2+FMA + thread-pool (attention scalar)";
#else
    return "x86 scalar fallback (no AVX2 at compile time)";
#endif
  }

  void matvec(const float* W, const float* x, float* out, int rows,
              int cols) override {
    if (rows < 2048) {
      detail::matvec_rows(W, x, out, 0, rows, cols);
      return;
    }
    int nth = detail::hardware_threads();
    std::vector<std::thread> th;
    int chunk = (rows + nth - 1) / nth;
    for (int t = 0; t < nth; ++t) {
      int r0 = t * chunk, r1 = std::min(r0 + chunk, rows);
      if (r0 >= r1) break;
      th.emplace_back(detail::matvec_rows, W, x, out, r0, r1, cols);
    }
    for (auto& t : th) t.join();
  }

  void rmsnorm(const float* x, const float* w, float* out, int dim,
               float eps) override {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
      __m256 v = _mm256_loadu_ps(x + i);
      acc = _mm256_fmadd_ps(v, v, acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float ss = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] +
               tmp[7];
    for (; i < dim; ++i) ss += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ss / dim + eps);
    __m256 vinv = _mm256_set1_ps(inv);
    i = 0;
    for (; i + 8 <= dim; i += 8)
      _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(
                                                  _mm256_loadu_ps(x + i),
                                                  vinv),
                                              _mm256_loadu_ps(w + i)));
    for (; i < dim; ++i) out[i] = x[i] * inv * w[i];
#else
    ReferenceBackend::rmsnorm(x, w, out, dim, eps);
#endif
  }

  void elem_mul(float* acc, const float* a, const float* b, int n) override {
#if defined(__AVX2__)
    int i = 0;
    for (; i + 8 <= n; i += 8)
      _mm256_storeu_ps(acc + i, _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                              _mm256_loadu_ps(b + i)));
    for (; i < n; ++i) acc[i] = a[i] * b[i];
#else
    ReferenceBackend::elem_mul(acc, a, b, n);
#endif
  }
};

}  // namespace intel
}  // namespace plugin
}  // namespace gemma2
}  // namespace inference

extern "C" {
inference::core::IComputeBackend* CreateBackend() {
  return new inference::gemma2::plugin::intel::WindowsIntelBackend();
}
void DestroyBackend(inference::core::IComputeBackend* p) { delete p; }
}
