// mac_mseries plugin (Llama-2): Apple Silicon (M1/M2/M3/M4) backend.
//
// Subclasses core::ReferenceBackend — only accelerated kernels are
// overridden (rope/silu inherit the shared optimized implementation).
//
// HW-NOTE:
// - matvec  -> Accelerate cblas_sgemv (AMX/NEON, tuned for unified memory).
// - rmsnorm/softmax/elem_mul -> vDSP vector ops.
// - attention -> per-head vDSP dot/axpy + GCD dispatch_apply fan-out.
// - Builds (scalar-correct) on non-Apple platforms via the base class.

#include <cmath>
#include <string>

#include "inference/core/reference_backend.h"

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#endif

namespace inference {
namespace llama2 {
namespace plugin {
namespace mac {

class MacMSeriesBackend final : public core::ReferenceBackend {
 public:
  std::string name() const override { return "llama2/mac_mseries"; }
  std::string device_info() const override {
#if defined(__APPLE__) && defined(__arm64__)
    return "Apple Silicon + Accelerate (cblas/vDSP/GCD)";
#elif defined(__APPLE__)
    return "Apple Accelerate (non-arm64 build)";
#else
    return "mac_mseries fallback (non-Apple build, scalar)";
#endif
  }

  void matvec(const float* W, const float* x, float* out, int rows,
              int cols) override {
#if defined(__APPLE__)
    // HW-NOTE: row-major sgemv, alpha=1 beta=0. AMX handles the blocking.
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0f, W, cols, x, 1,
                0.0f, out, 1);
#else
    ReferenceBackend::matvec(W, x, out, rows, cols);
#endif
  }

  void rmsnorm(const float* x, const float* w, float* out, int dim,
               float eps) override {
#if defined(__APPLE__)
    // HW-NOTE: vDSP_svesq = sum(x^2) in one NEON pass.
    float ss = 0.0f;
    vDSP_svesq(const_cast<float*>(x), 1, &ss, (vDSP_Length)dim);
    float inv = 1.0f / std::sqrt(ss / dim + eps);
    vDSP_vsmul(const_cast<float*>(x), 1, &inv, out, 1, (vDSP_Length)dim);
    vDSP_vmul(out, 1, const_cast<float*>(w), 1, out, 1, (vDSP_Length)dim);
#else
    ReferenceBackend::rmsnorm(x, w, out, dim, eps);
#endif
  }

  void softmax(float* x, int n) override {
#if defined(__APPLE__)
    float mx = x[0];
    vDSP_maxv(x, 1, &mx, (vDSP_Length)n);
    float neg = -mx;
    vDSP_vsadd(x, 1, &neg, x, 1, (vDSP_Length)n);
    int nn = n;
    vvexpf(x, x, &nn);  // HW-NOTE: vectorized exp
    float sum = 0;
    vDSP_sve(x, 1, &sum, (vDSP_Length)n);
    float inv = 1.0f / sum;
    vDSP_vsmul(x, 1, &inv, x, 1, (vDSP_Length)n);
#else
    ReferenceBackend::softmax(x, n);
#endif
  }

  void elem_mul(float* acc, const float* a, const float* b, int n) override {
#if defined(__APPLE__)
    vDSP_vmul(const_cast<float*>(a), 1, const_cast<float*>(b), 1, acc, 1,
              (vDSP_Length)n);
#else
    ReferenceBackend::elem_mul(acc, a, b, n);
#endif
  }

  void attention(const float* q, const float* k_cache, const float* v_cache,
                 float* out, float* att_buf, int n_heads, int n_kv_heads,
                 int head_dim, int pos) override {
    const int mult = n_heads / n_kv_heads;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    auto one_head = [&](int h) {
      const float* qq = q + (size_t)h * head_dim;
      float* att = att_buf + (size_t)h * (pos + 1);
      const int kv_h = h / mult;
      for (int t = 0; t <= pos; ++t) {
        const float* kk = k_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
#if defined(__APPLE__)
        float s = 0;
        vDSP_dotpr(const_cast<float*>(qq), 1, const_cast<float*>(kk), 1, &s,
                   (vDSP_Length)head_dim);
        att[t] = s * scale;
#else
        float s = 0;
        for (int i = 0; i < head_dim; ++i) s += qq[i] * kk[i];
        att[t] = s * scale;
#endif
      }
      softmax(att, pos + 1);  // virtual: uses the vDSP override on Apple
      float* oo = out + (size_t)h * head_dim;
      for (int i = 0; i < head_dim; ++i) oo[i] = 0.0f;
      for (int t = 0; t <= pos; ++t) {
        const float* vv = v_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
        float p = att[t];
#if defined(__APPLE__)
        vDSP_vsma(const_cast<float*>(vv), 1, &p, oo, 1, oo, 1,
                  (vDSP_Length)head_dim);  // oo += p*vv
#else
        for (int i = 0; i < head_dim; ++i) oo[i] += p * vv[i];
#endif
      }
    };
#if defined(__APPLE__)
    // HW-NOTE: heads are independent -> GCD fan-out across P-cores.
    dispatch_apply((size_t)n_heads,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t h) { one_head((int)h); });
#else
    for (int h = 0; h < n_heads; ++h) one_head(h);
#endif
  }
};

}  // namespace mac
}  // namespace plugin
}  // namespace llama2
}  // namespace inference

extern "C" {
inference::core::IComputeBackend* CreateBackend() {
  return new inference::llama2::plugin::mac::MacMSeriesBackend();
}
void DestroyBackend(inference::core::IComputeBackend* p) { delete p; }
}
