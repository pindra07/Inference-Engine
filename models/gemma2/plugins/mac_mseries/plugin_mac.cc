// mac_mseries plugin (Gemma2): Apple Silicon accelerated backend.
//
// Subclasses core::ReferenceBackend — only accelerated kernels are
// overridden (rope/silu/gelu_tanh inherit the shared implementation).
//
// HW-NOTE:
// - matvec/rmsnorm/softmax/elem_mul -> Accelerate (cblas/vDSP), as in llama2.
// - attention_softcap -> per-head vDSP dot/axpy + GCD fan-out, with the
//   Gemma softcap + sliding-window mask folded into the score loop.
// - tanh_softcap -> vForce vvtanhf (vector tanh).
// - Builds (scalar-correct) on non-Apple platforms via the base class.

#include <cmath>
#include <string>

#include "inference/core/reference_backend.h"

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#endif

namespace inference {
namespace gemma2 {
namespace plugin {
namespace mac {

class MacMSeriesBackend final : public core::ReferenceBackend {
 public:
  std::string name() const override { return "gemma2/mac_mseries"; }
  std::string device_info() const override {
#if defined(__APPLE__) && defined(__arm64__)
    return "Apple Silicon + Accelerate (cblas/vDSP/vForce/GCD)";
#elif defined(__APPLE__)
    return "Apple Accelerate (non-arm64 build)";
#else
    return "mac_mseries fallback (non-Apple build, scalar)";
#endif
  }

  void matvec(const float* W, const float* x, float* out, int rows,
              int cols) override {
#if defined(__APPLE__)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0f, W, cols, x, 1,
                0.0f, out, 1);
#else
    ReferenceBackend::matvec(W, x, out, rows, cols);
#endif
  }

  void rmsnorm(const float* x, const float* w, float* out, int dim,
               float eps) override {
#if defined(__APPLE__)
    // Same kernel as llama2: Gemma's (1+w) is pre-folded by the loader,
    // so no ABI or kernel change was needed (SPEC/10).
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
    vvexpf(x, x, &nn);
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

  void attention_softcap(const float* q, const float* k_cache,
                         const float* v_cache, float* out, float* att_buf,
                         int n_heads, int n_kv_heads, int head_dim, int pos,
                         float scale, float softcap,
                         int sliding_window) override {
    const int mult = n_heads / n_kv_heads;
    const int lo = sliding_window > 0 ? std::max(0, pos - sliding_window + 1)
                                      : 0;
    const int m = pos - lo + 1;
    auto one_head = [&](int h) {
      const float* qq = q + (size_t)h * head_dim;
      float* att = att_buf + (size_t)h * (pos + 1);
      const int kv_h = h / mult;
      for (int j = 0; j < m; ++j) {
        const int t = lo + j;
        const float* kk = k_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
#if defined(__APPLE__)
        float s = 0;
        vDSP_dotpr(const_cast<float*>(qq), 1, const_cast<float*>(kk), 1, &s,
                   (vDSP_Length)head_dim);
        s *= scale;
#else
        float s = 0;
        for (int i = 0; i < head_dim; ++i) s += qq[i] * kk[i];
        s *= scale;
#endif
        if (softcap > 0.0f) s = softcap * std::tanh(s / softcap);
        att[j] = s;
      }
      softmax(att, m);  // virtual: vDSP override on Apple
      float* oo = out + (size_t)h * head_dim;
      for (int i = 0; i < head_dim; ++i) oo[i] = 0.0f;
      for (int j = 0; j < m; ++j) {
        const int t = lo + j;
        const float* vv = v_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
        float p = att[j];
#if defined(__APPLE__)
        vDSP_vsma(const_cast<float*>(vv), 1, &p, oo, 1, oo, 1,
                  (vDSP_Length)head_dim);
#else
        for (int i = 0; i < head_dim; ++i) oo[i] += p * vv[i];
#endif
      }
    };
#if defined(__APPLE__)
    dispatch_apply((size_t)n_heads,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t h) { one_head((int)h); });
#else
    for (int h = 0; h < n_heads; ++h) one_head(h);
#endif
  }

  void tanh_softcap(float* x, int n, float cap) override {
#if defined(__APPLE__)
    if (cap <= 0.0f) return;
    // x = cap*tanh(x/cap): scale, vector tanh, scale back.
    float inv = 1.0f / cap;
    vDSP_vsmul(x, 1, &inv, x, 1, (vDSP_Length)n);
    int nn = n;
    vvtanhf(x, x, &nn);  // HW-NOTE: vForce vector tanh
    vDSP_vsmul(x, 1, &cap, x, 1, (vDSP_Length)n);
#else
    ReferenceBackend::tanh_softcap(x, n, cap);
#endif
  }
};

}  // namespace mac
}  // namespace plugin
}  // namespace gemma2
}  // namespace inference

extern "C" {
inference::core::IComputeBackend* CreateBackend() {
  return new inference::gemma2::plugin::mac::MacMSeriesBackend();
}
void DestroyBackend(inference::core::IComputeBackend* p) { delete p; }
}
