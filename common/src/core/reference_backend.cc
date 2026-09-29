// Reference scalar kernels shared by all plugins (via inheritance).
// Correctness matches llama2.c: float accumulation, same op order.

#include "inference/core/reference_backend.h"

#include <cmath>
#include <vector>

#if defined(__linux__)
#define INFERENCE_SINCOSF sincosf  // one libm call instead of sin+cos
#elif defined(__APPLE__)
#define INFERENCE_SINCOSF __sincosf  // Apple SDK name for the same routine
#endif

namespace inference {
namespace core {

void ReferenceBackend::matvec(const float* W, const float* x, float* out,
                              int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* w = W + (size_t)r * cols;
    float acc = 0.0f;
    for (int c = 0; c < cols; ++c) acc += w[c] * x[c];
    out[r] = acc;
  }
}

void ReferenceBackend::rmsnorm(const float* x, const float* w, float* out,
                               int dim, float eps) {
  float ss = 0.0f;
  for (int i = 0; i < dim; ++i) ss += x[i] * x[i];
  float inv = 1.0f / std::sqrt(ss / dim + eps);
  for (int i = 0; i < dim; ++i) out[i] = x[i] * inv * w[i];
}

void ReferenceBackend::softmax(float* x, int n) {
  float mx = x[0];
  for (int i = 1; i < n; ++i) mx = std::max(mx, x[i]);
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - mx);
    sum += x[i];
  }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

void ReferenceBackend::rope(float* q, float* k, int head_dim, int n_heads,
                            int n_kv_heads, int pos, float theta) {
  // OPT (SPEC/08): pair frequencies depend only on (pair, pos), NOT on the
  // head — compute them once per call instead of once per head (~64x fewer
  // transcendentals on 7B). head_dim <= 256 in practice; heap fallback for
  // exotic sizes so this can never overflow the stack buffers.
  const int half = head_dim / 2;
  float cos_stack[128];
  float sin_stack[128];
  std::vector<float> heap;  // [cos | sin], only if half > 128
  float* cos_tab = cos_stack;
  float* sin_tab = sin_stack;
  if (half > 128) {
    heap.resize((size_t)2 * half);
    cos_tab = heap.data();
    sin_tab = heap.data() + half;
  }
  for (int i = 0; i < half; ++i) {
    float freq = 1.0f / std::pow(theta, (float)(2 * i) / (float)head_dim);
    float val = (float)pos * freq;
#ifdef INFERENCE_SINCOSF
    INFERENCE_SINCOSF(val, &sin_tab[i], &cos_tab[i]);
#else
    sin_tab[i] = std::sin(val);
    cos_tab[i] = std::cos(val);
#endif
  }
  auto rot = [&](float* v, int heads) {
    for (int h = 0; h < heads; ++h) {
      float* p = v + (size_t)h * head_dim;
      for (int i = 0; i < half; ++i) {
        float v0 = p[2 * i], v1 = p[2 * i + 1];
        p[2 * i] = v0 * cos_tab[i] - v1 * sin_tab[i];
        p[2 * i + 1] = v0 * sin_tab[i] + v1 * cos_tab[i];
      }
    }
  };
  rot(q, n_heads);
  rot(k, n_kv_heads);
}

void ReferenceBackend::silu(float* x, int n) {
  for (int i = 0; i < n; ++i) x[i] *= 1.0f / (1.0f + std::exp(-x[i]));
}

void ReferenceBackend::elem_mul(float* acc, const float* a, const float* b,
                                int n) {
  for (int i = 0; i < n; ++i) acc[i] = a[i] * b[i];
}

void ReferenceBackend::attention(const float* q, const float* k_cache,
                                const float* v_cache, float* out,
                                float* att_buf, int n_heads, int n_kv_heads,
                                int head_dim, int pos) {  const int mult = n_heads / n_kv_heads;  // GQA repeat factor (1 for MHA)
  const float scale = 1.0f / std::sqrt((float)head_dim);
  for (int h = 0; h < n_heads; ++h) {
    const float* qq = q + (size_t)h * head_dim;
    float* att = att_buf + (size_t)h * (pos + 1);
    const int kv_h = h / mult;
    for (int t = 0; t <= pos; ++t) {
      const float* kk = k_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
      float s = 0.0f;
      for (int i = 0; i < head_dim; ++i) s += qq[i] * kk[i];
      att[t] = s * scale;
    }
    softmax(att, pos + 1);  // virtual: accelerated backends reuse their own
    float* oo = out + (size_t)h * head_dim;
    for (int i = 0; i < head_dim; ++i) oo[i] = 0.0f;
    for (int t = 0; t <= pos; ++t) {
      const float* vv = v_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
      float p = att[t];
      for (int i = 0; i < head_dim; ++i) oo[i] += p * vv[i];
    }
  }
}

void ReferenceBackend::attention_softcap(
    const float* q, const float* k_cache, const float* v_cache, float* out,
    float* att_buf, int n_heads, int n_kv_heads, int head_dim, int pos,
    float scale, float softcap, int sliding_window) {
  const int mult = n_heads / n_kv_heads;  // GQA repeat factor (1 for MHA)
  const int lo = sliding_window > 0 ? std::max(0, pos - sliding_window + 1)
                                    : 0;
  const int m = pos - lo + 1;  // visible positions
  for (int h = 0; h < n_heads; ++h) {
    const float* qq = q + (size_t)h * head_dim;
    float* att = att_buf + (size_t)h * (pos + 1);
    const int kv_h = h / mult;
    for (int j = 0; j < m; ++j) {
      const int t = lo + j;
      const float* kk = k_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
      float s = 0.0f;
      for (int i = 0; i < head_dim; ++i) s += qq[i] * kk[i];
      s *= scale;
      if (softcap > 0.0f) s = softcap * std::tanh(s / softcap);
      att[j] = s;
    }
    softmax(att, m);
    float* oo = out + (size_t)h * head_dim;
    for (int i = 0; i < head_dim; ++i) oo[i] = 0.0f;
    for (int j = 0; j < m; ++j) {
      const int t = lo + j;
      const float* vv = v_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
      float p = att[j];
      for (int i = 0; i < head_dim; ++i) oo[i] += p * vv[i];
    }
  }
}

void ReferenceBackend::tanh_softcap(float* x, int n, float cap) {
  if (cap <= 0.0f) return;
  for (int i = 0; i < n; ++i) x[i] = cap * std::tanh(x[i] / cap);
}

void ReferenceBackend::gelu_tanh(float* x, int n) {
  // Exact HF gelu_pytorch_tanh (keep formula + constants identical).
  const float c = 0.7978845608028654f;  // sqrt(2/pi), literal for MSVC (no M_PI)
  for (int i = 0; i < n; ++i) {
    float v = x[i];
    float inner = c * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.0f + std::tanh(inner));
  }
}

}  // namespace core
}  // namespace inference
