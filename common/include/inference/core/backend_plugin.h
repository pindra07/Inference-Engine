#pragma once
// Plugin ABI. Core only depends on this; never on a concrete backend.
// See SPEC/04_PLUGIN_API.md.

#include <string>

namespace inference {
namespace core {

class IComputeBackend {
 public:
  virtual ~IComputeBackend() = default;

  virtual std::string name() const = 0;
  virtual std::string device_info() const = 0;

  // out = W @ x ; W is rows×cols row-major.
  virtual void matvec(const float* W, const float* x, float* out, int rows,
                      int cols) = 0;
  // out[i] = x[i]/sqrt(mean(x^2)+eps) * w[i]
  virtual void rmsnorm(const float* x, const float* w, float* out, int dim,
                       float eps) = 0;
  // in-place softmax
  virtual void softmax(float* x, int n) = 0;
  // RoPE rotation on q (n_heads*head_dim) and k (n_kv_heads*head_dim)
  virtual void rope(float* q, float* k, int head_dim, int n_heads,
                    int n_kv_heads, int pos, float theta) = 0;
  // in-place SiLU: x *= sigmoid(x)
  virtual void silu(float* x, int n) = 0;
  // Elementwise multiply: acc[i] = a[i]*b[i]. May alias (acc == a or b).
  virtual void elem_mul(float* acc, const float* a, const float* b,
                        int n) = 0;
  // Single-query causal attention at position pos.
  // att_buf scratch size >= (pos+1) floats.
  virtual void attention(const float* q, const float* k_cache,
                         const float* v_cache, float* out, float* att_buf,
                         int n_heads, int n_kv_heads, int head_dim,
                         int pos) = 0;

  // --- Gemma-family extensions (ABI v1.1; see SPEC/04, SPEC/10) ---
  // Soft-capped sliding-or-causal attention. scale is an EXPLICIT score
  // multiplier (not derived from head_dim). softcap <= 0 disables capping.
  // sliding_window <= 0 attends [0..pos], else [max(0,pos-window+1)..pos].
  // att_buf scratch: >= (pos+1) floats (only the visible prefix is used).
  virtual void attention_softcap(const float* q, const float* k_cache,
                                 const float* v_cache, float* out,
                                 float* att_buf, int n_heads, int n_kv_heads,
                                 int head_dim, int pos, float scale,
                                 float softcap, int sliding_window) = 0;
  // Elementwise x = cap*tanh(x/cap). No-op when cap <= 0.
  virtual void tanh_softcap(float* x, int n, float cap) = 0;
  // In-place tanh-approximate GELU, exactly HF's gelu_pytorch_tanh:
  // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
  virtual void gelu_tanh(float* x, int n) = 0;
};

// Every plugin shared-lib must export these two C symbols.
using CreateBackendFn = IComputeBackend* (*)();
using DestroyBackendFn = void (*)(IComputeBackend*);

}  // namespace core
}  // namespace inference
