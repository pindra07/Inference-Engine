#pragma once
// Reference scalar backend: correct-first implementation of the full ABI.
//
// This is ALSO the base class for every hardware plugin: subclass it and
// override only the kernels you accelerate (see SPEC/04_PLUGIN_API.md).
// Shared kernels (rope/silu/...) live here once, so an optimization or fix
// applies to all plugins at the same time. Numerics match llama2.c
// (float accumulation, same op order) — the differential test gate.

#include <string>

#include "inference/core/backend_plugin.h"

namespace inference {
namespace core {

class ReferenceBackend : public IComputeBackend {
 public:
  ~ReferenceBackend() override = default;

  std::string name() const override { return "reference"; }
  std::string device_info() const override {
    return "portable scalar reference (common)";
  }

  void matvec(const float* W, const float* x, float* out, int rows,
              int cols) override;
  void rmsnorm(const float* x, const float* w, float* out, int dim,
               float eps) override;
  void softmax(float* x, int n) override;
  void rope(float* q, float* k, int head_dim, int n_heads, int n_kv_heads,
            int pos, float theta) override;
  void silu(float* x, int n) override;
  void elem_mul(float* acc, const float* a, const float* b, int n) override;
  void attention(const float* q, const float* k_cache, const float* v_cache,
                 float* out, float* att_buf, int n_heads, int n_kv_heads,
                 int head_dim, int pos) override;
  void attention_softcap(const float* q, const float* k_cache,
                         const float* v_cache, float* out, float* att_buf,
                         int n_heads, int n_kv_heads, int head_dim, int pos,
                         float scale, float softcap,
                         int sliding_window) override;
  void tanh_softcap(float* x, int n, float cap) override;
  void gelu_tanh(float* x, int n) override;
};

}  // namespace core
}  // namespace inference
