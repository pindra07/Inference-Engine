# 04 — Plugin API (v1, frozen)

## ABI
Each plugin is a shared library exporting exactly two C symbols:

```cpp
extern "C" {
  inference::core::IComputeBackend* CreateBackend();
  void DestroyBackend(inference::core::IComputeBackend* p);
}
```

`PluginLoader` (`common/include/inference/core/plugin_loader.h`) wraps
`dlopen` (POSIX) / `LoadLibrary` (Windows). Handles stay open for process
lifetime (closing a dylib under a live backend is UB).

## Interface (`common/include/inference/core/backend_plugin.h`)

```cpp
class IComputeBackend {
 public:
  virtual ~IComputeBackend() = default;
  virtual std::string name() const = 0;          // e.g. "llama2/mac_mseries"
  virtual std::string device_info() const = 0;

  // --- bulk ops (hot) ---
  virtual void matvec(const float* W, const float* x, float* out,
                      int rows, int cols) = 0;   // out = W @ x, row-major W
  virtual void rmsnorm(const float* x, const float* w, float* out,
                       int dim, float eps) = 0;
  virtual void softmax(float* x, int n) = 0;     // in-place
  virtual void rope(float* q, float* k, int head_dim, int n_heads,
                    int n_kv_heads, int pos, float theta) = 0;
  virtual void silu(float* x, int n) = 0;        // in-place sigmoid(x)*x
  virtual void elem_mul(float* acc, const float* a, const float* b,
                        int n) = 0;              // acc = a*b, may alias
  virtual void attention(const float* q, const float* k_cache,
                         const float* v_cache, float* out, float* att_buf,
                         int n_heads, int n_kv_heads, int head_dim,
                         int pos) = 0;
};
```

- Engine allocates all buffers 64-byte aligned.
- `attention`: single-query causal SDPA over `cache[0..pos]`,
  `1/sqrt(head_dim)` scaling, GQA repeat (`kv_head = head/(H/Hkv)`).
- v1 ABI is **frozen**. New kernels (e.g. `matmul` for batched prefill,
  quantized matvec) arrive with `Capability` query + mixin defaults so old
  plugins keep loading (see `recommendation.md`).

## The normal way to write a plugin: subclass `ReferenceBackend`

`core::ReferenceBackend` (`common/.../reference_backend.h`) implements the
whole ABI in portable scalar C++ (numerics match `llama2.c`). A plugin
overrides **only what it accelerates**:

```cpp
class MacMSeriesBackend final : public core::ReferenceBackend {
  std::string name() const override { return "llama2/mac_mseries"; }
  void matvec(...) override { /* Accelerate */ }
  void rmsnorm(...) override { /* vDSP */ }
  // rope/silu/softmax/attention... inherited unless overridden
};
```

Rules: keep overrides numerically within 1e-4 of the reference (the
correctness gate); non-Apple/non-AVX2 builds delegate to the base so every
plugin compiles everywhere.

## How to add a new plugin (for model `<m>`, hardware `<h>`)
1. Copy `models/<m>/plugins/cpu_baseline/` to `models/<m>/plugins/<h>/`.
2. Subclass `core::ReferenceBackend`, override accelerated kernels.
3. Export `CreateBackend` / `DestroyBackend` (copy the 6 lines at the file end).
4. Register with `llama2_plugin(...)` (or the model's equivalent) in
   `models/<m>/CMakeLists.txt`.
5. Select at runtime: `--backend <path>` or `INFERENCE_BACKEND` env.

## HW notes (llama2)
- **mac_mseries**: `cblas_sgemv` matvec, `vDSP` norm/softmax/elem_mul,
  GCD `dispatch_apply` over heads.
- **windows_intel**: AVX2+FMA + row/head thread pools (scalar base on ARM).
- **cpu_baseline**: zero overrides — the reference, repackaged.

## ABI v1.1 additions (Gemma-family; v1's 7 methods untouched)

```cpp
  // Soft-capped sliding/causal attention. scale is EXPLICIT (Gemma's scale
  // is 1/sqrt(query_pre_attn_scalar), not 1/sqrt(head_dim)).
  // softcap <= 0 disables capping; window <= 0 = full causal [0..pos].
  virtual void attention_softcap(const float* q, const float* k_cache,
                                 const float* v_cache, float* out,
                                 float* att_buf, int n_heads, int n_kv_heads,
                                 int head_dim, int pos, float scale,
                                 float softcap, int sliding_window) = 0;
  virtual void tanh_softcap(float* x, int n, float cap) = 0;  // x=cap*tanh(x/cap)
  virtual void gelu_tanh(float* x, int n) = 0;  // exact HF gelu_pytorch_tanh
```

`ReferenceBackend` implements all three scalar, so existing plugins inherit
them with zero changes (llama2 plugins needed none). Model plugins override
for speed where it matters (gemma2/mac: `attention_softcap` + `tanh_softcap`
via vForce; gemma2/intel: inherits attention, accelerates matvec). Rule for
future additions: same pattern — scalar in `ReferenceBackend`, override
selectively, verify against the mixin within 1e-4.
