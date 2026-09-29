# 08 — Optimization Strategy & Decisions

Measured on: Apple M4 (arm64), 16 GB unified memory, Clang Release (`-O3`).
Workload: `stories15M.bin` (60 MB FP32), greedy, 200 tokens, prompt "Once upon a time".

## Baseline (before this pass)

| Backend | tok/s | Notes |
|---|---|---|
| built-in CPU / `cpu_baseline` plugin | ~177 | scalar, double accum in places |
| `mac_mseries` (Accelerate) | ~909 | cblas + vDSP + GCD |
| `windows_intel` (ARM build → scalar fallback) | ~319 | scalar + float accum |

## Performance model (why these numbers, and what actually matters)

Autoregressive decode reads **all weights once per token**. So:

```
tok/s  ≤  memory_bandwidth / bytes_per_token
```

- stories15M FP32 = ~60 MB/token. M4 streams well above 50 GB/s, so the
  ceiling is >800 tok/s — and indeed the Accelerate backend gets ~909.
  At this size we are **kernel-overhead bound**: transcendentals in RoPE,
  scalar reductions, virtual-call + threading overhead all show up.
- Llama-2-7B FP32 = ~26 GB/token. Same machine → ceiling ≈ 2–4 tok/s
  **no matter how good the kernels are**. At that size we are purely
  **bandwidth bound**, and the only big lever is *fewer bytes per token*
  (quantization), not faster math.

That split drives every decision below: kill overheads and duplication now
(small-model speed + readability), leave bytes-per-token to quantization
(`recommendation.md`, biggest future win).

## What changed in this pass (and why)

1. **`core::ReferenceBackend` (new, `core/reference_backend.h`)**
   Scalar implementation of the full ABI in *one* place. All three plugins
   now **subclass it and override only what they accelerate**
   (mac: matvec/rmsnorm/softmax/elem_mul/attention; intel: same set;
   cpu_baseline: nothing — ~15 lines). Before: the same rope/silu/softmax
   code existed 4× and had already diverged. One copy = one place to make
   the scalar path fast and correct.

2. **RoPE: hoist pair frequencies out of the head loop**
   `freq = theta^(-2i/head_dim)` depends only on the pair index and `pos`,
   not on the head — but the old code recomputed `powf+cosf+sinf` per pair
   **per head** (7B: 64 heads × 64 pairs = 4096 transcendental triples per
   layer per token). Now computed once per `rope()` call into small stack
   buffers, then pure multiply-add rotation per head. ~64× fewer
   transcendentals. Uses single-call `sincosf` where available
   (Apple/Linux), portable `sinf`+`cosf` fallback otherwise.

3. **Float (not double) accumulation in scalar kernels**
   Matches `llama2.c` exactly (our differential test still passes bit-exact
   to 1e-4) and lets the compiler auto-vectorize the scalar path with NEON.

4. **Transformer uses the backend for the SwiGLU multiply**
   `hb *= hb2` was a hand loop in `transformer.cc` while the ABI had
   `elem_mul_add` sitting unused. Now `backend->elem_mul(...)` is used, and
   the misleading name was fixed: `elem_mul_add` → **`elem_mul`**
   (it computes `acc = a*b`, never `+=`). ABI rename is safe: v0.1, no
   external plugins exist; `SPEC/04` updated.

5. **Sampler: `nth_element` for top-k**
   Old code `partial_sort`ed the whole shortlist even for small top-k.
   Now: `nth_element` partition + sort only the top-k window. Same
   sampling distribution, less work when `top_k << vocab`.

6. **`madvise(WILLNEED)` after `mmap`**
   Hints the OS to readahead weights; shortens time-to-first-token on cold
   page cache. No-op where unsupported (guarded by `#ifdef MADV_WILLNEED`).

7. **Dead code removed**: unused `core/tensor.h` views, unused
   `Engine::aligned_buf_`, internal `MakeCpuBackend` factory header
   (replaced by public `ReferenceBackend`), stateless `PluginLoader`
   simplified (handles intentionally live for process lifetime — see code
   comment; closing a dylib under a live backend is UB).

8. **Build hygiene**: global `CMAKE_POSITION_INDEPENDENT_CODE=ON`
   (required now that plugins link the static common lib — arm64 dylibs
   fault without PIC), Release stays `-O3`.

## Deliberately NOT done now (see `recommendation.md` for ranking)

- **Batched prefill / GEMM path** — biggest latency win for long prompts,
  but needs a `matmul` in the ABI + a batched transformer path. Correct
  first, fast second; queued as rec #2.
- **Quantization (INT8/INT4)** — the *only* thing that moves 7B-class
  tok/s meaningfully. Needs ABI capability flags + quant weight layout.
  Queued as rec #1.
- **Metal compute shaders / AMX-direct kernels** — Accelerate already
  saturates bandwidth on M-series for GEMV; custom shaders pay off for
  fused attention, not for matvec. Queued.
- **Fast-math approximations** (`expf`/`tanh` LUTs) — rejected for v1:
  changes numerics vs reference; keep bit-exact, revisit with a
  tolerance-gated test.

## Results (after this pass — same machine, same workload)

| Backend | Before | After | Δ | Why |
|---|---|---|---|---|
| built-in / cpu_baseline | ~177 tok/s | ~195 tok/s | **+10%** | float accum (autovec) + rope hoist |
| mac_mseries | ~909 tok/s | ~850–910 tok/s | ~0 (noise) | GEMV bandwidth-bound; rope was <1% here |
| windows_intel (scalar on ARM) | ~319 tok/s | ~337 tok/s | **+6%** | same scalar wins |

Correctness gate: greedy story on stories15M is **byte-identical** to
`llama2.c` output across all four backends (compared over the full
comparable range; only prompt-echo and step-count semantics differ).

Honest read: on Apple Silicon the Accelerate path was already at the
bandwidth ceiling, so kernel micro-opts can't move it. The durable wins
are structural — one scalar implementation for all plugins, float
accumulation matching the reference, and the folder layout in `09` —
plus the removal of ~170 lines of duplicated kernels. tok/s for 7B-class
models will only move with quantization (rec #1).
