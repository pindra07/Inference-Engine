# Recommendation — what to do next, ranked

Basis: Apple M4 measurements in `08` (stories15M: ref ~195, intel ~337,
mac ~850 tok/s; 7B-FP32 is bandwidth-capped at ~2–4 tok/s no matter what).
Rank = impact ÷ effort. Each item says *what*, *why now/later*, and the
concrete first step.

## P0 — Performance (bytes per token dominate everything)

| # | What | Impact | Effort | First step |
|---|---|---|---|---|
| 1 | **INT8 weight quantization** (then INT4) | 4× tok/s on 7B-class; only lever that matters at scale | L | ABI: `Capability query()` + `matvec_q8(const BlockQ8* ...)`; layout: grouped-by-tensor with per-row scales (mirrors current binding code); mac: Accelerate BNNS/AMX int8 path, intel: VNNI `_mm256_dpbusd_epi32`; keep FP32 reference path for the 1e-4 gate |
| 2 | **Batched prefill (GEMM path)** | 10–50× faster time-to-first-token on long/chat prompts | M | Add `matmul(C= A@B)` to ABI (default mixin = loop over matvec so old plugins load); add `forward_prefill()` processing the prompt in one matmul per weight; reuse for the classifier only when beneficial |
| 3 | **FP16 KV cache** | Halves cache bytes + bandwidth (7B/2k ctx: ~2.1 → ~1 GB); bigger contexts fit | M | Store cache as `__fp16`/bf16 with float compute; capability flag; measure quality delta on a perplexity probe |
| 4 | **Fused attention** (QKᵀ+softmax+AV in one pass, FlashAttention-style tiling) | Matters at 2k+ contexts where attention stops being negligible | M | Per-model-plugin kernel (lives in `models/<m>/plugins/` per `09`); Metal compute shader on Mac *only* for this — Accelerate already saturates GEMV |

Deliberately after the above: speculative decoding, MoE support, server/batching —
all need 1–4 first.

## P0 — Correctness infrastructure (do before any P0-perf item)

| # | What | Why |
|---|---|---|
| 5 | **Golden differential test in CI** | We verified bit-exactness by hand twice and caught a real layout bug that way. Commit it: `tests/golden_llama2.{cc,py}` asserting prompt IDs + post-prefill top-5 logits + 20 greedy tokens for stories15M (data-free: hardcode the ~30 expected numbers), tolerance 1e-4. Runs on every commit, all backends. |
| 6 | **Build matrix CI** (mac-arm64, linux-x86, windows) | Every plugin compiles everywhere by design — prove it on each PR. Catches `#ifdef` drift like the `sincosf` one we hit. |

## P1 — Performance details (small, safe)

7. **Persistent thread pool** in the intel plugin (and any future threaded
   backend) instead of spawning `std::thread`s per large matvec; pin threads
   on Windows. Spawning costs ~10–50 µs — invisible at 7B, visible at 15M.
8. **Runtime CPU dispatch** for x86: compile AVX2/FMA *and* scalar/AVX variants,
   pick via `cpuid` at `CreateBackend`. Today AVX2 is a compile-time flag, so
   one binary doesn't run everywhere.
9. **NEON-dotprod path** (`sdot`) for non-Apple ARM (Linux ARM servers): the
   intel-plugin equivalent for that platform; currently only scalar there.
10. **Sampler fast path**: skip softmax entirely for greedy (argmax directly on
    logits — already done for temp=0; extend: argmax-then-verify for top-k=1).
11. **LTO + `-mcpu=native` release preset** (`cmake -DOPTIMIZE_NATIVE=ON`) for
    benchmark builds; keep portable defaults for releases.

## P1 — Readability / maintainability

12. **Shared CLI arg-parsing helper in `common/`** — `run_llama.cc` will be
    copy-pasted per model otherwise (first duplication smell of the new layout).
13. **Tiny `Logger` in `common/`** replacing raw `fprintf` in engine/CLI
    (levels: info/warn; keeps library quiet when embedded).
14. **Move `tools/export_hf_to_bin.py` → `models/llama2/tools/`** and add a
    `tests/` round-trip: export a random-init HF config → load → forward →
    compare vs PyTorch logits (catches layout bugs like the grouped-by-tensor
    one at the source).
15. **Root `README.md`** (10 lines: what/where → pointers to `SPEC/DOC.md`).
    Newcomers currently land in a folder of equals.
16. **C API wrapper** (`inference_llama2.h` with opaque handles) when the first
    non-C++ consumer appears (Python demo, mobile app) — not before.

## P2 — Structural (when the 2nd–3rd model lands)

17. **Evaluate GGUF as the common checkpoint format.** Our `.bin` is ideal for
    learning (100-line loader) but every new model re-invents conversion.
    Keep `.bin` for llama2; adopt GGUF reader in `common/` if/when llama3 or
    Mistral arrives and the converter burden repeats.
18. **Op-registry re-evaluation.** `09` rejected a generic op-graph for v0.1
    (hides the math). Revisit at 3 models: if graphs are 90% shared rope/norm/
    matmul plumbing, a small registry removes more duplication than per-model
    engines cost. Decide with line counts, not taste.
19. **Plugin capability negotiation** (`query()` returning fp16/int8/matmul
    support) as part of rec #1 — design it then, not now.

## Explicit non-goals (keep saying no)
Training/gradients, multi-GPU tensor parallelism, HTTP server, on-device
fine-tuning, supporting every HF architecture — each would triple the scope
and none is needed for fast local single-stream inference, which is the
project's job.
