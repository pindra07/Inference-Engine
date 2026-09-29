# 03 — Thinking / Design Decisions

## Why a plugin architecture?
We want one portable model graph + N hardware kernels. History shows matmul,
norm, and attention dominate Llama-2 runtime (>95%). Everything else (sampling,
tokenizer, KV bookkeeping) is negligible. So:

- Put **what to compute** in core (transformer graph, fixed math).
- Put **how to compute fast** in plugins (matvec, rmsnorm, softmax, RoPE, attention).

This lets us ship `mac_mseries` today, `windows_intel` today, and `cuda/rocm/npu`
tomorrow without touching the model code.

## Why not llama.cpp / ONNX directly?
- llama.cpp is excellent but monolithic for our learning goal; we want a minimal
  readable core (~2k LOC) with an explicit ABI boundary.
- ONNX Runtime is heavy and hides the transformer math we want to own.
- Our `.bin` format (Karpathy-compatible) keeps the loader to ~100 lines and lets
  us `mmap` weights with zero parse cost.

## Why FP32 v1?
- Correctness first. FP32 matches Meta reference logits and makes Mac/Intel
  plugins comparable bit-for-bit (tolerance 1e-4).
- Quantization (Q8/Q4) will be a backend `Capability` flag + new weight loader;
  the transformer graph does not change.

## Why mmap weights?
- Llama-2-7B is ~26 GB in FP32 (13 GB in FP16 export). `mmap` avoids a copy,
  lets the OS page in lazily, and shares pages across processes.
- On Apple Silicon unified memory, `mmap` + Accelerate gives near-zero-copy
  matvec. On Windows, `CreateFileMapping` gives the same benefit.

## Why GQA-ready?
- Llama-2-7B uses MHA (32 heads, 32 kv heads); 70B uses GQA. We implement KV heads
  with `n_kv_heads` + head-repeat from day one so 70B works without rewrite.

## Why RoPE in backend?
- RoPE is a paired rotation applied to q/k per position. It is memory-bound and
  benefits from NEON/AVX vectorization, so it lives in the backend, not core.

## Risks / non-goals (v1)
- No speculative decoding, no batching, no server. Single-stream decode only.
- No training, no gradients.
- Windows plugin tested for correctness on any x86_64; AVX2 path is best-effort
  and falls back to scalar when `__AVX2__` is unavailable.
