# 01 — Architecture: Multi-Model Inference Libraries

## Goal
Small, modular, dependency-free C++17 inference libraries — one per model
family — deployable to many hardwares via **plugins**. Shared, model-agnostic
code lives in `common/`; everything model-specific lives in `models/<name>/`.
Full rationale: `SPEC/09_MODEL_LAYOUT.md`.

```
  models/llama2/examples/run_llama (thin CLI)
        | uses
  libinference_llama2 (static)          libinference_common (static)
  - transformer graph (SwiGLU/GQA)  +   - IComputeBackend ABI + ReferenceBackend
  - weights loader (.bin layout)        - BPE tokenizer, sampler, plugin loader
  - LlamaEngine (KV cache, decode)      - ModelConfig
        | calls through ABI             ^
        +--------v----------+           | plugins subclass ReferenceBackend
        | IComputeBackend   | ----------+  (override only accelerated kernels)
        +----+----+----+----+
             |    |    |
  models/llama2/plugins/{cpu_baseline, mac_mseries, windows_intel}
  (.dylib/.so/.dll, exporting CreateBackend/DestroyBackend with C linkage)
```

Next model (`models/llama3/`, …) repeats the right column; the left column
is untouched.

## Modules

Shared (`common/`):

| Module | Headers | Responsibility |
|---|---|---|
| `core` | `config.h`, `backend_plugin.h`, `reference_backend.h`, `plugin_loader.h` | Backend ABI + scalar reference kernels, plugin loading, dense-decoder config. No model math. |
| `tokenizer` | `tokenizer.h` | BPE encode/decode (llama2/llama3-compatible). File-format only. |
| `sampling` | `sampler.h` | Greedy / temperature / top-p / top-k. Pure CPU, deterministic with seed. |

Per model (`models/<name>/`, e.g. `llama2` in namespace `inference::llama2`):

| Module | Headers | Responsibility |
|---|---|---|
| (model root) | `llama2/weights.h`, `llama2/transformer.h`, `llama2/engine.h` | Weight layout + loader, model graph + forward, orchestrator (`load()`/`generate()`). |
| `plugins/*` | `plugin_*.cc` | `ReferenceBackend` subclass per hardware; override only accelerated ops. |
| `weights/` | (checkpoints, gitignored) | `.bin` + `tokenizer.bin` live next to the code that reads them. |
| `examples/` | `run_llama.cc` | Thin CLI; library never touches argv. |

## Key invariants
1. **Common never includes model headers; models never include plugin headers.**
   The only cross-boundary types are `IComputeBackend*` and `ModelConfig`.
2. **Transformer never allocates per-token.** All activations + KV cache are
   pre-allocated in `Engine::load()`.
3. **Weights are read-only.** `mmap` (POSIX, +`MADV_WILLNEED`) /
   malloc+fread fallback (Windows), shared across threads. Plugins receive
   `const float*`.
4. **Precision:** FP32 activations, FP32 weights for v1. Quantization (INT8/4) is a
   future backend capability flag, not a graph change.
5. **Threading belongs to the backend.** Engines are single-threaded and not
   thread-safe (one instance per stream); backends may use GCD (Mac),
   thread-pool (Intel), etc.

## Data flow (one Llama-2 decode step)
```
token_id -> embedding lookup -> for each layer:
  rmsnorm -> qkv matvec (backend) -> RoPE (backend) -> attention (backend+cache)
  -> o_proj -> residual -> rmsnorm -> w1/w3 matvec -> SwiGLU -> w2 matvec -> residual
-> final rmsnorm -> logits matvec -> sampler -> next token_id
```

## File formats supported (v1)
- Checkpoint: karpathy-style `llama2.bin` (grouped-by-tensor FP32, see `06_MEMORY_WEIGHTS`).
- Tokenizer: `tokenizer.bin` (karpathy format) **or** HF `tokenizer.json` via export script.
- HF `.safetensors` / `.pth` are **not** read directly; convert with `tools/export_hf_to_bin.py`.
