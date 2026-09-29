# 09 — Multi-Model Folder Design

## Problem
Today there is exactly one model (Llama-2) and the tree mixes generic code
(tokenizer, sampler, plugin ABI) with Llama-2 specifics (transformer graph,
`.bin` weight layout, engine) and hardware plugins in one flat `plugins/`.
Adding Llama-3 / Mistral / Qwen would mean either copy-paste (divergence) or
`if (arch == ...)` branches inside hot code (breaks the modularity we just
built).

## Decision: `common/` + `models/<name>/`

```
inference/
├── SPEC/
├── common/                        # model-AGNOSTIC, stable, rarely changes
│   ├── include/inference/
│   │   ├── core/                  # backend_plugin.h (ABI), plugin_loader.h,
│   │   │                          # config.h, reference_backend.h
│   │   ├── tokenizer/             # shared BPE (llama2+llama3 compatible)
│   │   └── sampling/              # shared sampler
│   └── src/core|tokenizer|sampling/
│
├── models/<name>/                 # everything model-SPECIFIC lives here
│   └── llama2/
│       ├── include/inference/llama2/  # weights.h, transformer.h, engine.h
│       ├── src/                       # weights.cc, transformer.cc, engine.cc
│       ├── plugins/                   # hardware backends FOR THIS MODEL
│       │   ├── cpu_baseline/
│       │   ├── mac_mseries/
│       │   └── windows_intel/
│       ├── weights/                   # checkpoints go here (gitignored)
│       └── examples/run_llama.cc      # thin CLI for this model
│
└── tools/export_hf_to_bin.py      # per-model converters live with the model
    └── (moved: models/llama2/tools/ in future; single model → keep top-level)
```

## What is shared vs model-specific (the rule)

| Shared (`common/`) | Model-specific (`models/<name>/`) |
|---|---|
| `IComputeBackend` ABI (matvec, rmsnorm, softmax, rope, silu, elem_mul, attention) | Transformer graph (SwiGLU vs GELU, GQA vs MHA vs sliding-window, norm placement) |
| `ReferenceBackend` scalar kernels — plugins subclass it | Weight file layout + loader (llama2 `.bin` grouped-by-tensor) |
| `ModelConfig` (dim/hidden/layers/heads/…, rope_theta) — a dense decoder-only config; models extend only if needed | Engine orchestration (KV shapes, prefill/decode, chat template) |
| BPE tokenizer, sampler, plugin loader | Plugins (can grow model-specific *fused* kernels later), CLI, checkpoints |

Rationale per row:
- **ABI in common, plugins per model.** Primitive ops are identical math
  everywhere, so one ABI. But plugin *binaries* live with the model so a
  future fused kernel (e.g. `llama2_fused_attn`) can't break `llama3`.
  Today each model plugin is thin: subclass `ReferenceBackend`, override
  only accelerated ops.
- **Engine per model, not `if (arch)`.** Graph differences (Mistral sliding
  window, Qwen QKV-bias, Llama-3 128k vocab + 8k context) touch allocation,
  forward order, and sampling context. A per-model engine keeps each graph
  readable; shared helpers stay in `common`.
- **`weights/` inside the model folder.** Checkpoints are large, licensed,
  and model-specific. `models/llama2/weights/` + `.gitignore` keeps them
  next to the code that reads them; `DOC.md` download paths point there.

## Alternatives considered (and rejected)

1. **One engine + `enum Arch` switches.** Rejected: branches in
   allocation/forward for every future arch; every model pays for every
   other model's complexity. Exactly the monolith we avoided with plugins.
2. **Fully generic op-graph (registry of named ops).** Attractive long-term
   (GGUF-style), but v0.1 needs readable, debuggable code — a registry
   hides the math. Revisit when the 3rd model lands (see recommendation.md).
3. **Flat tree + copy `plugins/` per model later.** Rejected: the weight
   layout bug (SPEC/08 §"FOUND IT") proved format knowledge must live next
   to its model, not in shared code.

## ABI stability note
`IComputeBackend` is frozen at v1 after this pass (only change: the
`elem_mul_add`→`elem_mul` rename, done *before* any external plugin
exists). Future kernel additions (e.g. `matmul` for batched prefill, quant
matvec) go through `Capability` query + default no-op/mixin
implementations so old plugins keep loading.

## Adding a new model (checklist for `models/llama3/`)
1. Copy `models/llama2/{include,src}` → same shape, new namespace
   `inference::llama3`; adjust graph + loader.
2. Copy `models/llama2/plugins/` → keep generic overrides; add fused
   kernels only if measured worthwhile.
3. Add `weights/` + README with download/convert commands.
4. Add `examples/` CLI (or share arg-parsing via a tiny `common` helper).
5. One line in root `CMakeLists.txt`: `add_subdirectory(models/llama3)`.
6. Extend `SPEC/` with `10_LLAMA3.md` (deltas only) + DOC section.

## Migration map (old → new, this pass)
`include/inference/{core/tokenizer/sampling}` → `common/include/...` (same
`#include "inference/..."` paths, zero code churn) · `transformer/`,
`engine/`, `weights.*` → `models/llama2/` under `inference::llama2` ·
`plugins/*` → `models/llama2/plugins/*` · `examples/run_llama.cc` →
`models/llama2/examples/` · checkpoints → `models/llama2/weights/`.
