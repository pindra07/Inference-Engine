# 02 — Coding Style & Conventions

## Language
- C++17, `-Wall -Wextra -Werror` clean on Clang (Mac), GCC, MSVC.
- No external dependencies for `libinference_core`. Plugins may use OS frameworks
  (Accelerate on Mac, intrinsics on x86) but must compile to a no-op fallback elsewhere.

## Naming
- Files: `snake_case.h / .cc`. Headers live under `include/inference/<module>/`.
- Types: `PascalCase` (`LlamaEngine`, `ModelConfig`). Functions/methods: `snake_case`.
- Namespaces: `inference::core`, `inference::transformer`, `inference::tokenizer`,
  `inference::sampling`, `inference::engine`. Plugins: `inference::plugin::<name>`.
- Constants for model dims: `dim`, `hidden_dim`, `n_layers`, `n_heads`, `n_kv_heads`,
  `vocab_size`, `seq_len` — same names as Meta's `params.json`.

## Headers
- `#pragma once`. No `using namespace` in headers. Forward-declare where possible.
- Every public header documents ownership: who allocates, who frees, thread-safety.

## Memory / errors
- Hot path: no `new`/`malloc`, no exceptions, no `std::vector` resize. Pre-allocate.
- Cold path (load/init): exceptions allowed (`std::runtime_error` with file + line).
- Raw weight pointers are `const float*` + `size_t num_floats`, owned by `Weights` RAII class.
- Tensors are **views** (`TensorView { const float* data; int rows, cols; }`), never owners.

## Backend kernels
- Signature style: `void rmsnorm(const float* x, const float* w, float* out, int dim)`.
- All kernel args validated with `assert` in debug, early-return in release.
- Each plugin file has `// HW-NOTE:` comment explaining the intrinsic/framework used.
- New plugins **subclass `core::ReferenceBackend`** and override only
  accelerated kernels (see `SPEC/04`). Shared scalar code is never duplicated.
- Platform fast-paths (e.g. `__sincosf`, AVX2) use a named macro with a
  portable fallback in the same function — no separate code paths to drift.

## Tests / examples
- `examples/run_llama.cc` is the only binary that parses argv. Library never touches CLI.
- Keep functions < 60 lines. If longer, split into `detail::` helper.

## Formatting
- 2-space indent, 100-col limit, braces on same line (LLVM-ish, relaxed).
- CMake: `cmake_minimum_required(3.16)`, `target_*` scoped commands only.
