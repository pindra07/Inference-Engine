# Inference — modular C++ inference library with hardware plugins

A small, dependency-free (C++17, no third-party libraries) inference engine
for modern decoder-only LLMs, designed so **each model family gets an engine
folder** and **each hardware target gets a plugin**. Today the repo ships two
models (`llama2`, `gemma2`) with three hardware plugins each
(`mac_mseries`, `windows_intel`, `cpu_baseline`), plus an interactive
type-and-respond chat mode. Adding a new model family or a new chip means
adding a folder — never rewriting the engine.

Design notes, optimization decisions, and the multi-model roadmap live in
[`SPEC/`](SPEC/) (start with `SPEC/DOC.md` for the short run guide and
`SPEC/recommendation.md` for what to build next).

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Prerequisites](#2-prerequisites)
3. [Build](#3-build)
4. [Install a model (download)](#4-install-a-model-download)
5. [Run locally (CLI)](#5-run-locally-cli)
6. [Run on specific hardware](#6-run-on-specific-hardware)
7. [Interactive chat mode](#7-interactive-chat-mode)
8. [Gemma2](#8-gemma2)
9. [Use this repo as a library in your own project](#9-use-this-repo-as-a-library-in-your-own-project)
10. [Write your own hardware plugin](#10-write-your-own-hardware-plugin)
11. [Troubleshooting](#11-troubleshooting)
12. [Benchmarking](#12-benchmarking)

---

## 1. Repository layout

```
inference/
├── README.md                    # this file
├── CMakeLists.txt               # top-level: builds common/ + models/*/
├── SPEC/                        # all design thinking (01–10), DOC, recommendation.md
├── common/                      # model-agnostic, shared by every model
│   ├── include/inference/core/      # plugin ABI, reference kernels, loader, config
│   ├── include/inference/tokenizer/ # shared BPE tokenizer
│   ├── include/inference/sampling/  # shared sampler (greedy/temp/top-k/top-p)
│   ├── include/inference/app/       # shared interactive REPL helper
│   └── src/                         # implementations of the above
├── models/llama2/               # everything Llama-2-specific
│   ├── include/inference/llama2/    # weights.h, transformer.h, engine.h
│   ├── src/                         # weights.cc, transformer.cc, engine.cc
│   ├── plugins/                     # one folder per hardware target
│   │   ├── cpu_baseline/            # portable reference (validates the others)
│   │   ├── mac_mseries/             # Apple Silicon via Accelerate + GCD
│   │   └── windows_intel/           # x86-64 via AVX2+FMA + thread pool
│   ├── weights/                     # checkpoints go HERE (gitignored) + README
│   └── examples/run_llama.cc        # thin CLI demo (one-shot + --interactive)
├── models/gemma2/               # everything Gemma2-specific (same shape)
│   ├── include/inference/gemma2/    # config.h, weights.h, transformer.h, engine.h
│   ├── src/ · plugins/{cpu_baseline,mac_mseries,windows_intel}/
│   ├── weights/ · tools/            # checkpoints + HF→GGM2 converters
│   └── examples/run_gemma.cc
└── tools/export_hf_to_bin.py    # converts Hugging Face Llama checkpoints to .bin
```

Build output goes to `build/`:

```
build/
├── libinference_common.a
├── models/llama2/
│   ├── libinference_llama2.a
│   ├── examples/run_llama                        # the CLI binary
│   └── plugins/
│       ├── libplugin_llama2_cpu_baseline.dylib   # (.so on Linux, .dll on Windows)
│       ├── libplugin_llama2_mac_mseries.dylib
│       └── libplugin_llama2_windows_intel.dylib
└── models/gemma2/                 # same shape: libinference_gemma2.a,
    ...                           # run_gemma, libplugin_gemma2_*.dylib
```

---

## 2. Prerequisites

You need a C++17 compiler, CMake ≥ 3.16, Python ≥ 3.9 (only for the
Hugging Face converter script), and ~200 MB free for the test model
(~30 GB free if you later run Llama-2-7B in FP32).

**macOS (Apple Silicon, M1–M4)**
```bash
xcode-select --install          # compilers + headers (run once)
cmake --version || pip install cmake
python3 --version
```

**Linux (x86-64 / ARM)**
```bash
sudo apt install build-essential cmake python3 python3-pip   # Debian/Ubuntu
```
No extra libraries needed — the Intel plugin uses only compiler intrinsics
and `std::thread`.

**Windows (Intel/AMD 64-bit)**
- Install Visual Studio (Community is fine) with the “Desktop development
  with C++” workload, plus CMake (or `pip install cmake`) and Python 3.9+.
- Build from a “Developer Command Prompt” so the compiler is on `PATH`.
- Plugin files will be `.dll` instead of `.dylib`; everything else below is
  identical (use the `windows_intel` plugin, section 6).

---

## 3. Build

```bash
git clone <this-repo> inference && cd inference
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

What this produces:

| Artifact | Path | What it is |
|---|---|---|
| `libinference_common.a` | `build/` | shared kernels, tokenizer, sampler, plugin loader, REPL |
| `libinference_llama2.a` | `build/models/llama2/` | Llama-2 graph + engine (link this + common in your app) |
| `libinference_gemma2.a` | `build/models/gemma2/` | Gemma2 graph + engine |
| `run_llama` / `run_gemma` | `build/models/<m>/examples/` | demo CLIs (one-shot + `--interactive`) |
| `libplugin_<m>_*.dylib` | `build/models/<m>/plugins/` | the three hardware plugins per model |

Verify the build worked:
```bash
ls build/models/llama2/examples/run_llama build/models/gemma2/examples/run_gemma
ls build/models/llama2/plugins/ build/models/gemma2/plugins/
./build/models/llama2/examples/run_llama --help
```

---

## 4. Install a model (download)

Model files live in `models/<model>/weights/` (gitignored — checkpoints are
large and license-gated, so they are never committed). Each model needs a
**checkpoint** and a **tokenizer**. This section covers Llama-2; Gemma2
downloads are in [section 8](#8-gemma2).

### Option A — TinyStories 15M (recommended first step, no approval needed)

A real Llama-2-architecture checkpoint (~60 MB) trained on TinyStories.
Runs in seconds on any machine and is what the correctness gate uses.

```bash
mkdir -p models/llama2/weights
curl -L -o models/llama2/weights/stories15M.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o models/llama2/weights/tokenizer.bin \
  https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
ls -lh models/llama2/weights/
# stories15M.bin  (~58 MB)   tokenizer.bin (~424 KB)
```

> Important: `stories15M` uses the standard 32k Llama `tokenizer.bin`.
> Do **not** use `tok512.bin` — that vocabulary belongs only to the
> `stories260K` model and will produce garbage with any other checkpoint.

### Option B — Real Llama-2-7B from Hugging Face (gated)

Llama-2 weights require accepting Meta's license. Steps in detail:

**Step 1 — Get access.**
1. Create an account at https://huggingface.co/join (if you don't have one).
2. Open https://huggingface.co/meta-llama/Llama-2-7b-hf and fill in the
   access-request form (it references the Llama 2 Community License).
   Approval is usually automatic within minutes-to-hours.
3. Create a read-only access token at
   https://huggingface.co/settings/tokens → “Create new token”, type `Read`.

**Step 2 — Install the download/convert tools.**
```bash
pip install huggingface_hub transformers safetensors torch numpy
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx   # paste your token from step 1.3
```

**Step 3 — Download the checkpoint (~13 GB of safetensors).**
```bash
huggingface-cli download meta-llama/Llama-2-7b-hf \
  --local-dir /tmp/Llama-2-7b-hf
ls /tmp/Llama-2-7b-hf
# config.json  tokenizer.json  model-00001-of-00002.safetensors  ...
```

**Step 4 — Convert to this library's `.bin` format.**
This engine does not parse `safetensors` directly (keeps the loader to
~100 lines and lets weights be `mmap`'d with zero parsing). Convert once:
```bash
python tools/export_hf_to_bin.py \
  --hf-dir /tmp/Llama-2-7b-hf \
  --out-bin models/llama2/weights/llama2-7b.bin \
  --out-tok models/llama2/weights/tokenizer.bin
ls -lh models/llama2/weights/
# llama2-7b.bin (~26 GB, FP32)   tokenizer.bin
```

**Step 5 — Sanity-check the files.**
- `llama2-7b.bin` should be ~26 GB. If it is ~13 GB you exported in FP16 by
  accident — re-run the script (it writes FP32).
- Keep `tokenizer.bin` from this step (it matches the 7B vocabulary).

> Hardware note: FP32 7B needs a machine with plenty of RAM (16 GB minimum,
> 32 GB comfortable). On an 8 GB Mac, stick to TinyStories/TinyLlama, or wait
> for INT8 quantization (`SPEC/recommendation.md`, item 1).

---

## 5. Run locally (CLI)

General form:
```bash
./build/models/llama2/examples/run_llama \
  --checkpoint <MODEL.bin> --tokenizer <TOKENIZER.bin> [options]
```

All flags (`--help` prints the same list):

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint PATH` | (required) | model `.bin` from section 4 |
| `--tokenizer PATH` | (required) | matching `tokenizer.bin` |
| `--backend PATH` | built-in reference | hardware plugin `.dylib`/`.so`/`.dll` (or `INFERENCE_BACKEND` env var) |
| `--prompt STR` | `"Hello"` | input text (ignored with `--interactive`) |
| `--chat` | off | wrap prompt in the model's chat template (`[INST]` / `<start_of_turn>`) |
| `--interactive`, `-i` | off | chat loop per line (see section 7) |
| `--steps N` | 128 | max **new** tokens to generate |
| `--temperature F` | 0.7 | `0` = greedy/deterministic; higher = more random |
| `--top_p F` | 0.9 | nucleus sampling cutoff (`1.0` = off) |
| `--top_k N` | 0 = off | keep only top-K candidates per step |
| `--seed N` | 42 | RNG seed (same seed + same flags = same output) |
| `--seq_len N` | model's max | lower it to shrink the KV cache on small machines |

**Smoke test (deterministic, should print a coherent story):**
```bash
W=models/llama2/weights; B=build/models/llama2
./$B/examples/run_llama \
  --checkpoint $W/stories15M.bin --tokenizer $W/tokenizer.bin \
  --backend $B/plugins/libplugin_llama2_mac_mseries.dylib \
  --prompt "Once upon a time" --steps 200 --temperature 0.0
```

**Creative sampling + chat template (7B example):**
```bash
W=models/llama2/weights; B=build/models/llama2
./$B/examples/run_llama \
  --checkpoint $W/llama2-7b.bin --tokenizer $W/tokenizer.bin \
  --backend $B/plugins/libplugin_llama2_mac_mseries.dylib \
  --chat --steps 256 --temperature 0.7 --top_p 0.9 --seed 7 \
  --prompt "Explain recursion in one sentence."
```

The first stderr line always tells you which backend is actually running,
e.g. `backend=llama2/mac_mseries` — if you asked for a plugin and see
`backend=reference`, your `--backend` path was ignored (see troubleshooting).

---

## 6. Run on specific hardware

The engine selects kernels through `--backend <plugin file>` (or the
`INFERENCE_BACKEND` environment variable with the same path). Omit both and
you get the built-in portable reference backend — always correct, never the
fastest. Commands below use Llama-2 paths; the Gemma2 equivalents (same
flags, `run_gemma` + `libplugin_gemma2_*`) are in §8.1–8.4.

### 6.1 Apple Silicon (M1/M2/M3/M4) — `mac_mseries` plugin ⭐

This is the recommended backend on Mac. It routes matrix math through
Apple's Accelerate framework (`cblas_sgemv`, `vDSP`) and fans attention heads
across performance cores with GCD — tuned for unified memory, where the CPU
and GPU share one pool and weight streaming is the bottleneck.

```bash
W=models/llama2/weights; B=build/models/llama2
./$B/examples/run_llama \
  --checkpoint $W/stories15M.bin --tokenizer $W/tokenizer.bin \
  --backend $B/plugins/libplugin_llama2_mac_mseries.dylib \
  --prompt "Once upon a time" --steps 200 --temperature 0.0
# expect: backend=llama2/mac_mseries, ~850 tok/s (stories15M, M4)
```

Expectations: stories15M in the high hundreds of tok/s; 7B-FP32 only ~2–4
tok/s — that is the memory-bandwidth wall (26 GB of weights re-read per
token), not a kernel problem. It is fixed by quantization, not by faster
math (`SPEC/08_OPTIMIZATION.md` explains the arithmetic).

### 6.2 Windows / Linux on Intel/AMD x86-64 — `windows_intel` plugin

Uses AVX2 + FMA intrinsics (`_mm256_*`, 8 floats per instruction) with a
thread-pool split over matrix rows and attention heads. Enabled automatically
at compile time on `x86_64`/`AMD64` (`-mavx2 -mfma`, or `/arch:AVX2` on MSVC).

```bash
# Linux example (.so extension); on Windows the file is .dll
./build/models/llama2/examples/run_llama \
  --checkpoint models/llama2/weights/stories15M.bin \
  --tokenizer models/llama2/weights/tokenizer.bin \
  --backend build/models/llama2/plugins/libplugin_llama2_windows_intel.so \
  --prompt "Once upon a time" --steps 200 --temperature 0.0
# expect: backend=llama2/windows_intel
```

Notes:
- CPUs without AVX2 still work: the plugin was validated on machines where
  only the scalar fallback compiles (same numerics, less speed).
- Threading kicks in only for large matrices (≥ 2048 rows) and long contexts
  (> 512 tokens) — below that, thread-spawn overhead would cost more than it
  saves.

### 6.3 Any CPU (fallback) — built-in reference or `cpu_baseline` plugin

Two identical options, useful for correctness checks and platforms without a
tuned plugin yet:

```bash
# Option 1: omit --backend entirely (built-in reference backend)
./build/models/llama2/examples/run_llama \
  --checkpoint models/llama2/weights/stories15M.bin \
  --tokenizer models/llama2/weights/tokenizer.bin \
  --prompt "Once upon a time" --steps 60 --temperature 0.0
# expect: backend=reference, ~195 tok/s (stories15M, M4)

# Option 2: the same code as an explicit plugin (validates plugin loading)
./build/models/llama2/examples/run_llama \
  --checkpoint models/llama2/weights/stories15M.bin \
  --tokenizer models/llama2/weights/tokenizer.bin \
  --backend build/models/llama2/plugins/libplugin_llama2_cpu_baseline.dylib \
  --prompt "Once upon a time" --steps 60 --temperature 0.0
# expect: backend=llama2/cpu_baseline
```

Correctness contract: greedy (`--temperature 0.0`) output is byte-identical
to `llama2.c`'s reference runner on every backend. If you ever suspect a new
plugin, run the same prompt with `--temperature 0.0` on `cpu_baseline` and
diff the text.

### 6.4 How backend selection works (so you can reason about it)

1. `--backend PATH` wins if given.
2. Else `INFERENCE_BACKEND` env var, if set — handy for scripts:
   `export INFERENCE_BACKEND=$PWD/build/models/llama2/plugins/libplugin_llama2_mac_mseries.dylib`
3. Else the built-in `reference` backend.
4. The engine prints `backend=<name>` on stderr at startup — trust that line,
   not your flags.

---

## 7. Interactive chat mode

Both CLIs (`run_llama`, `run_gemma`) support `--interactive` (or `-i`): instead
of one `--prompt`, you get a `>` prompt — type a line, read the model's
response, type the next line. End with `quit` (or `/quit`/`exit`/`:q`),
Ctrl+D, or an empty line being skipped (blank lines are never sent).

```bash
W=models/llama2/weights; B=build/models/llama2
./$B/examples/run_llama \
  --checkpoint $W/stories15M.bin --tokenizer $W/tokenizer.bin \
  --backend $B/plugins/libplugin_llama2_mac_mseries.dylib \
  --steps 100 --temperature 0.7 --seed 7 --interactive
# backend=llama2/mac_mseries
# [interactive: type text, Enter for response]
# > Once upon a time
# , there was a little girl named Lily. ...
# > quit
```

Combine with `--chat` to apply the model's chat template to *every* line you
type (Llama `[INST]` / Gemma `<start_of_turn>` — see each CLI's `--help`).
Notes:

- Each line is an **independent prompt**: the KV cache restarts every turn,
  so the model has no memory of previous lines (multi-turn state is a
  planned engine feature, `SPEC/recommendation.md`).
- `--prompt` is ignored in interactive mode; `--steps`, `--temperature`,
  `--top_p`, `--top_k`, `--seed`, `--backend` all still apply per response.
- The loop lives in `common/` (`inference::app::run_repl`), so every model's
  CLI behaves identically; engines stay non-interactive and testable.

---

## 8. Gemma2

The second model family (`models/gemma2/`, namespace `inference::gemma2`),
same plugin doctrine: `cpu_baseline`, `mac_mseries`, `windows_intel`
under `models/gemma2/plugins/`. Architecture deltas vs Llama-2 (GeGLU,
sliding-window attention, logit soft-capping, folded `(1+w)` norms, explicit
256-dim heads) are documented in `SPEC/10_GEMMA2.md`; the ABI additions they
required (`attention_softcap`, `tanh_softcap`, `gelu_tanh`) in `SPEC/04`.

**Install a test model (ungated, ~12 MB total).**
```bash
pip install torch safetensors sentencepiece huggingface_hub
python3 -c "from huggingface_hub import snapshot_download;
snapshot_download('yujiepan/gemma-2-tiny-random', local_dir='/tmp/gemma-2-tiny-random')"
python models/gemma2/tools/export_hf_to_gemma2.py \
  --hf-dir /tmp/gemma-2-tiny-random \
  --out-bin models/gemma2/weights/tiny-gemma2.bin
python models/gemma2/tools/export_gemma_tokenizer.py \
  --model /tmp/gemma-2-tiny-random/tokenizer.model \
  --json /tmp/gemma-2-tiny-random/tokenizer.json \
  --out models/gemma2/weights/tokenizer.bin
ls -lh models/gemma2/weights/
# tiny-gemma2.bin (~8 MB)   tokenizer.bin (~3.6 MB, 256k pieces)
```

**Run it (one-shot + interactive).**
```bash
W=models/gemma2/weights; B=build/models/gemma2
./$B/examples/run_gemma \
  --checkpoint $W/tiny-gemma2.bin --tokenizer $W/tokenizer.bin \
  --backend $B/plugins/libplugin_gemma2_mac_mseries.dylib \
  --prompt "Hello" --steps 20 --temperature 0.0
./$B/examples/run_gemma \
  --checkpoint $W/tiny-gemma2.bin --tokenizer $W/tokenizer.bin \
  --steps 60 --temperature 0.7 --interactive
```
(The tiny model has random weights, so output is gibberish — it exists to
prove the math: greedy output is bit-exact vs a pure-torch reference,
see `SPEC/10`.)

### 8.1 Gemma2 on Apple Silicon — `mac_mseries` plugin ⭐

Same Accelerate doctrine as Llama-2, plus Gemma kernels: `attention_softcap`
(vDSP dot/axpy + GCD fan-out, softcap + sliding-window mask folded into the
score loop) and `tanh_softcap` (vForce `vvtanhf`); `gelu_tanh`/`rope` inherit
the shared scalar implementation.

```bash
W=models/gemma2/weights; B=build/models/gemma2
./$B/examples/run_gemma \
  --checkpoint $W/tiny-gemma2.bin --tokenizer $W/tokenizer.bin \
  --backend $B/plugins/libplugin_gemma2_mac_mseries.dylib \
  --prompt "Hello" --steps 20 --temperature 0.0
# expect: backend=gemma2/mac_mseries
```

### 8.2 Gemma2 on Windows / Linux x86-64 — `windows_intel` plugin

AVX2+FMA `matvec`/`rmsnorm`/`elem_mul` (same kernels as the Llama-2 Intel
plugin) with the scalar `attention_softcap` inherited — matvec is ~95% of
decode time, so attention stays scalar in v1 by design (`SPEC/10`).
Use `.so` on Linux, `.dll` on Windows:

```bash
./build/models/gemma2/examples/run_gemma \
  --checkpoint models/gemma2/weights/tiny-gemma2.bin \
  --tokenizer models/gemma2/weights/tokenizer.bin \
  --backend build/models/gemma2/plugins/libplugin_gemma2_windows_intel.so \
  --prompt "Hello" --steps 20 --temperature 0.0
# expect: backend=gemma2/windows_intel
```

### 8.3 Gemma2 on any CPU (fallback) — built-in reference or `cpu_baseline`

```bash
# Option 1: omit --backend entirely (built-in reference backend)
./build/models/gemma2/examples/run_gemma \
  --checkpoint models/gemma2/weights/tiny-gemma2.bin \
  --tokenizer models/gemma2/weights/tokenizer.bin \
  --prompt "Hello" --steps 20 --temperature 0.0
# expect: backend=reference

# Option 2: the same code as an explicit plugin (validates plugin loading)
./build/models/gemma2/examples/run_gemma \
  --checkpoint models/gemma2/weights/tiny-gemma2.bin \
  --tokenizer models/gemma2/weights/tokenizer.bin \
  --backend build/models/gemma2/plugins/libplugin_gemma2_cpu_baseline.dylib \
  --prompt "Hello" --steps 20 --temperature 0.0
# expect: backend=gemma2/cpu_baseline
```

### 8.4 Gemma2 interactive chat

Same shared REPL as Llama-2 (§7); add `--chat` for the
`<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n` template
(control tokens ship in the converted tokenizer, registered automatically —
a warning prints if your tokenizer lacks them):

```bash
W=models/gemma2/weights; B=build/models/gemma2
./$B/examples/run_gemma \
  --checkpoint $W/tiny-gemma2.bin --tokenizer $W/tokenizer.bin \
  --backend $B/plugins/libplugin_gemma2_mac_mseries.dylib \
  --chat --steps 60 --temperature 0.7 --interactive
# > Hello there
# ...
# > quit
```

**Real Gemma-2-2B/9B (gated, free license).** Accept the terms at
https://huggingface.co/google/gemma-2-2b, download, and convert with the same
two scripts (exact commands in `models/gemma2/weights/README.md`). Then:
```bash
./build/models/gemma2/examples/run_gemma \
  --checkpoint models/gemma2/weights/gemma2-2b.bin \
  --tokenizer models/gemma2/weights/tokenizer.bin \
  --backend build/models/gemma2/plugins/libplugin_gemma2_mac_mseries.dylib \
  --chat --interactive --temperature 0.7
```

Correctness contract (same as Llama-2): greedy output identical on all four
backends; Gemma2 additionally verified bit-exact against HF `transformers`
math on the tiny model.

---

## 9. Use this repo as a library in your own project

You don't need the CLI — link the two static libraries and drive the engine
directly.

**Step 1 — Add the dependency (CMake).**
```cmake
add_subdirectory(inference)   # or FetchContent / git submodule
target_link_libraries(my_app PRIVATE inference_llama2 inference_common)
# Gemma2 instead (or as well): inference_gemma2 + inference_common,
# headers under inference/gemma2/, engine class inference::gemma2::GemmaEngine
```
`inference_llama2` publicly pulls in `inference_common`, so include paths for
`inference/...` headers come along automatically. (Consumers never need to
build the plugins unless they want runtime backend switching — the built-in
reference backend has zero extra dependencies.)

**Step 2 — Minimal example.**
```cpp
#include <cstdio>
#include "inference/llama2/engine.h"

int main() {
  inference::llama2::EngineConfig cfg;
  cfg.checkpoint = "models/llama2/weights/stories15M.bin";
  cfg.tokenizer  = "models/llama2/weights/tokenizer.bin";
  // Optional: use a hardware plugin instead of the built-in reference:
  // cfg.backend_path = "build/models/llama2/plugins/libplugin_llama2_mac_mseries.dylib";
  cfg.sample.temperature = 0.0f;  // greedy = deterministic

  inference::llama2::LlamaEngine engine;
  engine.load(cfg);  // throws std::runtime_error on bad paths/formats
  std::printf("backend=%s\n", engine.backend_name().c_str());

  engine.generate("Once upon a time", /*max_steps=*/100,
                  [](int id, const std::string& piece) {
                    std::fwrite(piece.data(), 1, piece.size(), stdout);
                    return true;  // return false to stop early
                  });
  std::printf("\n");
  return 0;
}
```

**API notes worth knowing:**
- `LlamaEngine::load()` does everything expensive once: `mmap`s weights,
  loads the tokenizer, resolves the backend, pre-allocates activations and
  the KV cache (64-byte aligned). Afterwards `generate()` allocates nothing
  per token.
- `generate()` first prefills the prompt, then decodes one token per step
  until `max_steps` or EOS. The callback receives each decoded piece for
  streaming; returning `false` stops generation.
- One `LlamaEngine` instance = one independent stream (not thread-safe by
  design; use one instance per thread — weights are read-only and shared).
- Sampling is configured via `cfg.sample` (`temperature`, `top_p`, `top_k`,
  `seed`); temperature `0` means greedy argmax.

**Loading a plugin programmatically** (what `EngineConfig::backend_path`
does under the hood — useful if you manage backends yourself):
```cpp
#include "inference/core/plugin_loader.h"
inference::core::PluginLoader loader;
inference::core::BackendPtr backend =
    loader.load("build/models/llama2/plugins/libplugin_llama2_mac_mseries.dylib");
// backend->matvec(...), backend->rmsnorm(...), ...  (see SPEC/04)
```
`BackendPtr` calls the plugin's `DestroyBackend` on destruction; the OS
library handle intentionally stays open for process lifetime.

---

## 10. Write your own hardware plugin

The normal case is ~30 lines: subclass the shared reference backend and
override only the kernels your hardware accelerates. Full contract in
`SPEC/04_PLUGIN_API.md`; checklist (shown for llama2, same for gemma2 with
`gemma2_plugin(...)` — or add a fused model-specific kernel there):

1. Copy `models/llama2/plugins/mac_mseries/` → `models/llama2/plugins/my_hw/`.
2. Subclass `core::ReferenceBackend`; override e.g. `matvec`/`rmsnorm` and
   keep `name()` as `"llama2/my_hw"`. Inherit the rest (rope/silu/…).
3. Keep the `CreateBackend`/`DestroyBackend` exports at the file's end.
4. Register one line in `models/llama2/CMakeLists.txt`:
   `llama2_plugin(plugin_llama2_my_hw plugins/my_hw/plugin_my_hw.cc)`
   plus any `target_link_libraries` for your SDK/framework.
5. Rebuild, then validate: run the same greedy prompt against
   `cpu_baseline` and `diff` — outputs must be identical (tolerance 1e-4
   on logits; greedy text must match exactly).

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `dlopen … image not found` | Bad `--backend` path — use an absolute path and check `file *.dylib` reports `arm64` (or `x86-64` on Intel). |
| `checkpoint truncated` / `unexpected trailing size` | You pointed `--checkpoint` at a Hugging Face file (`.safetensors`/`.pt`) — convert first (llama: `tools/export_hf_to_bin.py`, §4 option B step 4; gemma: `models/gemma2/tools/export_hf_to_gemma2.py`, §8). |
| `bad magic (not a GGM2 file)` | A llama `.bin` passed to `run_gemma` (or vice versa) — each engine reads only its own format (`SPEC/06` vs `SPEC/10`). |
| Garbled text (wrong words, not just sampling noise) | Wrong tokenizer for the checkpoint (e.g. `tok512.bin` with `stories15M`). Each checkpoint needs its matching `tokenizer.bin` (§4). |
| Process killed / OOM on 7B | Not enough RAM for FP32 7B — use stories15M, a smaller model, or `--seq_len 512` to shrink the KV cache. |
| Slower than expected on Mac | Check the startup line: if it says `backend=reference`, your `--backend` flag didn't take effect. |
| `cmake: command not found` | `pip install cmake`, then re-run section 3. |
| Windows: AVX2 code won't compile on old MSVC | Update to VS 2019+; the plugin falls back to scalar automatically where `__AVX2__` is unavailable, so it still builds and runs. |

Still stuck? The design docs usually answer “why does it work this way”:
`SPEC/05` (transformer math), `SPEC/06` (llama weight layout),
`SPEC/07` (tokenizer/sampler), `SPEC/08` (what was optimized, with
measurements), `SPEC/09` (folder design + how to add a model family),
`SPEC/10` (Gemma2 deltas + verification), `SPEC/recommendation.md` (what to
build next — quantization is #1).

---

## 12. Benchmarking

`perf/` holds the measurement tools: a shared harness (`load` time, TTFT,
decode/total tok/s, p50/p95 token latency, peak RSS; text/csv/json output)
plus one thin binary per model and a script that compares all backends:

```bash
./build/perf/bench_llama --checkpoint models/llama2/weights/stories15M.bin \
  --tokenizer models/llama2/weights/tokenizer.bin \
  --backend build/models/llama2/plugins/libplugin_llama2_mac_mseries.dylib \
  --prompt "Once upon a time" --steps 128
./build/perf/bench_gemma --checkpoint models/gemma2/weights/tiny-gemma2.bin \
  --tokenizer models/gemma2/weights/tokenizer.bin --prompt "Hello" --steps 50
./perf/compare_backends.sh llama2 "Once upon a time" 128   # all four, one table + CSV
```

Commands in detail: `perf/README.md`. What each number means and how to
compare fairly (warmup, greedy default, two performance regimes with our
M4 data): `SPEC/11_BENCHMARKING.md`.

### Measured results (Apple M4, 16 GB, Release build, greedy, warmup 16, 3 repeats)

Regenerate any time: `./perf/compare_backends.sh llama2 "Once upon a time" 200`.

**Llama-2 — stories15M (60 MB, prompt "Once upon a time", 200 tokens):**

| Backend | Avg decode tok/s | Best decode tok/s | TTFT (mean) |
|---|---|---|---|
| `reference` (built-in) | 175.9 | 194.7 | 25.7 ms |
| `llama2/cpu_baseline` | 197.3 | 198.6 | 24.6 ms |
| `llama2/windows_intel` (scalar on ARM) | 349.9 | 352.2 | 13.8 ms |
| `llama2/mac_mseries` ⭐ | 914.0 | 921.0 | 4.8 ms |

**Gemma2 — tiny-random (8 MB, prompt "Hello", 50 tokens):**

| Backend | Avg decode tok/s | Best decode tok/s | TTFT (mean) |
|---|---|---|---|
| `reference` (built-in) | 944.4 | 946.5 | 1.7 ms |
| `gemma2/cpu_baseline` | 948.6 | 951.0 | 1.6 ms |
| `gemma2/mac_mseries` | 917.0 | 926.7 | 1.8 ms |
| `gemma2/windows_intel` (scalar on ARM) | 1070.0 | 1076.9 | 1.4 ms |

### What "good" looks like

- **Mac plugin ≈ 900+ tok/s on stories15M** is the streaming ceiling
  (~60 MB/token ⇒ ~55 GB/s) — a good result means you're at it, and no
  kernel tweak will go meaningfully past it.
- **Scalar backends at 150–350 tok/s** on the same model is healthy
  (overhead-bound); below ~100 means something regressed — bisect with
  `cpu_baseline` as the reference.
- **Tiny models (≈8 MB) saturate everything near ~1000 tok/s.** All
  backends tying (as in the Gemma2 table) is *expected*, not a signal —
  never claim a backend win from a tiny model; re-run on the biggest
  checkpoint that fits.
- **Projecting to 7B-FP32 (~26 GB/token):** expect ~2–4 tok/s on *every*
  backend on this class of machine. That's the bandwidth wall, and the
  reason quantization is `recommendation.md` #1 — it is the only change
  that moves that number (~4× for INT8).
- **Avg vs best:** report both. They should agree within ~10%; a wide gap
  (see `reference` above: 175.9 vs 194.7) means one repeat caught machine
  noise — re-run before concluding anything.
