# perf/ — benchmarking the inference engines

Modular perf tooling: one shared harness + one thin binary per model + one
compare script. Methodology (what each number means, how to compare fairly)
lives in `SPEC/11_BENCHMARKING.md`.

```
perf/
├── README.md                    # this file (commands)
├── CMakeLists.txt               # bench_common lib + bench_<model> binaries
├── include/inference/bench/
│   └── harness.h                # BenchConfig, BenchResult, GenerateFn,
│                                #   run_benchmark(), text/csv/json reporters,
│                                #   parse_bench_args(), peak_rss_bytes()
├── src/harness.cc               # timing loop, percentile stats, reporters
├── llama2/bench_llama.cc        # adapts LlamaEngine (~70 lines)
├── gemma2/bench_gemma.cc        # adapts GemmaEngine (~70 lines)
├── compare_backends.sh          # all backends, one table (see below)
└── results/                     # CSV output lands here (gitignored)
```

Binaries land in `build/perf/`: `bench_llama`, `bench_gemma`.

## Quick commands

```bash
cmake --build build -j   # builds bench_* alongside everything else
```

**Single run, human-readable table (Llama-2, Mac plugin):**
```bash
./build/perf/bench_llama \
  --checkpoint models/llama2/weights/stories15M.bin \
  --tokenizer models/llama2/weights/tokenizer.bin \
  --backend build/models/llama2/plugins/libplugin_llama2_mac_mseries.dylib \
  --prompt "Once upon a time" --steps 128
# model=llama2 backend=llama2/mac_mseries prompt_tokens=5 gen=128x3 warmup=16
# load=0.04s peak_rss=73MB
# repeat ttft_ms    decode_t/s total_t/s  tokens
# 0      4.9        898.2      868.4      100
# ...
# summary: decode_t/s mean=915.2 best=932.1 | ttft mean=5.0ms | token_ms p50=1.05 p95=1.29
```

**Same, Gemma2 on the built-in backend:**
```bash
./build/perf/bench_gemma \
  --checkpoint models/gemma2/weights/tiny-gemma2.bin \
  --tokenizer models/gemma2/weights/tokenizer.bin \
  --prompt "Hello" --steps 50
```

**Machine-readable output (-logging, plotting, CI):**
```bash
./build/perf/bench_llama --checkpoint ... --tokenizer ... \
  --format csv >> perf/results/llama2-mac.csv
./build/perf/bench_llama --checkpoint ... --tokenizer ... \
  --format json | python3 -m json.tool | head -20
```

**Compare all four backends with one command:**
```bash
./perf/compare_backends.sh llama2 "Once upon a time" 128
./perf/compare_backends.sh gemma2 "Hello" 50
# ---- summary (best of repeats) ----
# reference      best    208.3 tok/s | ttft mean    23.3 ms
# cpu_baseline   best    204.4 tok/s | ttft mean    24.3 ms
# mac_mseries    best    961.8 tok/s | ttft mean     4.9 ms
# windows_intel  best    368.1 tok/s | ttft mean    13.3 ms
# full per-repeat data: perf/results/llama2-20260929-225204.csv
```

Defaults use the in-repo test checkpoints; point at bigger models with
environment overrides (paths must exist):
```bash
CKPT=models/llama2/weights/llama2-7b.bin TOK=models/llama2/weights/tokenizer.bin \
  ./perf/compare_backends.sh llama2 "Explain recursion." 256
```

## All flags (`bench_llama --help`, identical for `bench_gemma`)

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint`, `--tokenizer` | (required) | model files |
| `--backend PATH` | built-in | plugin (or `INFERENCE_BACKEND` env) |
| `--prompt STR` | `"Once upon a time"` | benchmark prompt (fixed across runs being compared) |
| `--steps N` | 128 | new tokens per measured repeat |
| `--warmup N` | 16 | untimed warmup tokens (`0` = skip; see SPEC/11 for why warmup exists) |
| `--repeats N` | 3 | measured repeats |
| `--temperature F` | 0.0 | greedy by default — removes sampling noise between runs |
| `--top_p`, `--top_k`, `--seed` | 0.9 / 0 / 42 | sampler settings, passed through |
| `--format S` | `text` | `text` (table), `csv` (one row/repeat), `json` (full) |

## What the numbers mean (short version, long version in SPEC/11)

- **`load`** — one-time `engine.load()` (mmap + tokenizer + alloc). Cold page
  cache makes the first-ever run slower; that's the OS, not the engine.
- **`ttft`** — time-to-first-token *including prefill*. Grows with prompt
  length; compare only at equal prompt-token counts (`prompt_tokens` printed).
- **`decode_t/s`** — steady-state generation rate, prefill excluded. This is
  the headline number — for big models it equals
  `bandwidth / bytes-per-token`.
- **`p50/p95 token_ms`** — per-token latency distribution (first token
  excluded); p95 catches threading/GC-style hiccups text averages hide.
- **`peak_rss`** — peak process memory (weights + KV cache + activations).

## Adding a model (checklist)

1. Copy `perf/gemma2/bench_gemma.cc` → `perf/<m>/bench_<m>.cc`, swap the
   engine types/namespace (it only touches `EngineConfig`, `load`,
   `generate`, `backend_name`, `count_prompt_tokens`).
2. One line in `perf/CMakeLists.txt`:
   `model_bench(bench_<m> <m>/bench_<m>.cc inference_<m>)`.
3. Extend the `case` in `compare_backends.sh` with default `CKPT`/`TOK`.
No harness changes needed — new models only adapt, never modify.
