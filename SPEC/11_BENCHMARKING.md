# 11 — Benchmarking Methodology

How `perf/` measures, why it measures that way, and how to read the numbers
without fooling yourself. The harness (`perf/.../bench/harness.h`) is the
executable version of this doc.

## What we measure (and why each exists)

```
load → [warmup] → repeat × N → report
```

| Metric | Definition | Why it exists |
|---|---|---|
| `load_s` | `engine.load()` wall time, once | Separates one-time costs (mmap, tokenizer parse, calloc) from steady state. Cold page cache inflates it — that measures your disk/SSD, not kernels. |
| `ttft_s` | start of `generate()` → first token, per repeat | Time-to-first-token = prefill cost. Dominated by prompt length; only comparable at equal `prompt_tokens` (printed by every run). The number users *feel*. |
| `decode_tps` | `(n-1)/(t_last − t_first)` | Steady-state rate with prefill excluded. The headline number. For big models: `≈ bandwidth / bytes-per-token` — no kernel work moves it (see `08`). |
| `total_tps` | `n/total` | What a user observes end-to-end for short generations (prefill included). |
| `token_ms p50/p95` | per-token latencies, first token excluded | p95 exposes hiccups (thread spawn, GCD contention, thermal events) that means hide. |
| `peak_rss_mb` | max RSS after load+runs | Capacity planning: weights + KV cache + activations must fit. (macOS reports bytes, Linux kilobytes, Windows working set — normalized in code.) |

## Decisions (and the reasoning)

1. **Warmup before measuring (default 16 tokens).** First tokens pay
   one-time costs: page faults on freshly-mmapped weights, branch predictor
   + cache warmup, GCD thread-pool spin-up, Accelerate kernel selection.
   Warmup is *untimed and unreported*; `--warmup 0` disables it when you
   explicitly want cold-start numbers (e.g. measuring `madvise` effects).
2. **Greedy by default (`temperature 0`).** Sampling draws RNG per token and
   early-EOS stops runs short — both inject run-to-run variance that has
   nothing to do with engine speed. Deterministic decoding makes repeats
   comparable; use `--temperature` only when benchmarking the *sampler*.
3. **First token excluded from decode stats.** TTFT contains the entire
   prefill (a GEMV per prompt token); folding it into per-token percentiles
   would drown the steady-state signal. It gets its own metric instead.
4. **Best-of-repeats AND mean reported.** Best approximates the machine's
   ceiling (noise only ever slows runs down); mean shows stability. If they
   disagree wildly, the machine is throttling or loaded — fix the
   environment, not the code.
5. **No threads pinned, no realtime priority.** We measure the system as
   users experience it. Pinning would flatter numbers and hide contention
   the real deployment also has.

## Reading numbers: two regimes (with our own data)

M4, stories15M (60 MB), greedy, prompt "Once upon a time":

| Backend | decode tok/s | Regime |
|---|---|---|
| reference / cpu_baseline | ~205 | overhead-bound: scalar loops, transcendentals |
| windows_intel (scalar on ARM) | ~368 | fewer overheads, still latency-bound |
| mac_mseries | ~962 | near the streaming ceiling for 60 MB/token |

Same machine, tiny-gemma2 (8 MB): *all* backends land ~950–1090 tok/s —
the model is so small that fixed overheads (virtual calls, GCD dispatch,
thread checks) dominate and kernels don't matter. Lesson: **never conclude
"backend X is as fast as Y" from a tiny model**; re-run on the biggest
model that fits before claiming a speedup. Conversely, 7B-FP32 (~26 GB)
is pure bandwidth-bound everywhere: expect ~2–4 tok/s on *every* backend
until quantization lands (`recommendation.md` #1).

## Comparing fairly (checklist)

- Same machine, same power state (plug in laptops; watch for thermal
  throttling on long runs), same prompt, same `--steps`, same checkpoint.
- Close browsers/compilers; check `mean` vs `best` agreement.
- Report `prompt_tokens`: TTFT scales with it, decode rate shouldn't
  (until contexts get long enough for attention to matter — then say so).
- Keep `--temperature 0` unless the experiment is *about* sampling.

## Extending (no harness changes)

New model? Adapt-only: copy a `bench_<m>.cc` (~70 lines, touches just
`EngineConfig`/`load`/`generate`/`backend_name`/`count_prompt_tokens`),
one `model_bench(...)` line, one `case` in `compare_backends.sh`.
New metric? Add it to `BenchResult` + all three reporters together —
a metric invisible in one format rots. New backend? Zero perf-code
changes: it appears in the compare script automatically once built.
