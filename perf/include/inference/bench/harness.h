#pragma once
// Shared benchmarking harness: timing, statistics, reporting.
// Engine-agnostic — each model's bench tool adapts its engine to GenerateFn.
// Methodology: SPEC/11_BENCHMARKING.md.

#include <functional>
#include <string>
#include <vector>

namespace inference {
namespace bench {

// What to measure. Defaults favor stable, comparable numbers:
// greedy decoding (temperature 0) removes sampling noise.
struct BenchConfig {
  std::string checkpoint;
  std::string tokenizer;
  std::string backend_path;  // empty => built-in backend
  std::string prompt = "Once upon a time";
  int steps = 128;          // new tokens per measured repeat
  int warmup_steps = 16;    // untimed warmup tokens (0 = skip)
  int repeats = 3;          // measured repeats
  float temperature = 0.0f;
  float top_p = 0.9f;
  int top_k = 0;
  uint64_t seed = 42;
};

// One timed generation. on_token(id, piece) mirrors the engine API;
// return false from it to stop early (counts as a short repeat).
using GenerateFn = std::function<void(
    const std::string& prompt, int steps,
    const std::function<bool(int, const std::string&)>& on_token)>;

struct RepeatStats {
  int gen_tokens = 0;
  double ttft_s = 0;      // start -> first token (includes prefill)
  double decode_tps = 0;  // (n-1)/(last-first) for n>1, else total rate
  double total_tps = 0;   // n/total wall time
  std::vector<double> token_ms;  // per-token latency, ms
};

struct BenchResult {
  std::string model;    // e.g. "llama2"
  std::string backend;  // backend name() string
  double load_s = 0;    // one-time engine.load() wall time
  long peak_rss_mb = 0;
  int prompt_tokens = 0;  // informational (set by caller when known)
  std::vector<RepeatStats> repeats;

  double mean_decode_tps() const;
  double best_decode_tps() const;
  double mean_ttft_s() const;
  // p50/p95 over ALL measured token latencies (first token excluded:
  // it contains prefill — see SPEC/11 for why).
  double p50_decode_ms() const;
  double p95_decode_ms() const;
};

// Runs warmup (untimed) + repeats (timed). load_s/secondaries set by caller.
BenchResult run_benchmark(const BenchConfig& cfg, const std::string& model,
                          const std::string& backend, GenerateFn generate);

// Reporters (stdout).
void print_text(const BenchResult& r, const BenchConfig& cfg);
void print_csv(const BenchResult& r, const BenchConfig& cfg);  // one row/repeat
void print_json(const BenchResult& r, const BenchConfig& cfg);

// Parses common CLI flags (--checkpoint/--tokenizer/--backend/--prompt/
// --steps/--warmup/--repeats/--temperature/--top_p/--top_k/--seed/
// --format text|csv|json/--help). prog is argv[0] for usage text.
// Returns false when the caller should exit (help shown or error printed).
// Note: no --chat flag by design — benchmarks use raw prompts so numbers
// stay comparable; embed any template in --prompt if needed.
bool parse_bench_args(int argc, char** argv, const std::string& prog,
                      BenchConfig& cfg, std::string& format);

// Peak RSS in bytes, portable (getrusage / GetProcessMemoryInfo).
long peak_rss_bytes();

}  // namespace bench
}  // namespace inference
