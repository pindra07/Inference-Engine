// bench_gemma: benchmark GemmaEngine through the shared harness.
// See perf/README.md for commands, SPEC/11_BENCHMARKING.md for methodology.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

#include "inference/bench/harness.h"
#include "inference/gemma2/engine.h"

int main(int argc, char** argv) {
  namespace bench = inference::bench;
  bench::BenchConfig cfg;
  std::string format;
  if (!bench::parse_bench_args(argc, argv, argv[0], cfg, format)) return 1;
  if (cfg.backend_path.empty()) {
    if (const char* e = std::getenv("INFERENCE_BACKEND")) cfg.backend_path = e;
  }

  inference::gemma2::EngineConfig ecfg;
  ecfg.checkpoint = cfg.checkpoint;
  ecfg.tokenizer = cfg.tokenizer;
  ecfg.backend_path = cfg.backend_path;
  ecfg.sample.temperature = cfg.temperature;
  ecfg.sample.top_p = cfg.top_p;
  ecfg.sample.top_k = cfg.top_k;
  ecfg.sample.seed = cfg.seed;

  try {
    inference::gemma2::GemmaEngine engine;
    auto t0 = std::chrono::steady_clock::now();
    engine.load(ecfg);
    double load_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();

    bench::BenchResult r = bench::run_benchmark(
        cfg, "gemma2", engine.backend_name(),
        [&](const std::string& prompt, int steps,
            const std::function<bool(int, const std::string&)>& on_token) {
          engine.generate(prompt, steps, on_token);
        });
    r.load_s = load_s;
    r.prompt_tokens = engine.count_prompt_tokens(cfg.prompt);
    r.peak_rss_mb = bench::peak_rss_bytes() / (1024 * 1024);

    if (format == "csv")
      bench::print_csv(r, cfg);
    else if (format == "json")
      bench::print_json(r, cfg);
    else
      bench::print_text(r, cfg);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
