// Shared benchmark harness: steady-clock timing around generate(),
// percentile stats, text/csv/json reporters.

#include "inference/bench/harness.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>

#ifdef _WIN32
#include <windows.h>

#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace inference {
namespace bench {
namespace {

using Clock = std::chrono::steady_clock;

double mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// Nearest-rank percentile over a sorted vector.
double percentile(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) return 0.0;
  size_t i = (size_t)std::ceil(p / 100.0 * sorted.size());
  i = std::min(i < 1 ? 1 : i, sorted.size()) - 1;
  return sorted[i];
}

std::vector<double> all_decode_ms(const BenchResult& r) {
  std::vector<double> v;
  for (const auto& rep : r.repeats)
    for (size_t i = 1; i < rep.token_ms.size(); ++i) v.push_back(rep.token_ms[i]);
  std::sort(v.begin(), v.end());
  return v;
}

}  // namespace

long peak_rss_bytes() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS c{};
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &c, sizeof(c))) return -1;
  return (long)c.PeakWorkingSetSize;
#else
  struct rusage u{};
  if (getrusage(RUSAGE_SELF, &u) != 0) return -1;
#if defined(__APPLE__)
  return (long)u.ru_maxrss;  // bytes on macOS
#else
  return (long)u.ru_maxrss * 1024;  // kilobytes on Linux
#endif
#endif
}

BenchResult run_benchmark(const BenchConfig& cfg, const std::string& model,
                          const std::string& backend, GenerateFn generate) {
  BenchResult r;
  r.model = model;
  r.backend = backend;

  if (cfg.warmup_steps > 0) {
    generate(cfg.prompt, cfg.warmup_steps,
             [](int, const std::string&) { return true; });
  }
  for (int i = 0; i < cfg.repeats; ++i) {
    RepeatStats s;
    auto t0 = Clock::now();
    std::vector<Clock::time_point> stamps;
    stamps.reserve((size_t)cfg.steps + 1);
    generate(cfg.prompt, cfg.steps, [&](int, const std::string&) {
      stamps.push_back(Clock::now());
      return true;
    });
    auto sec = [](Clock::time_point a, Clock::time_point b) {
      return std::chrono::duration<double>(b - a).count();
    };
    s.gen_tokens = (int)stamps.size();
    Clock::time_point prev = t0;
    for (auto tp : stamps) {
      s.token_ms.push_back(sec(prev, tp) * 1000.0);
      prev = tp;
    }
    if (!stamps.empty()) {
      s.ttft_s = sec(t0, stamps.front());
      double total = sec(t0, stamps.back());
      s.total_tps = s.gen_tokens / (total > 0 ? total : 1e-9);
      s.decode_tps = s.gen_tokens > 1
                         ? (s.gen_tokens - 1) / (total - s.ttft_s > 0
                                                     ? total - s.ttft_s
                                                     : 1e-9)
                         : s.total_tps;
    }
    r.repeats.push_back(std::move(s));
  }
  return r;
}

double BenchResult::mean_decode_tps() const {
  std::vector<double> v;
  for (const auto& r : repeats) v.push_back(r.decode_tps);
  return mean(v);
}

double BenchResult::best_decode_tps() const {
  double b = 0.0;
  for (const auto& r : repeats) b = std::max(b, r.decode_tps);
  return b;
}

double BenchResult::mean_ttft_s() const {
  std::vector<double> v;
  for (const auto& r : repeats) v.push_back(r.ttft_s);
  return mean(v);
}

double BenchResult::p50_decode_ms() const {
  return percentile(all_decode_ms(*this), 50.0);
}

double BenchResult::p95_decode_ms() const {
  return percentile(all_decode_ms(*this), 95.0);
}

void print_text(const BenchResult& r, const BenchConfig& cfg) {
  std::printf("model=%s backend=%s prompt_tokens=%d gen=%dx%d warmup=%d\n",
              r.model.c_str(), r.backend.c_str(), r.prompt_tokens, cfg.steps,
              cfg.repeats, cfg.warmup_steps);
  std::printf("load=%.2fs peak_rss=%ldMB\n", r.load_s, r.peak_rss_mb);
  std::printf("%-6s %-10s %-10s %-10s %-10s\n", "repeat", "ttft_ms",
              "decode_t/s", "total_t/s", "tokens");
  for (size_t i = 0; i < r.repeats.size(); ++i) {
    const auto& s = r.repeats[i];
    std::printf("%-6zu %-10.1f %-10.1f %-10.1f %-10d\n", i, s.ttft_s * 1000.0,
                s.decode_tps, s.total_tps, s.gen_tokens);
  }
  std::printf(
      "summary: decode_t/s mean=%.1f best=%.1f | ttft mean=%.1fms | "
      "token_ms p50=%.2f p95=%.2f\n",
      r.mean_decode_tps(), r.best_decode_tps(), r.mean_ttft_s() * 1000.0,
      r.p50_decode_ms(), r.p95_decode_ms());
}

void print_csv(const BenchResult& r, const BenchConfig& cfg) {
  std::printf("model,backend,prompt_tokens,steps,repeat,gen_tokens,ttft_s,"
              "decode_tps,total_tps,load_s,peak_rss_mb\n");
  for (size_t i = 0; i < r.repeats.size(); ++i) {
    const auto& s = r.repeats[i];
    std::printf("%s,%s,%d,%d,%zu,%d,%.4f,%.2f,%.2f,%.2f,%ld\n", r.model.c_str(),
                r.backend.c_str(), r.prompt_tokens, cfg.steps, i,
                s.gen_tokens, s.ttft_s, s.decode_tps, s.total_tps, r.load_s,
                r.peak_rss_mb);
  }
}

void print_json(const BenchResult& r, const BenchConfig& cfg) {  std::printf("{\"model\":\"%s\",\"backend\":\"%s\",\"prompt_tokens\":%d,"
              "\"steps\":%d,\"temperature\":%.2f,\"load_s\":%.3f,"
              "\"peak_rss_mb\":%ld,\"mean_decode_tps\":%.2f,"
              "\"best_decode_tps\":%.2f,\"mean_ttft_s\":%.4f,"
              "\"p50_decode_ms\":%.3f,\"p95_decode_ms\":%.3f,\"repeats\":[",
              r.model.c_str(), r.backend.c_str(), r.prompt_tokens, cfg.steps,
              cfg.temperature, r.load_s, r.peak_rss_mb, r.mean_decode_tps(),
              r.best_decode_tps(), r.mean_ttft_s(), r.p50_decode_ms(),
              r.p95_decode_ms());
  for (size_t i = 0; i < r.repeats.size(); ++i) {
    const auto& s = r.repeats[i];
    std::printf("%s{\"gen_tokens\":%d,\"ttft_s\":%.4f,\"decode_tps\":%.2f,"
                "\"total_tps\":%.2f}",
                i ? "," : "", s.gen_tokens, s.ttft_s, s.decode_tps,
                s.total_tps);
  }
  std::printf("]}\n");
}

bool parse_bench_args(int argc, char** argv, const std::string& prog,
                      BenchConfig& cfg, std::string& format) {
  auto usage = [&] {
    std::printf(
        "Usage: %s --checkpoint FILE --tokenizer FILE [opts]\n"
        "  --backend PATH   plugin (default: built-in)\n"
        "  --prompt STR     benchmark prompt (default: \"%s\")\n"
        "  --steps N        new tokens per repeat (default 128)\n"
        "  --warmup N       untimed warmup tokens (default 16, 0=skip)\n"
        "  --repeats N      measured repeats (default 3)\n"
        "  --temperature F  (default 0.0=greedy, recommended for benches)\n"
        "  --top_p F        (default 0.9)\n"
        "  --top_k N        (default 0=off)\n"
        "  --seed N         (default 42)\n"
        "  --format S       text|csv|json (default text)\n",
        prog.c_str(), cfg.prompt.c_str());
  };
  format = "text";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", flag);
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--checkpoint")
      cfg.checkpoint = need("--checkpoint");
    else if (a == "--tokenizer")
      cfg.tokenizer = need("--tokenizer");
    else if (a == "--backend")
      cfg.backend_path = need("--backend");
    else if (a == "--prompt")
      cfg.prompt = need("--prompt");
    else if (a == "--steps")
      cfg.steps = std::stoi(need("--steps"));
    else if (a == "--warmup")
      cfg.warmup_steps = std::stoi(need("--warmup"));
    else if (a == "--repeats")
      cfg.repeats = std::stoi(need("--repeats"));
    else if (a == "--temperature")
      cfg.temperature = std::stof(need("--temperature"));
    else if (a == "--top_p")
      cfg.top_p = std::stof(need("--top_p"));
    else if (a == "--top_k")
      cfg.top_k = std::stoi(need("--top_k"));
    else if (a == "--seed")
      cfg.seed = (uint64_t)std::stoull(need("--seed"));
    else if (a == "--format")
      format = need("--format");
    else if (a == "--help" || a == "-h") {
      usage();
      return false;
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
      usage();
      return false;
    }
  }
  if (cfg.checkpoint.empty() || cfg.tokenizer.empty() || cfg.steps <= 0 ||
      cfg.repeats <= 0 || cfg.warmup_steps < 0 ||
      (format != "text" && format != "csv" && format != "json")) {
    usage();
    return false;
  }
  return true;
}

}  // namespace bench
}  // namespace inference
