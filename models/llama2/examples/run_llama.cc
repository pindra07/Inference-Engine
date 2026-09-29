// run_llama: thin CLI over LlamaEngine. All flags optional except
// --checkpoint and --tokenizer.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

#include "inference/app/repl.h"
#include "inference/llama2/engine.h"

namespace {

void usage(const char* argv0) {
  std::printf(
      "Usage: %s --checkpoint MODEL.bin --tokenizer TOK.bin [opts]\n"
      "  --backend PATH     plugin .dylib/.so/.dll (default: built-in CPU)\n"
      "  --prompt STR       input prompt (default: \"Hello\")\n"
      "  --chat             wrap prompt in Llama-2-Chat [INST] template\n"
      "  --interactive, -i  chat loop: type a line, get a response (quit/EOF ends)\n"
      "  --steps N          max new tokens (default 128)\n"
      "  --temperature F    0=greedy (default 0.7)\n"
      "  --top_p F          (default 0.9)\n"
      "  --top_k N          (default 0=off)\n"
      "  --seed N           (default 42)\n"
      "  --seq_len N        override max context\n"
      "  --help             this message\n",
      argv0);
}

std::string chat_wrap(const std::string& user) {
  return "<s>[INST] " + user + " [/INST]";
}

}  // namespace

int main(int argc, char** argv) {
  inference::llama2::EngineConfig cfg;
  std::string prompt = "Hello";
  bool chat = false;
  bool interactive = false;
  int steps = 128;
  cfg.sample.temperature = 0.7f;

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
      prompt = need("--prompt");
    else if (a == "--chat")
      chat = true;
    else if (a == "--interactive" || a == "-i")
      interactive = true;
    else if (a == "--steps")
      steps = std::stoi(need("--steps"));
    else if (a == "--temperature")
      cfg.sample.temperature = std::stof(need("--temperature"));
    else if (a == "--top_p")
      cfg.sample.top_p = std::stof(need("--top_p"));
    else if (a == "--top_k")
      cfg.sample.top_k = std::stoi(need("--top_k"));
    else if (a == "--seed")
      cfg.sample.seed = (uint64_t)std::stoull(need("--seed"));
    else if (a == "--seq_len")
      cfg.seq_len_override = std::stoi(need("--seq_len"));
    else if (a == "--help" || a == "-h") {
      usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
      usage(argv[0]);
      return 1;
    }
  }
  if (cfg.checkpoint.empty() || cfg.tokenizer.empty()) {
    usage(argv[0]);
    return 1;
  }
  // INFERENCE_BACKEND env overrides when --backend not passed.
  if (cfg.backend_path.empty()) {
    if (const char* e = std::getenv("INFERENCE_BACKEND")) cfg.backend_path = e;
  }
  if (chat) prompt = chat_wrap(prompt);

  try {
    inference::llama2::LlamaEngine engine;
    engine.load(cfg);
    std::fprintf(stderr, "backend=%s prompt=\"%s\"\n",
                 engine.backend_name().c_str(), prompt.c_str());
    if (interactive) {
      // Each typed line is a fresh prompt (no cross-turn memory — the KV
      // cache restarts every generate call). Ctrl+D or "quit" ends.
      std::fprintf(stderr, "[interactive: type text, Enter for response]\n");
      inference::app::run_repl(
          "> ", [&](const std::string& line,
                    const std::function<void(const std::string&)>& emit) {
            std::string in = chat ? chat_wrap(line) : line;
            engine.generate(in, steps, [&](int, const std::string& piece) {
              emit(piece);
              return true;
            });
          });
      return 0;
    }
    auto t0 = std::chrono::steady_clock::now();
    int n_tok = 0;
    engine.generate(prompt, steps, [&](int, const std::string& piece) {
      std::fwrite(piece.data(), 1, piece.size(), stdout);
      std::fflush(stdout);
      ++n_tok;
      return true;
    });
    auto t1 = std::chrono::steady_clock::now();
    double secs =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() /
        1000.0;
    std::printf("\n\n[done: %d tokens in %.2fs (%.1f tok/s) backend=%s]\n",
                n_tok, secs, n_tok / (secs > 0 ? secs : 1e-9),
                engine.backend_name().c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
