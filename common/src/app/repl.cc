// Interactive REPL: stdin loop shared by all model CLIs.

#include "inference/app/repl.h"

#include <cctype>
#include <cstdio>
#include <iostream>
#include <string>

namespace inference {
namespace app {
namespace {

bool is_blank(const std::string& s) {
  for (char c : s)
    if (!std::isspace((unsigned char)c)) return false;
  return true;
}

bool is_quit(const std::string& s) {
  return s == "quit" || s == "/quit" || s == "exit" || s == ":q";
}

}  // namespace

void run_repl(
    const std::string& label,
    const std::function<void(
        const std::string& input,
        const std::function<void(const std::string&)>& emit)>& on_input) {
  std::string line;
  for (;;) {
    std::fputs(label.c_str(), stdout);
    std::fflush(stdout);
    if (!std::getline(std::cin, line)) {
      std::printf("\n");
      break;  // EOF (Ctrl+D) — clean exit, newline keeps the shell tidy
    }
    // Strip a trailing '\r' so Windows line endings don't reach the model.
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (is_quit(line)) break;
    if (is_blank(line)) continue;
    on_input(line, [](const std::string& piece) {
      std::fwrite(piece.data(), 1, piece.size(), stdout);
      std::fflush(stdout);
    });
    std::printf("\n");
    std::fflush(stdout);
  }
}

}  // namespace app
}  // namespace inference
