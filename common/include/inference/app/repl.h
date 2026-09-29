#pragma once
// Shared interactive chat loop for model CLIs: print a label, read a line,
// generate the response, repeat. Engines stay non-interactive; all stdin
// handling lives here so every model's CLI behaves identically.

#include <functional>
#include <string>

namespace inference {
namespace app {

// Runs until EOF (Ctrl+D) or a quit line ("quit", "/quit", "exit", ":q").
// Blank lines are skipped, never sent to the model.
// `on_input` receives the raw line and an `emit` callback for response
// pieces (the CLI writes them to stdout).
void run_repl(
    const std::string& label,
    const std::function<void(
        const std::string& input,
        const std::function<void(const std::string&)>& emit)>& on_input);

}  // namespace app
}  // namespace inference
