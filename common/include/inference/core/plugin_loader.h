#pragma once
// dlopen/LoadLibrary wrapper. Cold path only. Stateless: OS handles are
// intentionally kept open for process lifetime (a live backend may reference
// plugin code; dlclose under it is UB), so there is nothing to clean up.

#include <memory>
#include <string>

#include "inference/core/backend_plugin.h"

namespace inference {
namespace core {

// Owning pointer with plugin-aware deleter (calls DestroyBackend).
using BackendPtr = std::unique_ptr<IComputeBackend, DestroyBackendFn>;

class PluginLoader {
 public:
  PluginLoader() = default;

  PluginLoader(const PluginLoader&) = delete;
  PluginLoader& operator=(const PluginLoader&) = delete;

  // Loads shared lib, resolves CreateBackend, returns owned backend.
  // Throws std::runtime_error on failure.
  BackendPtr load(const std::string& path);
};

}  // namespace core
}  // namespace inference
