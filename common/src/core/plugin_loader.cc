// dlopen / LoadLibrary wrapper (cold path only).

#include "inference/core/plugin_loader.h"

#include <mutex>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace inference {
namespace core {
namespace {

// Never closed (see header). A handful of leaked handles per process is
// intentional and documented.
std::mutex& leaked_handles_mutex() {
  static std::mutex m;
  return m;
}
std::vector<void*>& leaked_handles() {
  static std::vector<void*> v;
  return v;
}

#ifdef _WIN32
void close_handle(void* h) { FreeLibrary((HMODULE)h); }
#else
void close_handle(void* h) { dlclose(h); }
#endif

}  // namespace

BackendPtr PluginLoader::load(const std::string& path) {
  void* handle = nullptr;
  CreateBackendFn create = nullptr;
  DestroyBackendFn destroy = nullptr;
#ifdef _WIN32
  handle = (void*)LoadLibraryA(path.c_str());
  if (!handle) throw std::runtime_error("LoadLibrary failed: " + path);
  create = (CreateBackendFn)GetProcAddress((HMODULE)handle, "CreateBackend");
  destroy =
      (DestroyBackendFn)GetProcAddress((HMODULE)handle, "DestroyBackend");
#else
  handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
  create = (CreateBackendFn)dlsym(handle, "CreateBackend");
  destroy = (DestroyBackendFn)dlsym(handle, "DestroyBackend");
#endif
  if (!create || !destroy) {
    close_handle(handle);
    throw std::runtime_error(
        "plugin missing CreateBackend/DestroyBackend: " + path);
  }
  IComputeBackend* raw = create();
  if (!raw) {
    close_handle(handle);
    throw std::runtime_error("CreateBackend returned null: " + path);
  }
  {
    std::lock_guard<std::mutex> lock(leaked_handles_mutex());
    leaked_handles().push_back(handle);
  }
  return BackendPtr(raw, destroy);
}

}  // namespace core
}  // namespace inference
