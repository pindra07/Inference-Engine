// cpu_baseline plugin: the shared scalar reference, repackaged as a plugin.
// No overrides — correctness baseline for validating accelerated plugins
// (all backends must agree within 1e-4; see SPEC/08_OPTIMIZATION.md).

#include "inference/core/reference_backend.h"

namespace inference {
namespace llama2 {
namespace plugin {
namespace cpu {

class CpuBaselineBackend final : public core::ReferenceBackend {
 public:
  std::string name() const override { return "llama2/cpu_baseline"; }
  std::string device_info() const override {
    return "portable scalar reference";
  }
};

}  // namespace cpu
}  // namespace plugin
}  // namespace llama2
}  // namespace inference

extern "C" {
inference::core::IComputeBackend* CreateBackend() {
  return new inference::llama2::plugin::cpu::CpuBaselineBackend();
}
void DestroyBackend(inference::core::IComputeBackend* p) { delete p; }
}
