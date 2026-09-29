// cpu_baseline plugin (Gemma2): shared scalar reference, repackaged.
// No overrides — correctness baseline (see SPEC/10_GEMMA2.md).

#include "inference/core/reference_backend.h"

namespace inference {
namespace gemma2 {
namespace plugin {
namespace cpu {

class CpuBaselineBackend final : public core::ReferenceBackend {
 public:
  std::string name() const override { return "gemma2/cpu_baseline"; }
  std::string device_info() const override {
    return "portable scalar reference";
  }
};

}  // namespace cpu
}  // namespace plugin
}  // namespace gemma2
}  // namespace inference

extern "C" {
inference::core::IComputeBackend* CreateBackend() {
  return new inference::gemma2::plugin::cpu::CpuBaselineBackend();
}
void DestroyBackend(inference::core::IComputeBackend* p) { delete p; }
}
