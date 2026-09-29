# Plugins (Gemma2) — one folder per hardware target.
# Each folder builds a shared lib exporting CreateBackend/DestroyBackend.
# See SPEC/04_PLUGIN_API.md (v1.1) and SPEC/10_GEMMA2.md. All plugins subclass
# core::ReferenceBackend and override only accelerated kernels.

- `cpu_baseline/` — zero overrides: the reference, repackaged as a plugin.
- `mac_mseries/` — Apple Silicon (M1–M4). Accelerate + GCD; overrides
  attention_softcap and tanh_softcap (vForce), inherits gelu_tanh scalar.
- `windows_intel/` — x86-64 Intel/AMD. AVX2+FMA matvec/norm/elem_mul;
  attention_softcap inherited scalar (matvec is ~95% of runtime).
