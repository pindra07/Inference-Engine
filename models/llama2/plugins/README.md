# Plugins (Llama-2) — one folder per hardware target.
# Each folder builds a shared lib exporting CreateBackend/DestroyBackend.
# See SPEC/04_PLUGIN_API.md. New plugins subclass core::ReferenceBackend
# and override only accelerated kernels.

- `cpu_baseline/` — zero overrides: the reference, repackaged as a plugin.
- `mac_mseries/` — Apple Silicon (M1–M4). Links Accelerate.framework + GCD.
- `windows_intel/` — x86-64 Intel/AMD. AVX2+FMA when available, thread-pool.
