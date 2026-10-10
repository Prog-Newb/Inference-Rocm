# runtime-rocm

AMD Radeon (Windows, ROCm/HIP) Candle runtime bundle — the ROCm sibling of `runtime-cuda`.

* **Scope today:** SeedVR2 (`seedvr2`, `seedvr2_3b`, `seedvr2_7b`) only. No LLM or audio lane.
* **Why smaller than `runtime-cuda`:** only providers validated on ROCm are composed in, so a ROCm
  worker never advertises something it cannot run. Add a provider here once it is validated.
* **Requires:** the ROCm-capable `candle-core`/`candle-nn` (workspace `[patch]`), the HIP SDK runtime
  libraries on `PATH`, and an RDNA3 Radeon (gfx1100 validated).
* Does not replace or alter `runtime-cuda`, `runtime-cpu` or `runtime-macos`.