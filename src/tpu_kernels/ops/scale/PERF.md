# Scale (×2)

Target: ≥80% HBM BW @ (M=8192, N=8192), bf16  (regression floor: 80% — see tests/perf/test_scale.py)
Current: 81.6% HBM BW as of 2026-04-30 (block_shape=(512, 1024), v5e 1 chip; timing="device").
Bottleneck: at the v5e single-chip plateau for naive copy. Device-clock measurements across the (bm, bn) sweep cluster between 81 and 82% across the top 12 blocks; once the tile is "big enough," the choice barely matters. The previously-stated 85% target was set before a device-timing harness existed — under direct kernel-clock measurement it isn't reachable on v5e for this kernel shape, so the target tightens to 80% as a realistic ceiling. The remaining ~18pp gap to 100% HBM BW is peak-vs-sustained + DMA setup overhead, not a kernel-structure issue.
Next: scale is done — leave it as the harness's roofline-validation primer and move on. Revisit if the kernel structure changes (e.g. a streaming variant).
