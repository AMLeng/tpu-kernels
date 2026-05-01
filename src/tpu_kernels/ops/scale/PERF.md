# Scale (×2)

Target: ≥85% HBM BW @ (M=8192, N=8192), bf16  (regression floor: 78% — see tests/perf/test_scale.py)
Current: 80.9% HBM BW as of 2026-04-30 (block_shape=(512, 1024), v5e 1 chip; timing="unroll", k=32 auto-sized).
Bottleneck: small and unclear. The previously-reported 63% / "presumed single-buffered DMA" was almost entirely host-side dispatch overhead being timed alongside the kernel; with corrected unroll-mode timing the top half of the (bm, bn) sweep clusters tightly between 80 and 81% and the choice of block barely matters. The remaining ~4pp gap to 85% is small enough to be residual unroll-mode amortization noise, v5e HBM peak-vs-sustained, or actual block tiling — current data can't distinguish.
Next: bench with `timing="device"` to read kernel time off the TPU hardware clock (no host overhead amortized in). If device BW% still comes in below 85%, the gap is structural and a triple-buffered `emit_pipeline` rewrite is the next lever; if device BW% clears 85%, the kernel was already at the v5e ceiling and the target was always reachable.
