# Scale (×2)

Target: ≥85% HBM BW @ (M=8192, N=8192), bf16  (regression floor: 60% — see tests/perf/test_scale.py)
Current: 63.6% HBM BW as of 2026-04-30 (block_shape=(512, 1024), v5e 1 chip).
Bottleneck: presumed single-buffered DMA — a 5×5 sweep of (bm, bn) ∈ {128..2048}² plateaus at 63–64% across the top nine blocks (within ~1pp), so structure (not block size) is the gap. The copy-style ceiling on v5e is unverified, so we don't yet know whether this plateau is the kernel or the hardware.
Next: rewrite with emit_pipeline + triple-buffering (prefetch N+1, compute N, drain N-1). If that climbs into the 80s the gap is structural; if it stalls near 63% we've found the ceiling.
