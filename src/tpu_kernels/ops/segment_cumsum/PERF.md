# Segment cumsum

Target: ≥80% HBM BW @ (M=2^28, mean_length=1024), bf16
Current: not measured yet (Pallas variant landed; bench pending on v5e)
Bottleneck: TBD pending bench. Hypothesis: HBM-bound — the kernel is VPU-only segmented Hillis-Steele (7 lane shifts + a small (T,) cross-row scan + one (T, 128) broadcast-add per tile), so VPU ops/elem sit well below the v5e compute-vs-BW crossover and DMA should be the floor.
Next: bench Pallas at the canonical (M=2^28, mean_length=1024) on v5e; if below 80%, profile to identify which stage stalls — likely candidates are the (T, 128) row-carry broadcast at large block_shape or DMA-vs-VPU overlap on the cross-row pass.
