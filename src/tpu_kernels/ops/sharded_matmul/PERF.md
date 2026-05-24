# Sharded matmul (DP batch + TP contracting, reduce-scatter)

Target: ≥80% of speed-of-light @ (B=8192, D=8192, F=8192) bf16, dp2/tp2 on v5e-4.
Current: 47.6% SoL xla / 40.7% naive (compute-bound, so SoL = MFU), (B=8192, D=8192, F=8192) bf16, dp2/tp2 v5e-4, timing="device", 2026-05-24 — below the 80% target: the collective isn't hidden at this shape.
Bottleneck: unhidden reduce-scatter comm, not compute — compute scales ∝ B·D·F while the collective is ∝ B·F, so large-D shapes amortize it and approach speed-of-light in exploratory runs.
Next: improve pipeline in the ring to better hide communication.
