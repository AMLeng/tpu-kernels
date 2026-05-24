# Sharded matmul (DP batch + TP contracting, reduce-scatter)

Target: ≥80% of speed-of-light @ (B=8192, D=8192, F=8192) bf16, dp2/tp2 on v5e-4.
Current: not measured yet — no reduce-scatter baseline benched (4aba0a2's ~30% MFU was the earlier all-reduce variant).
Bottleneck: unhidden reduce-scatter comm, not compute — compute scales ∝ B·D·F while the collective is ∝ B·F, so large-D shapes amortize it and approach speed-of-light in exploratory runs.
Next: improve pipeline in the ring to better hide communication.
