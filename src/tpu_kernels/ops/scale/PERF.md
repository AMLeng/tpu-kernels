# Scale (×2)

Target: ≥80% HBM BW @ (M=8192, N=8192), bf16
Current: 81.6% HBM BW Pallas (block_shape=(512, 1024)), 2026-04-30, v5e 1 chip, timing="device".
Bottleneck: peak-vs-sustained + DMA setup overhead — the v5e single-chip plateau, not kernel structure.
Next: done; revisit only if the kernel structure changes (e.g. a streaming variant).
