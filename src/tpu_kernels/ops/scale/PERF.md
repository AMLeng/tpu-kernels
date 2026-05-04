# Scale (×2)

Target: ≥80% HBM BW @ (M=16384, N=16384), bf16
Current: 82.1% HBM BW Pallas / 80.3% xla (block_shape=(512, 1024)), 2026-05-04, v5e 1 chip, timing="device".
Bottleneck: peak-vs-sustained + DMA setup overhead — the v5e single-chip plateau, not kernel structure.
Next: done; revisit only if the kernel structure changes (e.g. a streaming variant).
