# Cumsum

Target: ≥80% HBM BW @ (M=2^28), bf16
Current: 80.0% HBM BW Pallas (bm=1048576) vs 19.0% xla, 2026-05-08, v5e 1 chip, timing="device".
Bottleneck: peak-vs-sustained + DMA setup overhead — v5e plateau, not kernel structure.
Next: done; revisit if a fused use site (e.g. `segment_cumsum`) needs a different block-shape regime.
