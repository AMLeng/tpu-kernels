# Softmax

Target: ≥80% HBM BW @ (8192, 8192) bf16
Current: 80.2% HBM BW Pallas (bm=128) vs 55.6% xla, 42.8% naive, 2026-05-03, v5e 1 chip, timing="device".
Bottleneck: peak-vs-sustained + DMA setup overhead, same as rmsnorm — not kernel structure.
Next: done; revisit if fused with attention scores.
