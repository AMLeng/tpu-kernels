# Softmax

Target: ≥80% HBM BW @ (8192, 8192) bf16
Current: 55.6% HBM BW xla (online) vs 42.8% naive, 2026-05-03, v5e 1 chip, timing="device".
Bottleneck: xla still reads x twice (paired reduction + divide), capping it at the same regime rmsnorm-xla landed in (~55%). Closing to 80% needs Pallas tiling to share x in VMEM across the two phases.
Next: add `pallas.py` to fold reduction + divide into one tile-resident pass over x.
