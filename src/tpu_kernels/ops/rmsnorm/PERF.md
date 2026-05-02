# RMSNorm

Target: ≥80% HBM BW @ (8192, 8192) bf16, scale (8192,) bf16
Current: 80.2% HBM BW Pallas (bm=128) vs 55.8% XLA, as of 2026-05-01 (v5e 1 chip; timing="device"). Under unroll: Pallas 79.2% / XLA 55.0% — device picks up ~+1pp on Pallas, mirroring scale.
Bottleneck: at the v5e single-chip plateau under device timing. Sweep over bm ∈ {32, 64, 128} clusters at 78.1-79.1% under unroll — once the tile is "big enough" the choice barely matters; bm=256 OOMs scoped VMEM (16.02M against 16M limit). Remaining ~20pp to 100% HBM is peak-vs-sustained + DMA setup, same as scale.
Next: rmsnorm is at target — done. Regression floor at 0.80 under timing="device" pinned in tests/perf/test_rmsnorm.py. Revisit if the kernel structure changes (e.g. fused with attention QK projection).
