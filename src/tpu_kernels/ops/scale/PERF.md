# Scale (×2)

Target: ≥95% HBM BW @ (M=8192, N=8192), bf16  (regression floor: 85% — see tests/perf/test_scale.py)
Current: not measured yet (no TPU run)
Bottleneck: n/a — pure memory-bound, only knob is block_shape
Next: bench on v5e, sweep block_shape ∈ {(128,128), (256,256), (512,512), (1024,1024)}
