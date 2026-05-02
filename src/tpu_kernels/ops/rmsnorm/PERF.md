# RMSNorm

Target: ≥85% HBM BW @ (8192, 8192) bf16, scale (8192,) bf16
Current: not measured yet (no TPU run)
Bottleneck: n/a — memory-bound; lever is whether XLA fuses square → mean → rsqrt → mul → scale into one elementwise loop
Next: bench on v5e, dump HLO to confirm a single fused loop; sweep dtype ∈ {bf16, f32} to see the BW% delta from the f32 accumulator
