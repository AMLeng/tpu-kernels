# Segment cumsum

Target: ≥80% HBM BW @ (M=2^28, mean_length=1024), bf16
Current: 26.3% HBM BW @ bm=262144 (M=2^28, mean_length=1024)
Bottleneck: VPU saturated by the within-row seg-HS. 7 sequential levels (log₂ 128) × ≥3 dependent ops/level (cmpi + select + addf on (T, 128) bf16), already minimal post MXU-issued lane shift. Measured 9.76 µs/tile vs 3.20 µs/tile budget for 80% BW — VPU work, not DMA, is the binding cost (default pallas_call already pipelines DMA; emit_pipeline is a no-op here).
Next: parked. Target left at 80% as the regime's roofline but unreachable for VPU-only segmented prefix on v5e. The MXU-prefix variant we tried (per-row cumsum from a triu matmul + forward-fill boundary correction) collapsed to dense seg-HS cost because TPU SIMD has no sparse-on-dense propagation — every "cost proportional to boundary count" idea reduces to full-row work. Closing the gap would need either MXU as the primary engine (no viable path found; per-row segment masks turn the triu trick into poorly-utilized matmul-vector ops) or different silicon (e.g. SparseCore).
