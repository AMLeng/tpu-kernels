# Segment cumsum

Target: ≥80% HBM BW @ (M=2^28, num_segments=1024), bf16
Current: not measured yet (xla baseline pending bench; Pallas not written)
Bottleneck: TBD — measure xla baseline first; expectation per the matmul / embedding-lookup precedent is that XLA's lowering of `cumsum` + `cummax` + gather saturates HBM (each is a primitive scan with no fusion left to discover) and the Pallas variant will match rather than beat it.
Next: bench xla, then write Pallas variant exercising the parallel-prefix scan with a carry across tiles plus the segment-boundary reset — the pattern that returns at every Stage D ragged kernel.
