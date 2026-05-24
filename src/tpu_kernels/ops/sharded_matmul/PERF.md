# Sharded matmul (DP batch + TP contracting, all_reduce after)

Target: ≥80% MFU @ (B=8192, D=8192, F=8192), bf16, dp=2/tp=2 on v5e-4
Current: not measured yet (scaffold only — naive + xla; the bench writes the JSON once the harness extension below lands)
Bottleneck: harness — `runner.py` rejects multi-chip XPlane planes (`_events_from_line` raises when events show up on more than one TPU plane), so device-timing benches can't close. The kernel itself is the unfused "local matmul + all_reduce over the contracting axis" row-parallel linear: the matmul dominates the roofline (compute-bound at this shape) and the all_reduce sits on the critical path. That serial comm is the overhead the Stage B `reduce_scatter ∘ all_gather` decomposition (and Stage C's overlapped TP layer) is meant to hide.
Next: extend the runner to coalesce XPlane events across TPU planes so device-timing works at `num_chips > 1`; then bench the naive/xla baseline at v5e-4 and pin a real Current.
