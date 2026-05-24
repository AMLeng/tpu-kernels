# Sharded matmul (DP batch + TP contracting, all_reduce after)

Target: ≥80% MFU @ (B=8192, D=8192, F=8192), bf16, dp=2/tp=2 on v5e-4
Current: not measured yet (device timing now closes at v5e-4 — `runner._coalesce_planes` coalesces XPlane events across all 4 chip planes; first baseline JSON lands in the follow-up commit)
Bottleneck: the local matmul runs in f32 — `naive.py` casts both operands to f32 for the contracting-dim inner product (oracle precision), and `xla.py` reuses that body — but the roofline scores MFU against the bf16 peak, so the f32 MAC path caps MFU far below the bf16 plateau (early device read: ~30% MFU, with inputs pre-sharded onto the mesh so no per-call reshard inflates the time). The serial all_reduce over the contracting axis also sits on the critical path. Disentangling the two (a bf16 local matmul vs the comm tail) is the first real tuning step; the serial comm is what Stage B's `reduce_scatter ∘ all_gather` and Stage C's overlapped TP layer are meant to hide.
Next: bench the naive/xla baseline at v5e-4 and pin a real Current, then probe whether a bf16 local matmul (f32 accumulate only) lifts MFU before reaching for the collective-overlap rewrite.
