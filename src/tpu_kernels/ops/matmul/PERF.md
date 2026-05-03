# Matmul (tiled bf16)

Target: ≥80% MFU @ (M=8192, N=8192, K=8192), bf16
Current: not measured yet (pre-bench scaffold).
Bottleneck: TBD — first measurement decides. xla likely close to the MXU plateau on a square shape; pallas at the default `(128, 128, 128)` cube is the minimum tile and almost certainly leaves perf on the table until the sweep finds a larger one.
Next: bench `naive`, `xla`, `pallas` on v5e; sweep `--block` over `(bm, bn, bk)` cubes around 256–512 to find the MXU sweet spot.
