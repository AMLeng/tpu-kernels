# TPU v5e — hardware reference

Numbers used by the roofline math live in `benchmarks/roofline.py`. Tune in
one place if Google revises figures.

## Per-chip peaks

| Quantity            | Value             | Notes                                          |
| ------------------- | ----------------- | ---------------------------------------------- |
| bf16 peak           | 197 TFLOP/s       | One MXU, 128×128 systolic                      |
| int8 peak           | 393 TOPS          | For int8 matmul                                |
| HBM capacity        | 16 GiB            |                                                |
| HBM bandwidth       | 819 GB/s          |                                                |
| VMEM capacity       | 128 MiB           | Per TensorCore; cross-checked vs pltpu.get_tpu_info |
| ICI per link (bidi) | 90 GB/s           | 45 GB/s/direction; 4 links → 360 GB/s/chip aggregate. One-way aggregate (200 GB/s) is the "1.6 Tbps" headline. 2D torus on multi-chip. |

Ridge point (peak_flops / peak_bw) ≈ **240 FLOPs/byte**. Below that you're
memory-bound, above you're compute-bound. RMSNorm/softmax sit far below;
matmul on reasonable shapes sits well above.

## Pod shapes

| Slice    | Chips | Topology       |
| -------- | ----- | -------------- |
| v5e-1    | 1     | 1×1            |
| v5e-4    | 4     | 2×2            |
| v5e-8    | 8     | 2×4            |
| v5e-16   | 16    | 4×4            |
| v5e-32   | 32    | 4×8            |
| v5e-64   | 64    | 8×8            |
| v5e-128  | 128   | 8×16           |
| v5e-256  | 256   | 16×16          |

Single-host slices: v5e-1 to v5e-8. Anything larger is multi-host and needs
`jax.distributed.initialize()` per worker.

## Pallas-relevant facts

- **Lane × sublane**: native VMEM tile is `(8, 128)`. Kernels that work in
  multiples of this layout avoid expensive transposes. Two-D arrays in
  Pallas are read with the trailing two dims as `(sublane, lane)`.
- **MXU shape**: 128×128 systolic. Matmuls work best when the contracting
  dim and output tile are multiples of 128.
- **Async DMA**: HBM↔VMEM moves are async; overlap with compute via
  `pltpu.make_async_copy` and double-buffering inside the grid.
- **Distributed kernels**: cross-device DMAs are issued via
  `pltpu.make_async_remote_copy` with semaphores in scratch
  (`pltpu.SemaphoreType.DMA`) and `pltpu.CompilerParams(collective_id=...)`.

## Sources

- Google Cloud — Cloud TPU v5e announcement & system architecture
- jax-ml/jax `docs/pallas/tpu/` — pipelining, distributed, sparse
- jax-ml/maxtext — production constants for v5e roofline
