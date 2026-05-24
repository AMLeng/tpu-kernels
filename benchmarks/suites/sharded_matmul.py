"""Bench suite for the sharded_matmul op (pre-Stage-B exploratory).

Run: `uv run python -m benchmarks.suites.sharded_matmul`
With a custom mesh: `... --dp 4 --tp 2` (needs dp*tp actual devices)
With HLO dump: `... --dump-hlo`
With xprof trace: `... --profile-dir /tmp/sharded_matmul_trace`

Single-shape bench of the reduce-scatter row-parallel linear
``A[B_x, D_y] @ W[D_y, F] -> Out[B_x, F_y]``: batch sharded data-parallel
over ``x`` (size ``--dp``), contracting dim sharded tensor-parallel over
``y`` (size ``--tp``), output feature dim reduce-scattered over ``y``.
Defaults to a 2x2 mesh.

Uses the base `--timing device` default: the runner now coalesces XPlane
events across all TPU planes (runner.py `_coalesce_planes`), so device
timing works multi-chip. It reads the per-call cost straight off the TPU
clock at k=1, which sidesteps the unroll-mode trap this op exposed —
the single-call cost (~10ms here) sits right on `_choose_k`'s 10ms target,
so unroll flips between k=1 and k=2 on sizing noise, and because chaining a
collective lets XLA overlap each call's reduce-scatter with the next call's
matmul, the per-call number swings ~1.5x with that coin-flip. `--timing
unroll` is still available for CPU.

The suite hard-fails unless ``dp * tp`` equals the device count: the
mesh has nowhere to land otherwise, and silently running on a different
topology would either crash deep in shard_map or — worse — report
numbers against the wrong mesh.

Variants are `naive` (a single `psum_scatter`) and `xla` (a hand-rolled
`ppermute` ring reduce-scatter). Both produce the same reduce-scattered
output; the side-by-side comparison checks the ring against the
single-collective oracle and validates the harness end-to-end.
"""

from __future__ import annotations

import argparse
from functools import partial

import jax
import jax.numpy as jnp

from benchmarks.compare import compare
from benchmarks.roofline import v5e
from benchmarks.suites._common import base_parser
from benchmarks.workload import Workload
from tpu_kernels.ops.sharded_matmul import sharded_matmul_naive, sharded_matmul_xla
from tpu_kernels.ops.sharded_matmul.sharding import (
    DEFAULT_DP,
    DEFAULT_TP,
    make_mesh,
    shard_inputs,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.add_argument("--batch", type=int, default=8192, help="B: batch dim (sharded over dp)")
    parser.add_argument("--d", type=int, default=8192, help="D: contracting dim (sharded over tp)")
    parser.add_argument("--f", type=int, default=8192, help="F: output feature dim")
    parser.add_argument("--dp", type=int, default=DEFAULT_DP, help="data-parallel mesh axis size")
    parser.add_argument("--tp", type=int, default=DEFAULT_TP, help="tensor-parallel mesh axis size")
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    n_devices = len(jax.devices())
    assert args.dp * args.tp == n_devices, (
        f"sharded_matmul needs dp*tp == device count: got dp={args.dp}, tp={args.tp} "
        f"(product {args.dp * args.tp}) but {n_devices} devices. "
        "On CPU, set the count with `XLA_FLAGS=--xla_force_host_platform_device_count=N`."
    )

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize
    batch, d, f = args.batch, args.d, args.f
    dp, tp = args.dp, args.tp
    # Build the mesh once and place the inputs on it before timing: in
    # production the activation/weight are already sharded across the mesh, so
    # the timed call should not pay a per-call scatter to lay them out for
    # shard_map. shard_inputs commits A to P(x, y) and W to P(y, None) — the
    # exact in_specs the kernel expects — so no reshard rides the hot path.
    mesh = make_mesh(dp, tp)
    a, w = shard_inputs(
        mesh,
        jax.random.normal(jax.random.key(0), (batch, d), dtype=dtype),  # A[B, D]
        jax.random.normal(jax.random.key(1), (d, f), dtype=dtype),  # W[D, F]
    )

    # Cluster totals (the harness divides by `total_*` peaks, which scale by
    # num_chips). 2*B*D*F total compute is the same as the unsharded matmul —
    # the work is just split dp*tp ways.
    flops = 2 * batch * d * f
    # HBM, cluster total: A[B_x, D_y] is fully partitioned (read once = B*D);
    # W[D_y, F] is replicated over the dp axis (read dp times = dp*D*F); the
    # reduce-scattered output O[B_x, F_y] is fully partitioned over tp (written
    # once = B*F, each chip its F-shard). The collective traffic rides on ICI,
    # not HBM.
    nbytes = (batch * d + dp * d * f + batch * f) * bytes_per_elem
    # Reduce-scatter of the (B/dp, F) f32 partial over the tp-axis ring moves,
    # summed across all dp*tp chips, (tp-1) * B * F f32 elements — half the
    # all_reduce volume, since there's no all_gather leg. Zero when tp == 1
    # (nothing to reduce). Partials are f32 (naive psum_scatters f32; the xla
    # ring accumulates in f32), so 4 bytes/element.
    ici_bytes = (tp - 1) * batch * f * 4
    workload = Workload(
        op="sharded_matmul",
        flops=flops,
        nbytes=nbytes,
        args=(a, w),
        flop_dtype=args.dtype,
    )

    compare(
        workload,
        variants={
            "naive": partial(sharded_matmul_naive, mesh=mesh),
            "xla": partial(sharded_matmul_xla, mesh=mesh),
        },
        hw=v5e(num_chips=n_devices),
        ici_bytes=ici_bytes,
        dump_hlo=args.dump_hlo,
        dump_mosaic=args.dump_mosaic,
        profile_dir=args.profile_dir,
        timing=args.timing,
        config={"batch": batch, "d": d, "f": f, "dp": dp, "tp": tp, "dtype": args.dtype},
    )


if __name__ == "__main__":
    main()
