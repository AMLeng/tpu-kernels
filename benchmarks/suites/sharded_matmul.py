"""Bench suite for the sharded_matmul op (pre-Stage-B exploratory).

Run: `uv run python -m benchmarks.suites.sharded_matmul`
With a custom mesh: `... --dp 4 --tp 2` (needs dp*tp actual devices)
With HLO dump: `... --dump-hlo`
With xprof trace: `... --profile-dir /tmp/sharded_matmul_trace`

Single-shape bench of the local-matmul + all_reduce row-parallel linear
``A[B_x, D_y] @ W[D_y, F] -> Out[B_x, F]``: batch sharded data-parallel
over ``x`` (size ``--dp``), contracting dim sharded tensor-parallel over
``y`` (size ``--tp``). Defaults to a 2x2 mesh.

Defaults to `--timing unroll` because the runner's device-timing path
doesn't yet coalesce events across TPU planes (runner.py
`_events_from_line` raises on >1 plane); unroll mode is wallclock and
works on multi-chip. The `--timing device` default flips back when the
harness extension lands (tracked in this op's PERF.md `Next`).

The suite hard-fails unless ``dp * tp`` equals the device count: the
mesh has nowhere to land otherwise, and silently running on a different
topology would either crash deep in shard_map or — worse — report
numbers against the wrong mesh.

Variants are `naive` and `xla` (today `jax.jit(naive)`). They share a
trace; the side-by-side comparison validates the harness end-to-end
and gives the xla slot a place to grow into when a JAX-side rewrite
(collective-aware reshard, manual reduce_scatter + all_gather, ...) is
worth comparing.
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
from tpu_kernels.ops.sharded_matmul.naive import DEFAULT_DP, DEFAULT_TP


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.add_argument("--batch", type=int, default=8192, help="B: batch dim (sharded over dp)")
    parser.add_argument("--d", type=int, default=8192, help="D: contracting dim (sharded over tp)")
    parser.add_argument("--f", type=int, default=8192, help="F: output feature dim")
    parser.add_argument("--dp", type=int, default=DEFAULT_DP, help="data-parallel mesh axis size")
    parser.add_argument("--tp", type=int, default=DEFAULT_TP, help="tensor-parallel mesh axis size")
    parser.set_defaults(timing="unroll")
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
    a = jax.random.normal(jax.random.key(0), (batch, d), dtype=dtype)  # A[B, D]
    w = jax.random.normal(jax.random.key(1), (d, f), dtype=dtype)  # W[D, F]

    # Cluster totals (the harness divides by `total_*` peaks, which scale by
    # num_chips). 2*B*D*F total compute is the same as the unsharded matmul —
    # the work is just split dp*tp ways.
    flops = 2 * batch * d * f
    # HBM, cluster total: A[B_x, D_y] is fully partitioned (read once = B*D);
    # W[D_y, F] is replicated over the dp axis (read dp times = dp*D*F); the
    # output O[B_x, F] is replicated over the tp axis (written tp times =
    # tp*B*F). The all_reduce traffic itself rides on ICI, not HBM.
    nbytes = (batch * d + dp * d * f + tp * batch * f) * bytes_per_elem
    # Ring all_reduce of the (B/dp, F) f32 partial over the tp-axis ring moves,
    # summed across all dp*tp chips, 2*(tp-1) * B * F f32 elements. Zero when
    # tp == 1 (no contraction to reduce). Partials are f32 inside
    # naive._local (the cast back happens after psum), so 4 bytes/element.
    ici_bytes = 2 * (tp - 1) * batch * f * 4
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
            "naive": partial(sharded_matmul_naive, dp=dp, tp=tp),
            "xla": partial(sharded_matmul_xla, dp=dp, tp=tp),
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
