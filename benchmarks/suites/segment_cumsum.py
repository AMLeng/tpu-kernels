"""Bench suite for the segment_cumsum op.

Run: `uv run python -m benchmarks.suites.segment_cumsum`
With HLO dump: `... --dump-hlo`
With xprof trace: `... --profile-dir /tmp/segment_cumsum_trace`

Memory-bound scan (1 add per element); speed-of-light is HBM bandwidth.
Min per-call HBM traffic: ``x`` read + ``segment_ids`` read + output
write = ``M * (bytes_per_elem + 4 + bytes_per_elem)``, reported as
``nbytes``. ``segment_ids`` is int32 regardless of ``--dtype``.

Pallas variant pending; for now `compare` runs xla only. No `--block` /
`--sweep-block` here — those flags only become meaningful once the
Pallas variant lands.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks.compare import compare
from benchmarks.suites._common import base_parser
from benchmarks.workload import Workload
from tpu_kernels.ops.segment_cumsum import segment_cumsum_xla


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[base_parser()])
    # M default clears the 4x v5e VMEM floor (CLAUDE.md / Bench inputs):
    # 2^28 bf16 elements = 512 MiB, matching scale's (16384, 16384).
    #
    # mean_length parametrizes the workload regime. Boundary-row density
    # is ~128/mean_length, so mean_length determines whether the kernel
    # exercises the within-row segment-reset path (the part that
    # distinguishes segment_cumsum from plain cumsum) or stays on the
    # plain-cumsum fast path. num_segments would also encode this but
    # only conditional on M, so two runs at the same num_segments and
    # different M sit in different regimes; mean_length is M-invariant.
    #
    # Default mean_length=1024 mirrors realistic packed-attention training
    # densities (typical packed seq lens 512-2048). ~12.5% of 128-lane
    # rows have a within-row boundary at this density — enough to make
    # the boundary-row path matter without dominating, which is the
    # regime the kernel needs to be good at.
    parser.add_argument("--m", type=int, default=2**28, help="number of elements")
    parser.add_argument(
        "--mean-length",
        type=int,
        default=1024,
        help="mean segment length (sets num_segments = m // mean_length)",
    )
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    if args.m % args.mean_length:
        parser.error(
            f"--m ({args.m}) must be divisible by --mean-length ({args.mean_length}) "
            f"for equal-width segments."
        )

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize

    x = jax.random.normal(jax.random.key(0), (args.m,), dtype=dtype)
    num_segments = args.m // args.mean_length
    segment_ids = jnp.repeat(
        jnp.arange(num_segments, dtype=jnp.int32),
        args.mean_length,
    )

    flops = args.m  # one add per element
    nbytes = (
        args.m * bytes_per_elem  # x read
        + args.m * 4  # segment_ids read (int32)
        + args.m * bytes_per_elem  # output write
    )
    workload = Workload(
        op="segment_cumsum",
        flops=flops,
        nbytes=nbytes,
        args=(x, segment_ids),
        flop_dtype=args.dtype,
    )

    compare(
        workload,
        variants={"xla": segment_cumsum_xla},
        dump_hlo=args.dump_hlo,
        profile_dir=args.profile_dir,
        timing=args.timing,
        config={
            "m": args.m,
            "mean_length": args.mean_length,
            "num_segments": num_segments,
            "dtype": args.dtype,
        },
    )


if __name__ == "__main__":
    main()
