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
    # Default M chosen to clear the 4x v5e VMEM floor (CLAUDE.md / Bench inputs):
    # 2^28 bf16 elements = 512 MiB, matching scale's (16384, 16384). The shape
    # is far above realistic per-call decode batches but is what the harness
    # needs to produce a meaningful BW%. num_segments=1024 keeps each segment
    # ~262K elements wide — enough boundaries that the segment-reset path runs
    # many times per call but small enough that within-segment scan dominates
    # the work.
    parser.add_argument("--m", type=int, default=2**28, help="number of elements")
    parser.add_argument(
        "--num-segments",
        type=int,
        default=1024,
        help="number of equal-width segments",
    )
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    if args.m % args.num_segments:
        parser.error(
            f"--m ({args.m}) must be divisible by --num-segments ({args.num_segments}) "
            f"for equal-width segments."
        )

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize

    x = jax.random.normal(jax.random.key(0), (args.m,), dtype=dtype)
    seg_size = args.m // args.num_segments
    segment_ids = jnp.repeat(
        jnp.arange(args.num_segments, dtype=jnp.int32),
        seg_size,
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
        config={"m": args.m, "num_segments": args.num_segments, "dtype": args.dtype},
    )


if __name__ == "__main__":
    main()
