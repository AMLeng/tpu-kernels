"""Bench suite for the cumsum op.

Single shape:
    `uv run python -m benchmarks.suites.cumsum`
    `... --block 8192`

Cartesian block sweep (Pallas tuning):
    `... --sweep-block 1024,2048,4096,8192,16384`

With HLO dump: `... --dump-hlo`
With Mosaic dump (Pallas-side equivalent): `... --dump-mosaic`
With xprof trace: `... --profile-dir /tmp/cumsum_trace`

Memory-bound scan (1 add per element); speed-of-light is HBM bandwidth.
Min per-call HBM traffic: ``x`` read + output write =
``2 * M * bytes_per_elem``, reported as ``nbytes``. No
``segment_ids`` traffic — that's the whole reason this op exists as a
side-by-side baseline for ``segment_cumsum``: at bf16 the per-element
HBM is ~4 B here vs ~8 B there, so the BW% rows are directly
comparable while the bytes/element ratio attributes any delta to the
segment-reset path.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks.compare import compare, pallas_variant
from benchmarks.suites._common import base_parser, validate_block_shapes
from benchmarks.sweep import sweep
from benchmarks.workload import Workload
from tpu_kernels.ops.cumsum import cumsum_pallas, cumsum_xla
from tpu_kernels.ops.cumsum.pallas import DEFAULT_BLOCK


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[base_parser()])
    # Default M chosen to clear the 4x v5e VMEM floor (CLAUDE.md / Bench
    # inputs): 2^28 bf16 elements = 512 MiB. Matches segment_cumsum's
    # default so the two suites' xla rows are apples-to-apples.
    parser.add_argument("--m", type=int, default=2**28, help="number of elements")
    parser.set_defaults(block=list(DEFAULT_BLOCK))
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    validate_block_shapes(args, expected_axes=1, parser=parser)

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize

    x = jax.random.normal(jax.random.key(0), (args.m,), dtype=dtype)

    flops = args.m  # one add per element
    nbytes = 2 * args.m * bytes_per_elem  # x read + output write
    workload = Workload(
        op="cumsum",
        flops=flops,
        nbytes=nbytes,
        args=(x,),
        flop_dtype=args.dtype,
    )

    if args.sweep_block is not None:
        (bms,) = args.sweep_block

        def is_valid(*, bm: int) -> bool:
            # Pre-emptive divisibility filter so the sweep doesn't crash
            # inside cumsum_pallas for shapes that obviously won't tile.
            return args.m % bm == 0

        sweep(
            workload,
            pallas_fn=cumsum_pallas,
            axes={"bm": list(bms)},
            is_valid=is_valid,
            timing=args.timing,
        )
        return

    block_shape: tuple[int] = (args.block[0],)

    compare(
        workload,
        variants={
            "xla": cumsum_xla,
            "pallas": pallas_variant(cumsum_pallas, block_shape=block_shape),
        },
        dump_hlo=args.dump_hlo,
        dump_mosaic=args.dump_mosaic,
        profile_dir=args.profile_dir,
        timing=args.timing,
        config={"m": args.m, "dtype": args.dtype, "block_shape": list(block_shape)},
    )


if __name__ == "__main__":
    main()
