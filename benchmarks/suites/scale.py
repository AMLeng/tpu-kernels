"""Bench suite for the scale op.

Single shape:
    `uv run python -m benchmarks.suites.scale`
    `... --block 512 512`

Cartesian block sweep (Pallas tuning):
    `... --sweep-block 8,16,32,64,128,256 128,256,512`

With HLO dump: `... --dump-hlo`
With xprof trace: `... --profile-dir /tmp/scale_trace`
    Then: `uv run xprof /tmp/scale_trace` (full UI on :8791), or drag
    `<dir>/plugins/profile/*/*.trace.json.gz` into ui.perfetto.dev.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks.compare import compare, pallas_variant
from benchmarks.suites._common import base_parser, validate_block_shapes
from benchmarks.sweep import sweep
from benchmarks.workload import Workload
from tpu_kernels.ops.scale import scale_pallas, scale_xla
from tpu_kernels.ops.scale.pallas import DEFAULT_BLOCK


def _make_parser() -> argparse.ArgumentParser:
    """Add op-specific shape flags and pin ``--block`` to ``DEFAULT_BLOCK``."""
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.add_argument("--m", type=int, default=16384)
    parser.add_argument("--n", type=int, default=16384)
    parser.set_defaults(block=list(DEFAULT_BLOCK))
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    validate_block_shapes(args, expected_axes=2, parser=parser)

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize
    x = jax.random.normal(jax.random.key(0), (args.m, args.n), dtype=dtype)

    flops = args.m * args.n  # one mul per element
    nbytes = 2 * args.m * args.n * bytes_per_elem  # read + write
    workload = Workload(op="scale", flops=flops, nbytes=nbytes, args=(x,), flop_dtype=args.dtype)

    if args.sweep_block is not None:
        bms, bns = args.sweep_block

        def is_valid(*, bm: int, bn: int) -> bool:
            # Pre-emptive divisibility filter so the sweep doesn't crash inside
            # `scale_pallas` for shapes that obviously won't tile.
            return args.m % bm == 0 and args.n % bn == 0

        sweep(
            workload,
            pallas_fn=scale_pallas,
            axes={"bm": list(bms), "bn": list(bns)},
            is_valid=is_valid,
            timing=args.timing,
        )
        return

    block_shape: tuple[int, int] = (args.block[0], args.block[1])

    compare(
        workload,
        variants={
            "xla": scale_xla,
            "pallas": pallas_variant(scale_pallas, block_shape=block_shape),
        },
        dump_hlo=args.dump_hlo,
        profile_dir=args.profile_dir,
        timing=args.timing,
        config={"block_shape": list(block_shape)},
    )


if __name__ == "__main__":
    main()
