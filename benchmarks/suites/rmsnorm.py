"""Bench suite for the rmsnorm op.

Run: `uv run python -m benchmarks.suites.rmsnorm`
With HLO dump: `... --dump-hlo`
With Mosaic dump (Pallas-side equivalent): `... --dump-mosaic`
With xprof trace: `... --profile-dir /tmp/rmsnorm_trace`
    Then: `uv run xprof /tmp/rmsnorm_trace` (full UI on :8791), or drag
    `<dir>/plugins/profile/*/*.trace.json.gz` into ui.perfetto.dev.

Pallas block-size sweep (1-D — hidden dim is always loaded in full):
    `... --sweep-block 8,16,32,64,128`

Memory-bound. The interesting knob is ``--dtype`` — the f32-accumulator
inside `naive` is unconditional, but the input/output dtype changes how
many bytes cross HBM, which moves BW% directly.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks.compare import compare, pallas_variant
from benchmarks.suites._common import base_parser, validate_block_shapes
from benchmarks.sweep import sweep
from benchmarks.workload import Workload
from tpu_kernels.ops.rmsnorm import rmsnorm_pallas, rmsnorm_xla
from tpu_kernels.ops.rmsnorm.pallas import DEFAULT_BLOCK


def _make_parser() -> argparse.ArgumentParser:
    """Add op-specific shape flags and pin ``--block`` to ``DEFAULT_BLOCK``."""
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.add_argument("--bs", type=int, default=32768, help="leading (batch * seq) dim")
    parser.add_argument("--hidden", type=int, default=8192)
    parser.set_defaults(block=list(DEFAULT_BLOCK))
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    validate_block_shapes(args, expected_axes=1, parser=parser)

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize
    bs, h = args.bs, args.hidden
    x = jax.random.normal(jax.random.key(0), (bs, h), dtype=dtype)
    scale = jax.random.normal(jax.random.key(1), (h,), dtype=dtype)

    # Per element: x*x, sum (~1), x*inv_rms, y*scale ≈ 4 flops; rsqrt is per-row, amortized.
    flops = 4 * bs * h
    # Read x + read scale + write y. Scale is small but we count it for honesty.
    nbytes = 2 * bs * h * bytes_per_elem + h * bytes_per_elem
    workload = Workload(
        op="rmsnorm", flops=flops, nbytes=nbytes, args=(x, scale), flop_dtype=args.dtype
    )

    if args.sweep_block is not None:
        (bms,) = args.sweep_block

        def is_valid(*, bm: int) -> bool:
            # Pre-emptive divisibility filter so the sweep doesn't crash inside
            # `rmsnorm_pallas` for shapes that obviously won't tile.
            return bs % bm == 0

        sweep(
            workload,
            pallas_fn=rmsnorm_pallas,
            axes={"bm": list(bms)},
            is_valid=is_valid,
            timing=args.timing,
        )
        return

    block_shape = tuple(args.block)

    compare(
        workload,
        variants={
            "xla": rmsnorm_xla,
            "pallas": pallas_variant(rmsnorm_pallas, block_shape=block_shape),
        },
        dump_hlo=args.dump_hlo,
        dump_mosaic=args.dump_mosaic,
        profile_dir=args.profile_dir,
        timing=args.timing,
        config={"bs": bs, "hidden": h, "dtype": args.dtype, "block_shape": list(block_shape)},
    )


if __name__ == "__main__":
    main()
