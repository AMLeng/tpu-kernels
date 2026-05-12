"""Bench suite for the segment_cumsum op.

Single shape:
    `uv run python -m benchmarks.suites.segment_cumsum`
    `... --block 8192`

Cartesian block sweep (Pallas tuning):
    `... --sweep-block 1024,2048,4096,8192,16384`

With HLO dump: `... --dump-hlo`
With Mosaic dump (Pallas-side equivalent): `... --dump-mosaic`
With xprof trace: `... --profile-dir /tmp/segment_cumsum_trace`

Memory-bound scan (1 add per element); speed-of-light is HBM bandwidth.
Min per-call HBM traffic: ``x`` read + ``segment_ids`` read + output
write = ``M * (bytes_per_elem + 4 + bytes_per_elem)``, reported as
``nbytes``. ``segment_ids`` is int32 regardless of ``--dtype``.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.compare import compare, pallas_variant
from benchmarks.suites._common import base_parser, validate_block_shapes
from benchmarks.sweep import sweep
from benchmarks.workload import Workload
from tpu_kernels.ops.segment_cumsum import segment_cumsum_pallas, segment_cumsum_xla
from tpu_kernels.ops.segment_cumsum.pallas import DEFAULT_BLOCK


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[base_parser()])
    # M default clears the 4x v5e VMEM floor (CLAUDE.md / Bench inputs):
    # 2^28 bf16 elements = 512 MiB, matching scale's (16384, 16384).
    #
    # mean_length parametrizes the workload regime via boundary-row
    # density (~128/mean_length): too sparse and the kernel runs on the
    # plain-cumsum fast path with negligible boundary correction, too
    # dense and the within-row segment-reset path dominates a regime no
    # real workload sees. Default 1024 mirrors packed-attention training
    # densities (typical packed seq lens 512-2048). Widths themselves
    # are *sampled* from a geometric distribution with this mean (see
    # ``_build_inputs``) — fixed-stride boundaries let kernels exploit
    # prefetch/branch predictability that real packed sequences don't
    # offer, and a stride that happens to be a multiple of 128 would
    # land every boundary at a row start and skip the within-row path
    # entirely.
    parser.add_argument("--m", type=int, default=2**28, help="number of elements")
    parser.add_argument(
        "--mean-length",
        type=int,
        default=1024,
        help="mean segment length (widths are geometrically distributed)",
    )
    parser.set_defaults(block=list(DEFAULT_BLOCK))
    return parser


def _build_inputs(args: argparse.Namespace) -> tuple[jax.Array, jax.Array, int]:
    """Construct ``(x, segment_ids, num_segments)`` from parsed args.

    Pulled out of ``main`` so the suite-level regression tests can
    assert on the data the bench actually feeds the kernel (per
    CLAUDE.md's TDD-for-harness rule).
    """
    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    x = jax.random.uniform(jax.random.key(0), (args.m,), dtype=dtype)

    # Variable-width segments sampled from a geometric distribution with
    # mean `mean_length`. Fixed-width segments let an algorithm benefit
    # from boundary-stride predictability that real packed-attention
    # workloads don't offer; sampling produces irregular boundaries that
    # mirror real packed sequences. Oversample 2x to be statistically
    # sure cumulative widths cover M, then close the final segment at M
    # so widths sum to exactly M.
    rng = np.random.default_rng(seed=0)
    n_oversample = max(int(2 * args.m / args.mean_length) + 100, 200)
    widths = rng.geometric(p=1.0 / args.mean_length, size=n_oversample)
    ends = np.cumsum(widths)
    ends_inside = ends[ends < args.m]
    widths_final = np.diff(np.concatenate([[0], ends_inside, [args.m]])).astype(np.int32)
    num_segments = len(widths_final)
    segment_ids = jnp.asarray(np.repeat(np.arange(num_segments, dtype=np.int32), widths_final))
    return x, segment_ids, int(num_segments)


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    validate_block_shapes(args, expected_axes=1, parser=parser)

    if args.mean_length > args.m:
        parser.error(f"--mean-length ({args.mean_length}) must be <= --m ({args.m}).")

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize

    x, segment_ids, num_segments = _build_inputs(args)

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

    if args.sweep_block is not None:
        (bms,) = args.sweep_block

        def is_valid(*, bm: int) -> bool:
            # Pre-emptive divisibility filter so the sweep doesn't crash
            # inside segment_cumsum_pallas for shapes that obviously won't
            # tile.
            return args.m % bm == 0

        sweep(
            workload,
            pallas_fn=segment_cumsum_pallas,
            axes={"bm": list(bms)},
            is_valid=is_valid,
            timing=args.timing,
        )
        return

    block_shape: tuple[int] = (args.block[0],)

    compare(
        workload,
        variants={
            "xla": segment_cumsum_xla,
            "pallas": pallas_variant(segment_cumsum_pallas, block_shape=block_shape),
        },
        dump_hlo=args.dump_hlo,
        dump_mosaic=args.dump_mosaic,
        profile_dir=args.profile_dir,
        timing=args.timing,
        config={
            "m": args.m,
            "mean_length": args.mean_length,
            "num_segments": num_segments,
            "dtype": args.dtype,
            "block_shape": list(block_shape),
        },
    )


if __name__ == "__main__":
    main()
