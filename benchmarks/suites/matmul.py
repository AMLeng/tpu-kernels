"""Bench suite for the matmul op.

Run: `uv run python -m benchmarks.suites.matmul`
With HLO dump: `... --dump-hlo`
With xprof trace: `... --profile-dir /tmp/matmul_trace`
    Then: `uv run xprof /tmp/matmul_trace` (full UI on :8791), or drag
    `<dir>/plugins/profile/*/*.trace.json.gz` into ui.perfetto.dev.

Pallas block-size sweep (3-D — bm, bn, bk):
    `... --sweep-block 128,256 128,256 128,256,512`

Compute-bound at the default square shape (intensity ≫ v5e ridge), so
the relevant axis is MFU%. ``naive`` (jit-wrapped f32 reference),
``xla`` (jnp.matmul, default precision), and ``pallas`` (3-axis tiled
with VMEM-resident f32 accumulator) bench side-by-side. xla likely
sits near the MXU plateau on this shape; pallas exists to teach the
tiled-MXU pattern flash attention reuses.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks.compare import compare, pallas_variant
from benchmarks.suites._common import base_parser, validate_block_shapes
from benchmarks.sweep import sweep
from benchmarks.workload import Workload
from tpu_kernels.ops.matmul import matmul_naive, matmul_pallas, matmul_xla
from tpu_kernels.ops.matmul.pallas import DEFAULT_BLOCK


def _make_parser() -> argparse.ArgumentParser:
    """Add op-specific shape flags and pin ``--block`` to ``DEFAULT_BLOCK``."""
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.add_argument("--m", type=int, default=16384, help="output rows / a leading dim")
    parser.add_argument("--n", type=int, default=16384, help="output cols / b trailing dim")
    parser.add_argument("--k", type=int, default=16384, help="contracting dim")
    parser.set_defaults(block=list(DEFAULT_BLOCK))
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    validate_block_shapes(args, expected_axes=3, parser=parser)

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize
    m, n, k = args.m, args.n, args.k
    a = jax.random.normal(jax.random.key(0), (m, k), dtype=dtype)
    b = jax.random.normal(jax.random.key(1), (k, n), dtype=dtype)

    # 2*M*N*K — one multiply + one add per inner-product term.
    flops = 2 * m * n * k
    # Read a + read b + write o. Minimum HBM traffic; counts the f32
    # accumulator as VMEM-resident (matches the Pallas kernel's contract).
    nbytes = (m * k + k * n + m * n) * bytes_per_elem
    workload = Workload(op="matmul", flops=flops, nbytes=nbytes, args=(a, b), flop_dtype=args.dtype)

    if args.sweep_block is not None:
        bms, bns, bks = args.sweep_block

        def is_valid(*, bm: int, bn: int, bk: int) -> bool:
            # Pre-emptive divisibility filter so the sweep doesn't crash
            # inside ``matmul_pallas`` for shapes that obviously won't tile.
            return m % bm == 0 and n % bn == 0 and k % bk == 0

        sweep(
            workload,
            pallas_fn=matmul_pallas,
            axes={"bm": list(bms), "bn": list(bns), "bk": list(bks)},
            is_valid=is_valid,
            timing=args.timing,
        )
        return

    block_shape = tuple(args.block)

    compare(
        workload,
        variants={
            "naive": matmul_naive,
            "xla": matmul_xla,
            "pallas": pallas_variant(matmul_pallas, block_shape=block_shape),
        },
        dump_hlo=args.dump_hlo,
        profile_dir=args.profile_dir,
        timing=args.timing,
        config={"m": m, "n": n, "k": k, "dtype": args.dtype, "block_shape": list(block_shape)},
    )


if __name__ == "__main__":
    main()
