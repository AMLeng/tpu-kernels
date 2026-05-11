"""Bench suite for embedding_lookup on the natural 2-D bf16 layout.

Run: `uv run python -m benchmarks.suites.embedding_lookup`
With HLO dump: `... --dump-hlo`
With Mosaic dump (Pallas-side equivalent): `... --dump-mosaic`
With xprof trace: `... --profile-dir /tmp/embedding_lookup_trace`

Memory-bound gather (0 flops/element); speed-of-light is HBM bandwidth.
The minimum per-call HBM traffic is M selected rows of `params` read
plus M rows of output written — `2 * M * hidden * bytes_per_elem`,
reported as `nbytes`. The full `params` is much larger
(`vocab * hidden`) but only M rows are touched per call.

This suite benches two variants on the *natural* `T(8, 128)(2, 1)`
layout:

  - xla: `jit(params[ids])` — XLA's `gather_custom_fusion` (~32% BW).
  - pallas: 8-row slab DMA + VMEM mask-extract (~12% BW cap; structural).

`naive` is the eager `params[ids]` form used as the correctness
oracle in `tests/correctness/test_embedding_lookup.py`; it lowers to
the same code as `xla` once the harness jit's it, so including it
here would just produce a duplicate row. Same convention as the
scale and rmsnorm suites.

Pallas loses to XLA on this layout — see the docstring in
``src/tpu_kernels/ops/embedding_lookup/__init__.py`` for why.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks.compare import compare, pallas_variant
from benchmarks.suites._common import base_parser, validate_block_shapes
from benchmarks.workload import Workload
from tpu_kernels.ops.embedding_lookup import (
    embedding_lookup_pallas,
    embedding_lookup_xla,
)
from tpu_kernels.ops.embedding_lookup.pallas import DEFAULT_BLOCK


def _make_parser() -> argparse.ArgumentParser:
    """Add op-specific shape flags and pin ``--block`` to ``DEFAULT_BLOCK``."""
    parser = argparse.ArgumentParser(parents=[base_parser()])
    # Defaults match Llama 3.1 8B: vocab=128256 (Llama 3 tokenizer), hidden=4096.
    # m is the per-call index count (= batch * seq for an LM forward pass);
    # 8192 is a typical 1-batch 8K-context inference shape.
    parser.add_argument("--vocab", type=int, default=128256, help="parameter table rows")
    parser.add_argument(
        "--hidden", type=int, default=4096, help="parameter table cols / output dim"
    )
    parser.add_argument("--m", type=int, default=8192, help="number of indices to look up")
    parser.set_defaults(block=list(DEFAULT_BLOCK))
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    validate_block_shapes(args, expected_axes=1, parser=parser)

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize

    params = jax.random.normal(jax.random.key(0), (args.vocab, args.hidden), dtype=dtype)
    ids = jax.random.randint(jax.random.key(1), (args.m,), 0, args.vocab, dtype=jnp.int32)

    flops = 0  # pure gather; MFU is meaningless, BW% is the metric.
    nbytes = 2 * args.m * args.hidden * bytes_per_elem
    block_shape: tuple[int, ...] = (args.block[0],)
    config = {
        "vocab": args.vocab,
        "hidden": args.hidden,
        "m": args.m,
        "dtype": args.dtype,
        "block_shape": list(block_shape),
    }

    compare(
        Workload(
            op="embedding_lookup",
            flops=flops,
            nbytes=nbytes,
            args=(params, ids),
            flop_dtype=args.dtype,
        ),
        variants={
            "xla": embedding_lookup_xla,
            "pallas": pallas_variant(embedding_lookup_pallas, block_shape=block_shape),
        },
        dump_hlo=args.dump_hlo,
        dump_mosaic=args.dump_mosaic,
        profile_dir=args.profile_dir,
        timing=args.timing,
        config=config,
    )


if __name__ == "__main__":
    main()
