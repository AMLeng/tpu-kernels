"""Bench suite for embedding_lookup_packed on the packed 4-D bf16 layout.

Run: `uv run python -m benchmarks.suites.embedding_lookup_packed`
With HLO dump: `... --dump-hlo`
With Mosaic dump (Pallas-side equivalent): `... --dump-mosaic`
With xprof trace: `... --profile-dir /tmp/embedding_lookup_packed_trace`

Same workload as ``embedding_lookup`` (memory-bound row gather; nbytes
is the M-row read + M-row write theoretical minimum). The difference
is the layout: this suite pre-formats ``params`` into the 4-D shape
``(vocab, hidden//1024, 8, 128)`` whose natural Mosaic
``T(8, 128)(2, 1)`` tile lives on the inner two dims — byte-equivalent
to 1-D ``T(1024)(128)(2, 1)``, where each row's bytes are contiguous
in memory.

Variants benched on this layout:

  - xla: `jit(params[ids])` — XLA emits per-row reads + bulk write
    back to HBM; ~65% BW.
  - pallas: explicit read-many / write-one (per-row reads HBM→VMEM,
    one bulk VMEM→HBM per grid step); ~69% BW.

`naive` is the eager `params[ids]` form used as the correctness
oracle in `tests/correctness/test_embedding_lookup_packed.py`; it
lowers to the same code as `xla` once the harness jit's it, so it'd
just produce a duplicate row. Same convention as the scale and
rmsnorm suites.

Pallas edges out XLA on this layout, but the layout is doing most of
the work — the inverse of the natural-2-D case where
`gather_custom_fusion` is special-cased and Pallas can't match it.
Pre-formatting (one-time ~1 GiB byte reorder) happens at setup,
outside the timing loop.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks.compare import compare, pallas_variant
from benchmarks.suites._common import base_parser, validate_block_shapes
from benchmarks.workload import Workload
from tpu_kernels.ops.embedding_lookup_packed import (
    embedding_lookup_packed_pallas,
    embedding_lookup_packed_xla,
)
from tpu_kernels.ops.embedding_lookup_packed.pallas import DEFAULT_BLOCK

# Inner (8, 128) tile that the packed layout pivots on; the dim-1 chunk
# size is 8 * 128 = 1024 elements per (8, 128)-group of one row.
ROW_CHUNK = 1024


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[base_parser()])
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
    if args.hidden % ROW_CHUNK:
        parser.error(
            f"--hidden must be divisible by {ROW_CHUNK} for the packed layout; got {args.hidden}"
        )

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize

    # Build params in natural 2-D, then pre-format into the packed shape.
    # The reshape is *not* a free metadata change — it's a real ~1 GiB
    # byte reorder. Done once outside the bench timing.
    params_2d = jax.random.normal(jax.random.key(0), (args.vocab, args.hidden), dtype=dtype)
    params_packed = jax.jit(lambda p: p.reshape(args.vocab, args.hidden // ROW_CHUNK, 8, 128))(
        params_2d
    )
    params_packed.block_until_ready()
    ids = jax.random.randint(jax.random.key(1), (args.m,), 0, args.vocab, dtype=jnp.int32)

    flops = 0
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
            op="embedding_lookup_packed",
            flops=flops,
            nbytes=nbytes,
            args=(params_packed, ids),
            flop_dtype=args.dtype,
        ),
        variants={
            "xla": embedding_lookup_packed_xla,
            "pallas": pallas_variant(embedding_lookup_packed_pallas, block_shape=block_shape),
        },
        dump_hlo=args.dump_hlo,
        dump_mosaic=args.dump_mosaic,
        profile_dir=args.profile_dir,
        timing=args.timing,
        config=config,
    )


if __name__ == "__main__":
    main()
