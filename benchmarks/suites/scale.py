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

from benchmarks.compare import compare
from benchmarks.roofline import v5e
from benchmarks.sweep import sweep
from tpu_kernels.ops.scale import scale_pallas, scale_xla


def _csv_ints(s: str) -> tuple[int, ...]:
    """Parse ``"8,16,32"`` → ``(8, 16, 32)``. argparse hook for sweep axes."""
    return tuple(int(x) for x in s.split(","))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument(
        "--block",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Single block shape (ignored when --sweep-block is set).",
    )
    parser.add_argument(
        "--sweep-block",
        type=_csv_ints,
        nargs=2,
        default=None,
        metavar=("BM_LIST", "BN_LIST"),
        help="Cartesian sweep over (bm, bn). Example: --sweep-block 8,16,32 128,256,512",
    )
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--dump-hlo", action="store_true")
    parser.add_argument("--profile-dir", default=None)
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize
    x = jax.random.normal(jax.random.key(0), (args.m, args.n), dtype=dtype)

    flops = args.m * args.n  # one mul per element
    nbytes = 2 * args.m * args.n * bytes_per_elem  # read + write

    if args.sweep_block is not None:
        bms, bns = args.sweep_block

        def variant_factory(*, bm: int, bn: int) -> jax.stages.Wrapped:
            block_shape = (bm, bn)

            @jax.jit
            def fn(y: jax.Array) -> jax.Array:
                return scale_pallas(y, block_shape=block_shape)

            return fn

        def is_valid(*, bm: int, bn: int) -> bool:
            # Pre-emptive divisibility filter so the sweep doesn't crash inside
            # `scale_pallas` for shapes that obviously won't tile.
            return args.m % bm == 0 and args.n % bn == 0

        sweep(
            op="scale",
            variant_factory=variant_factory,
            axes={"bm": list(bms), "bn": list(bns)},
            args=(x,),
            flops=flops,
            nbytes=nbytes,
            hw=v5e(),
            flop_dtype=args.dtype,
            is_valid=is_valid,
        )
        return

    block_shape: tuple[int, int] = (args.block[0], args.block[1])

    @jax.jit
    def pallas_fn(y: jax.Array) -> jax.Array:
        return scale_pallas(y, block_shape=block_shape)

    compare(
        op="scale",
        variants={"xla": scale_xla, "pallas": pallas_fn},
        args=(x,),
        flops=flops,
        nbytes=nbytes,
        hw=v5e(),
        flop_dtype=args.dtype,
        dump_hlo=args.dump_hlo,
        profile_dir=args.profile_dir,
        config={"block_shape": list(block_shape)},
    )


if __name__ == "__main__":
    main()
