"""Bench suite for the scale op.

Run: `uv run python -m benchmarks.suites.scale`
With HLO dump: `... --dump-hlo`
With xprof trace: `... --profile-dir /tmp/scale_trace`
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks.compare import compare
from benchmarks.roofline import v5e
from tpu_kernels.ops.scale import scale_pallas, scale_xla


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--block", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--dump-hlo", action="store_true")
    parser.add_argument("--profile-dir", default=None)
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize
    x = jax.random.normal(jax.random.key(0), (args.m, args.n), dtype=dtype)

    flops = args.m * args.n  # one mul per element
    nbytes = 2 * args.m * args.n * bytes_per_elem  # read + write

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
