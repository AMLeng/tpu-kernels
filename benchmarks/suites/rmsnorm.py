"""Bench suite for the rmsnorm op.

Run: `uv run python -m benchmarks.suites.rmsnorm`
With HLO dump: `... --dump-hlo`
With xprof trace: `... --profile-dir /tmp/rmsnorm_trace`

Memory-bound. The interesting knob is ``--dtype`` — the f32-accumulator
inside `naive` is unconditional, but the input/output dtype changes how
many bytes cross HBM, which moves BW% directly.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks.compare import compare
from benchmarks.roofline import v5e
from tpu_kernels.ops.rmsnorm import rmsnorm_xla


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=4096, help="leading (batch * seq) dim")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--dump-hlo", action="store_true")
    parser.add_argument("--profile-dir", default=None)
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize
    bs, h = args.bs, args.hidden
    x = jax.random.normal(jax.random.key(0), (bs, h), dtype=dtype)
    scale = jax.random.normal(jax.random.key(1), (h,), dtype=dtype)

    # Per element: x*x, sum (~1), x*inv_rms, y*scale ≈ 4 flops; rsqrt is per-row, amortized.
    flops = 4 * bs * h
    # Read x + read scale + write y. Scale is small but we count it for honesty.
    nbytes = 2 * bs * h * bytes_per_elem + h * bytes_per_elem

    compare(
        op="rmsnorm",
        variants={"xla": rmsnorm_xla},
        args=(x, scale),
        flops=flops,
        nbytes=nbytes,
        hw=v5e(),
        flop_dtype=args.dtype,
        dump_hlo=args.dump_hlo,
        profile_dir=args.profile_dir,
        config={"bs": bs, "hidden": h, "dtype": args.dtype},
    )


if __name__ == "__main__":
    main()
