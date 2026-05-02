"""Bench suite for the rmsnorm op.

Run: `uv run python -m benchmarks.suites.rmsnorm`
With HLO dump: `... --dump-hlo`
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

from benchmarks.compare import compare
from benchmarks.roofline import v5e
from benchmarks.sweep import sweep
from tpu_kernels.ops.rmsnorm import rmsnorm_pallas, rmsnorm_xla
from tpu_kernels.ops.rmsnorm.pallas import DEFAULT_BLOCK


def _csv_ints(s: str) -> tuple[int, ...]:
    """Parse ``"8,16,32"`` → ``(8, 16, 32)``. argparse hook for sweep axis."""
    return tuple(int(x) for x in s.split(","))


def _make_parser() -> argparse.ArgumentParser:
    """Build the suite's CLI parser. Factored so tests can pin defaults
    against bench() / kernel without launching the suite."""
    parser = argparse.ArgumentParser()
    # Default shape gives 128 MiB of bf16 input — 4x v5e VMEM, the floor
    # CLAUDE.md sets so a chained-call XLA pipeline can't keep the working
    # set on chip and inflate per-call BW% under unroll mode.
    parser.add_argument("--bs", type=int, default=8192, help="leading (batch * seq) dim")
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument(
        "--block",
        type=int,
        default=DEFAULT_BLOCK,
        help="Pallas row-tile (bm). Must divide --bs. Ignored when --sweep-block is set.",
    )
    parser.add_argument(
        "--sweep-block",
        type=_csv_ints,
        default=None,
        metavar="BM_LIST",
        help="Sweep over block_size values. Example: --sweep-block 8,16,32,64,128",
    )
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--dump-hlo", action="store_true")
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument(
        "--timing",
        choices=["unroll", "device"],
        default="unroll",
        help="Timing mode forwarded to bench(); device-mode is TPU-only.",
    )
    return parser


def main() -> None:
    args = _make_parser().parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize
    bs, h = args.bs, args.hidden
    x = jax.random.normal(jax.random.key(0), (bs, h), dtype=dtype)
    scale = jax.random.normal(jax.random.key(1), (h,), dtype=dtype)

    # Per element: x*x, sum (~1), x*inv_rms, y*scale ≈ 4 flops; rsqrt is per-row, amortized.
    flops = 4 * bs * h
    # Read x + read scale + write y. Scale is small but we count it for honesty.
    nbytes = 2 * bs * h * bytes_per_elem + h * bytes_per_elem

    if args.sweep_block is not None:
        bms = args.sweep_block

        def variant_factory(*, bm: int) -> jax.stages.Wrapped:
            @jax.jit
            def fn(y: jax.Array, s: jax.Array) -> jax.Array:
                return rmsnorm_pallas(y, s, block_size=bm)

            return fn

        def is_valid(*, bm: int) -> bool:
            # Pre-emptive divisibility filter so the sweep doesn't crash inside
            # `rmsnorm_pallas` for shapes that obviously won't tile.
            return bs % bm == 0

        sweep(
            op="rmsnorm",
            variant_factory=variant_factory,
            axes={"bm": list(bms)},
            args=(x, scale),
            flops=flops,
            nbytes=nbytes,
            hw=v5e(),
            flop_dtype=args.dtype,
            is_valid=is_valid,
            timing=args.timing,
        )
        return

    block_size = args.block

    @jax.jit
    def pallas_fn(y: jax.Array, s: jax.Array) -> jax.Array:
        return rmsnorm_pallas(y, s, block_size=block_size)

    compare(
        op="rmsnorm",
        variants={"xla": rmsnorm_xla, "pallas": pallas_fn},
        args=(x, scale),
        flops=flops,
        nbytes=nbytes,
        hw=v5e(),
        flop_dtype=args.dtype,
        dump_hlo=args.dump_hlo,
        profile_dir=args.profile_dir,
        timing=args.timing,
        config={"bs": bs, "hidden": h, "dtype": args.dtype, "block_size": block_size},
    )


if __name__ == "__main__":
    main()
