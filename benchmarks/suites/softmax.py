"""Bench suite for the softmax op.

Run: `uv run python -m benchmarks.suites.softmax`
With HLO dump: `... --dump-hlo`
With xprof trace: `... --profile-dir /tmp/softmax_trace`
    Then: `uv run xprof /tmp/softmax_trace` (full UI on :8791), or drag
    `<dir>/plugins/profile/*/*.trace.json.gz` into ui.perfetto.dev.

Memory-bound. ``naive`` (jit-wrapped, two reductions over x) and
``xla`` (online softmax, one paired reduction) bench side-by-side —
the comparison shows whether the saved read of x materializes once
XLA lowers the custom ``lax.reduce``. Read ``--dump-hlo`` to inspect
the fusion. Pallas joins (with --block / --sweep-block) only if the
better of the two falls short of the 80% HBM BW target. See
``ops/softmax/PERF.md``.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks.compare import compare
from benchmarks.suites._common import base_parser
from benchmarks.workload import Workload
from tpu_kernels.ops.softmax import softmax_naive, softmax_xla


def _make_parser() -> argparse.ArgumentParser:
    """Add op-specific shape flags. ``--block`` / ``--sweep-block`` from
    ``base_parser`` are unused while xla is the only variant; they wire up
    to ``compare`` / ``sweep`` once a Pallas variant exists."""
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.add_argument("--bs", type=int, default=8192, help="leading (batch * seq) dim")
    parser.add_argument("--hidden", type=int, default=8192)
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize
    bs, h = args.bs, args.hidden
    x = jax.random.normal(jax.random.key(0), (bs, h), dtype=dtype)

    # Per element: sub from max, exp, div by sum ≈ 4 flops; max / sum
    # reductions are amortized per row.
    flops = 4 * bs * h
    # Read x + write y. No scale parameter.
    nbytes = 2 * bs * h * bytes_per_elem
    workload = Workload(op="softmax", flops=flops, nbytes=nbytes, args=(x,), flop_dtype=args.dtype)

    compare(
        workload,
        variants={"naive": softmax_naive, "xla": softmax_xla},
        dump_hlo=args.dump_hlo,
        profile_dir=args.profile_dir,
        timing=args.timing,
        config={"bs": bs, "hidden": h, "dtype": args.dtype},
    )


if __name__ == "__main__":
    main()
