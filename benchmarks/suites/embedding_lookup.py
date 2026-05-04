"""Bench suite for the embedding_lookup op.

Run: `uv run python -m benchmarks.suites.embedding_lookup`
With HLO dump: `... --dump-hlo`
With xprof trace: `... --profile-dir /tmp/embedding_lookup_trace`
    Then: `uv run xprof /tmp/embedding_lookup_trace` (full UI on :8791), or
    drag `<dir>/plugins/profile/*/*.trace.json.gz` into ui.perfetto.dev.

Memory-bound gather (0 flops/element); speed-of-light is HBM bandwidth.
The minimum per-call HBM traffic is M selected rows of `params` read
plus M rows of output written — the `2 * M * hidden * bytes_per_elem`
figure the bench reports as `nbytes`. The full `params` table is much
larger (`vocab * hidden`) but only M of its rows are touched per call,
so the gather pattern's quality shows up as BW% relative to that
minimum.

Pallas variant pending; for now `compare` runs naive + xla. They
should land at near-identical BW% within noise (xla is `jit(naive)`).
No `--block` / `--sweep-block` here — those flags only become
meaningful once the Pallas variant lands.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks.compare import compare
from benchmarks.suites._common import base_parser
from benchmarks.workload import Workload
from tpu_kernels.ops.embedding_lookup import (
    embedding_lookup_naive,
    embedding_lookup_xla,
)


def _make_parser() -> argparse.ArgumentParser:
    """Add op-specific shape flags. ``--block``/``--sweep-block`` from the
    base parser are unused at this stage — they activate when the Pallas
    variant lands."""
    parser = argparse.ArgumentParser(parents=[base_parser()])
    # Defaults match Llama 3.1 8B: vocab=128256 (Llama 3 tokenizer), hidden=4096.
    # m is the per-call index count (= batch * seq for an LM forward pass);
    # 8192 is a typical 1-batch 8K-context inference shape.
    parser.add_argument("--vocab", type=int, default=128256, help="parameter table rows")
    parser.add_argument(
        "--hidden", type=int, default=4096, help="parameter table cols / output dim"
    )
    parser.add_argument("--m", type=int, default=8192, help="number of indices to look up")
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize

    params = jax.random.normal(jax.random.key(0), (args.vocab, args.hidden), dtype=dtype)
    ids = jax.random.randint(jax.random.key(1), (args.m,), 0, args.vocab, dtype=jnp.int32)

    flops = 0  # pure gather; MFU is meaningless here, BW% is the metric.
    # Min HBM traffic: M rows of params read + M rows of output written.
    # ids is `M * 4` bytes (negligible at production sizes).
    nbytes = 2 * args.m * args.hidden * bytes_per_elem
    workload = Workload(
        op="embedding_lookup",
        flops=flops,
        nbytes=nbytes,
        args=(params, ids),
        flop_dtype=args.dtype,
    )

    compare(
        workload,
        variants={
            "naive": embedding_lookup_naive,
            "xla": embedding_lookup_xla,
        },
        dump_hlo=args.dump_hlo,
        profile_dir=args.profile_dir,
        timing=args.timing,
        config={"vocab": args.vocab, "hidden": args.hidden, "m": args.m, "dtype": args.dtype},
    )


if __name__ == "__main__":
    main()
