"""Bench suite for the attention op (Stage D.1: no-cache, single-seq, fwd).

Run: `uv run python -m benchmarks.suites.attention`
With HLO dump: `... --dump-hlo`
With xprof trace: `... --profile-dir /tmp/attn_trace`
    Then: `uv run xprof /tmp/attn_trace` (full UI on :8791), or drag
    `<dir>/plugins/profile/*/*.trace.json.gz` into ui.perfetto.dev.

Compute-bound at the default training shape — intensity ≫ v5e ridge for
T=4096, so the relevant axis is MFU%. Two variants bench side-by-side:

- `naive`: stable scaled-dot-product on (T, N, H), f32-accumulated.
- `xla`: `jax.jit(naive)`. Pedagogically the *wall*: XLA can't fuse the
  softmax into the score and value matmuls, so the (N, T, T) scores
  tensor is materialized to HBM between them. Use `--dump-hlo` to read
  this directly.

Pallas (flash) is not in this scaffold yet — it lands in a follow-up
with the `--block` / `--sweep-block` plumbing every Pallas-bearing op
adopts. Until then `naive` vs `xla` is the comparison; the suite is
already wired so iterating on `xla.py` is `edit → re-run → read deltas
in bench_history/attention/`.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks.compare import compare
from benchmarks.suites._common import base_parser
from benchmarks.workload import Workload
from tpu_kernels.ops.attention import attention_naive, attention_xla


def _make_parser() -> argparse.ArgumentParser:
    """Inherit common flags (`--dtype`, `--timing`, `--dump-hlo`, ...). The
    `--block` / `--sweep-block` flags ride along from `base_parser` but are
    unused here — they wake up when the Pallas variant lands and the suite
    starts validating block shapes.
    """
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.add_argument("--t", type=int, default=4096, help="sequence length")
    parser.add_argument("--n", type=int, default=16, help="number of attention heads")
    parser.add_argument("--h", type=int, default=128, help="head dim")
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    bytes_per_elem = jnp.dtype(dtype).itemsize
    t, n, h = args.t, args.n, args.h
    keys = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(keys[0], (t, n, h), dtype=dtype)
    k = jax.random.normal(keys[1], (t, n, h), dtype=dtype)
    v = jax.random.normal(keys[2], (t, n, h), dtype=dtype)

    # Two N*T*T*H matmuls dominate; softmax (~4*N*T*T) is 1/H smaller and
    # folded into the rounding — matches the matmul-counting convention
    # `benchmarks/suites/matmul.py` uses.
    flops = 4 * n * t * t * h
    # Minimum HBM traffic: read Q + read K + read V + write O. The (N, T, T)
    # scores intermediate is what xla actually pushes through HBM and is the
    # source of the wall — BW% will reflect that overage against this floor.
    nbytes = 4 * t * n * h * bytes_per_elem
    workload = Workload(
        op="attention", flops=flops, nbytes=nbytes, args=(q, k, v), flop_dtype=args.dtype
    )

    compare(
        workload,
        variants={
            "naive": attention_naive,
            "xla": attention_xla,
        },
        dump_hlo=args.dump_hlo,
        dump_mosaic=args.dump_mosaic,
        profile_dir=args.profile_dir,
        timing=args.timing,
        config={"t": t, "n": n, "h": h, "dtype": args.dtype},
    )


if __name__ == "__main__":
    main()
