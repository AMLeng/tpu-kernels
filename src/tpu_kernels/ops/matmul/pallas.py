"""Pallas matmul: 3-axis tiled kernel with VMEM-resident f32 accumulator.

Grid is ``(M/bm, N/bn, K/bk)``. The inner (K) axis is the accumulate
axis: the same output block ``(bm, bn)`` is visited for ``K/bk`` steps
and the partial products land in an f32 scratch ref kept in VMEM.
First K-step zero-inits the scratch; last K-step casts it down to the
output dtype and writes the block. This keeps the long K reduction in
f32 without paying for an f32 output buffer in HBM.

``preferred_element_type=jnp.float32`` on the per-tile ``jnp.dot`` is
what tells the MXU to multiply-accumulate in f32 regardless of input
dtype — the same convention naive/xla follow at the language level.

``block_shape`` is a 3-tuple ``(bm, bn, bk)`` — the repo-wide Pallas
convention is one tuple under the parameter name ``block_shape``, so
the bench harness can validate the spelling and bake the value in via
``pallas_variant``. ``DEFAULT_BLOCK = (1024, 1024, 512)`` is the
sweep-winning cube on v5e at the (8192, 8192, 8192) bf16 shape — the
larger ``(1024, 1024, 1024)`` cube OOMs VMEM (18.4 MiB scratch + tiles
exceeds the 16 MiB scoped limit), so 512 on the K axis is the sweet
spot. Re-tune via the suite's ``--sweep-block`` for other shapes.

This module exposes a plain function — bench/test sites jit it themselves
(the bench captures ``block_shape`` in a closure, the correctness tests
use ``interpret=True`` and don't need jit).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_BLOCK = (1024, 1024, 512)


def _matmul_kernel(a_ref, b_ref, o_ref, acc_ref) -> None:
    @pl.when(pl.program_id(2) == 0)
    def _zero_acc() -> None:
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _flush_to_output() -> None:
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def matmul(
    a: jax.Array,
    b: jax.Array,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
    interpret: bool = False,
) -> jax.Array:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"matmul expects 2-D inputs, got a={a.shape}, b={b.shape}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul shape mismatch: a={a.shape}, b={b.shape}")
    if len(block_shape) != 3:
        raise ValueError(f"matmul expects a 3-axis block_shape, got {block_shape}")
    bm, bn, bk = block_shape
    m, k = a.shape
    _, n = b.shape
    if m % bm or n % bn or k % bk:
        raise ValueError(
            f"shape (m={m}, n={n}, k={k}) not divisible by block (bm={bm}, bn={bn}, bk={bk})"
        )

    return pl.pallas_call(
        _matmul_kernel,
        grid=(m // bm, n // bn, k // bk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        interpret=interpret,
    )(a, b)
