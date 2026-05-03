"""Pallas softmax: row-tiled stable-form kernel.

Each block holds ``(bm, hidden)`` — a slab of full rows. Both reductions
(max and sum) and the divide all run while the row sits in VMEM, so the
kernel hits 1 read of x + 1 write of y in HBM. That's the saving the
xla docstring identifies but can't realize: XLA has to re-stream x from
HBM for the divide, costing a second read.

f32 accumulator regardless of input dtype, matching naive/xla. The
divide-and-cast lands in one expression so the f32 result is downcast
exactly once at the kernel boundary.

``block_shape`` is a 1-tuple ``(bm,)`` — the same convention as
rmsnorm, which has the same row-reduction shape. The hidden dim is
implicit (full row per block), so there's nothing to tile along axis 1.

This module exposes a plain function — bench/test sites jit it themselves
(the bench captures ``block_shape`` in a closure, the correctness tests
use ``interpret=True`` and don't need jit).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

DEFAULT_BLOCK = (128,)


def _softmax_kernel(x_ref, o_ref) -> None:
    x_32 = x_ref[...].astype(jnp.float32)
    m = jnp.max(x_32, axis=-1, keepdims=True)
    e = jnp.exp(x_32 - m)
    o_ref[...] = (e / jnp.sum(e, axis=-1, keepdims=True)).astype(o_ref.dtype)


def softmax(
    x: jax.Array,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
    interpret: bool = False,
) -> jax.Array:
    if x.ndim != 2:
        raise ValueError(f"softmax expects 2-D input, got {x.shape}")
    if len(block_shape) != 1:
        raise ValueError(f"softmax expects a 1-axis block_shape, got {block_shape}")
    (bm,) = block_shape
    m, n = x.shape
    tile_shape = (bm, n)
    if m % bm:
        raise ValueError(f"shape {x.shape} not divisible by block {tile_shape}")

    return pl.pallas_call(
        _softmax_kernel,
        grid=(m // bm,),
        in_specs=[pl.BlockSpec(tile_shape, lambda i: (i, 0))],
        out_specs=pl.BlockSpec(tile_shape, lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x)
