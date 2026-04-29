"""Pallas scale: tiled elementwise kernel.

Trivial workload, but exercises the full Pallas surface used by every kernel:
`pallas_call`, a grid, `BlockSpec` with an index_map, and `out_shape`.

This module exposes a plain function — bench/test sites jit it themselves
(the bench captures `block_shape` in a closure, the correctness tests use
`interpret=True` and don't need jit).
"""

from __future__ import annotations

import jax
from jax.experimental import pallas as pl

DEFAULT_BLOCK = (256, 256)


def _scale_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2


def scale(
    x: jax.Array,
    block_shape: tuple[int, int] = DEFAULT_BLOCK,
    interpret: bool = False,
) -> jax.Array:
    if x.ndim != 2:
        raise ValueError(f"scale expects 2-D input, got {x.shape}")
    bm, bn = block_shape
    m, n = x.shape
    if m % bm or n % bn:
        raise ValueError(f"shape {x.shape} not divisible by block {block_shape}")

    return pl.pallas_call(
        _scale_kernel,
        grid=(m // bm, n // bn),
        in_specs=[pl.BlockSpec(block_shape, lambda i, j: (i, j))],
        out_specs=pl.BlockSpec(block_shape, lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x)
