"""Pallas cumsum: tiled scan with a per-grid scalar carry.

Per-tile structure: ``x`` is laid out as ``(T, 128)`` with
``T = bm/128``. Stage 1 uses a ``triu`` matmul to produce
within-128-lane prefix sums in one MXU call. Stage 2 is a
Hillis-Steele scan over the row-totals, broadcast to ``(T, 128)``
across the lane axis; the redundancy is intentional — VPU ops are
128-lane-wide regardless, so filling all lanes uses cycles that would
otherwise be idle. Cross-tile carry through a ``(1,)`` VMEM scratch.

This module exposes a plain function — bench/test sites jit it
themselves (the bench captures ``block_shape`` in a closure, the
correctness tests use ``interpret=True`` and don't need jit).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_BLOCK = (131072,)


def _cumsum_kernel(x_ref, u_ref, o_ref, scratch_ref):
    @pl.when(pl.program_id(0) == 0)
    def _init_scratch():
        scratch_ref[...] = jnp.zeros_like(scratch_ref)

    summed_rows = jnp.dot(x_ref[...], u_ref[...], preferred_element_type=jnp.float32).astype(
        x_ref.dtype
    )
    y = jnp.broadcast_to(summed_rows[:, -1:], summed_rows.shape)
    offset = 1
    while offset < x_ref.shape[0]:
        # Hillis-Steele adder on the row sums.
        shifted = jnp.concatenate([jnp.zeros((offset, 128), y.dtype), y[:-offset]])
        y = y + shifted
        offset *= 2
    tile_cumsum = summed_rows + jnp.concatenate([jnp.zeros((1, 128), y.dtype), y[:-1]])

    o_ref[...] = scratch_ref[...] + tile_cumsum
    scratch_ref[...] = scratch_ref[...] + tile_cumsum[-1:, -1]


def cumsum(
    x: jax.Array,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
    interpret: bool = False,
) -> jax.Array:
    if x.ndim != 1:
        raise ValueError(f"cumsum expects 1-D input, got {x.shape}")
    if len(block_shape) != 1:
        raise ValueError(f"cumsum expects a 1-axis block_shape, got {block_shape}")
    (m,) = x.shape
    (bm,) = block_shape
    if bm & (bm - 1):
        raise ValueError(f"block size {bm} not a power of 2")
    if m % bm:
        raise ValueError(f"shape {x.shape} not divisible by block {block_shape}")

    tile_shape = (bm // 128, 128)
    x = x.reshape(m // 128, 128)
    u = jnp.triu(jnp.ones((128, 128), dtype=x.dtype))
    grid_spec = pl.GridSpec(
        grid=(m // bm,),
        in_specs=[
            pl.BlockSpec(tile_shape, lambda i: (i, 0)),
            pl.BlockSpec(u.shape, lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec(tile_shape, lambda i: (i, 0)),
        scratch_shapes=[pltpu.VMEM((1,), x.dtype)],
    )

    return pl.pallas_call(
        _cumsum_kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x, u).ravel()
