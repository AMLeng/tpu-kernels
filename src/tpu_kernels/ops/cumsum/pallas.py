"""Pallas cumsum: tiled scan with recursive-MXU within-tile prefix.

Per-tile structure: ``x`` is laid out as ``(T, 128)`` with
``T = bm/128``. Stage 1 uses a ``triu`` matmul to produce
within-128-lane prefix sums in one MXU call.

For ``T >= 128`` (i.e. ``bm >= 16384``), the prefix across the row
axis is itself computed by a second MXU call: row-totals are reshaped
``(T,) -> (B, 128, 128)`` (with B = T/128, broadcast across an
artificial lane axis) and a 3D ``dot_general`` contracting on the
within-block-row axis produces the within-block prefix in one MXU
op — folding ``log2(128) = 7`` HS levels into hardware that runs in
parallel with the VPU. The result has the within-block index ``j`` as
the lane axis; one sublane↔lane transpose moves it to the sublane
position so it can be added directly to the (T, 128) output without
the lane-axis singleton broadcast Mosaic refuses. A small HS scan
across the ``B`` block-totals (32 elements at the canonical
``bm=524288``, so 5 trivial levels) finishes the cross-block prefix.

For ``T < 128`` (small ``bm``) the recursion has no purchase and a
single (T, 128)-broadcast HS handles the row-totals.

Cross-tile carry through a ``(1,)`` VMEM scratch.

This module exposes a plain function — bench/test sites jit it
themselves (the bench captures ``block_shape`` in a closure, the
correctness tests use ``interpret=True`` and don't need jit).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_BLOCK = (524288,)


def _hs_exclusive_prefix(totals: jax.Array) -> jax.Array:
    """Hillis-Steele exclusive prefix on (N, 128) lane-broadcast totals."""
    running = totals
    offset = 1
    while offset < totals.shape[0]:
        shifted = jnp.concatenate(
            [jnp.zeros((offset, 128), running.dtype), running[:-offset]], axis=0
        )
        running = running + shifted
        offset *= 2
    return jnp.concatenate([jnp.zeros((1, 128), running.dtype), running[:-1]], axis=0)


def _cumsum_kernel(x_ref, u_ref, o_ref, scratch_ref):
    @pl.when(pl.program_id(0) == 0)
    def _init_scratch():
        scratch_ref[...] = jnp.zeros_like(scratch_ref)

    t = x_ref.shape[0]  # bm / 128

    summed_rows = jnp.dot(x_ref[...], u_ref[...], preferred_element_type=jnp.float32).astype(
        x_ref.dtype
    )

    if t < 128:
        row_totals = jnp.broadcast_to(summed_rows[:, -1:], summed_rows.shape)
        tile_cumsum = summed_rows + _hs_exclusive_prefix(row_totals)
    else:
        b = t // 128

        # Lane-broadcast row totals so the (B, 128, 128) tensor has
        # row_total[b*128+j] at every lane l — the dot below contracts
        # on the within-block-row axis assuming this invariant.
        row_totals_full = jnp.broadcast_to(summed_rows[:, -1:], summed_rows.shape)
        row_totals_3d = row_totals_full.reshape(b, 128, 128)
        inner_inclusive = jax.lax.dot_general(
            row_totals_3d,
            u_ref[...],
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(x_ref.dtype)  # (B, l, j)

        # Sublane↔lane transpose moves j from lane to sublane to align
        # with summed_rows' (b, j_sublane, l_lane) layout. v5e has HW
        # support for this on (8, 128) tiles.
        inner_inclusive_t = inner_inclusive.transpose((0, 2, 1))  # (B, j, l)

        inner_exclusive_3d = jnp.concatenate(
            [
                jnp.zeros((b, 1, 128), inner_inclusive_t.dtype),
                inner_inclusive_t[:, :-1, :],
            ],
            axis=1,
        )

        block_totals = inner_inclusive_t[:, -1, :]  # (B, 128), lane-broadcast
        block_exclusive = _hs_exclusive_prefix(block_totals)

        # broadcast_in_dim mapping (0, 1) -> (0, 2) inserts the size-128
        # broadcast at sublane position; lane stays trailing — avoids
        # the trailing-singleton reshape Mosaic refuses.
        summed_rows_3d = summed_rows.reshape(b, 128, 128)
        block_exclusive_3d = jax.lax.broadcast_in_dim(
            block_exclusive, (b, 128, 128), broadcast_dimensions=(0, 2)
        )
        output_3d = summed_rows_3d + inner_exclusive_3d + block_exclusive_3d
        tile_cumsum = output_3d.reshape(t, 128)

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
