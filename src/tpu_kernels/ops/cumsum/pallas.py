"""Pallas cumsum: tiled scan with recursive-MXU within-tile prefix.

Per-tile structure: ``x`` is laid out as ``(T, 128)`` with
``T = bm/128``. Stage 1 uses a ``triu`` matmul to produce
within-128-lane prefix sums in one MXU call.

For ``T >= 128`` (i.e. ``bm >= 16384``), the prefix across the row
axis is itself computed by a second MXU call: row-totals are reshaped
``(T,) -> (B, 128, 128)`` (with B = T/128, broadcast across an
artificial lane axis). A second constant — a ``tril`` matrix
``v[k, j] = 1 iff j < k`` — is used as the LHS of a batched
``dot_general`` (batch dim ``B``), contracting ``v``'s lane axis
with ``row_totals_3d``'s within-block-row axis. This directly computes
the level 2 row-total prefixes in ``(B, k_sublane, l_lane)`` layout, aligning
with the (B, j_sublane, l_lane) layout of the per-row prefixes with no
post-MXU sublane↔lane transpose. The MXU folds ``log2(128) = 7`` HS
levels into hardware that runs in parallel with the VPU. A small HS
scan across the ``B`` block-totals (32 elements at the canonical
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
    # N=1 is a real path (bm=16384 → b=1). The final concat below would
    # otherwise produce a 0-sized vector that the TPU lowering rejects.
    if totals.shape[0] == 1:
        return jnp.zeros_like(totals)
    running = totals
    offset = 1
    while offset < totals.shape[0]:
        shifted = jnp.concatenate(
            [jnp.zeros((offset, 128), running.dtype), running[:-offset]], axis=0
        )
        running = running + shifted
        offset *= 2
    return jnp.concatenate([jnp.zeros((1, 128), running.dtype), running[:-1]], axis=0)


def _cumsum_kernel(x_ref, u_ref, v_ref, o_ref, scratch_ref):
    @pl.when(pl.program_id(0) == 0)
    def _init_scratch():
        scratch_ref[...] = jnp.zeros_like(scratch_ref)

    t = x_ref.shape[0]  # bm / 128
    x = x_ref[...]

    summed_rows = jnp.dot(x, u_ref[...], preferred_element_type=jnp.float32).astype(x_ref.dtype)
    # Row totals come from a fresh lane-axis reduction over ``x``, not
    # from slicing ``summed_rows[:, -1:]``. The slice would create a
    # serial dep on the matmul output (the broadcast can't start until
    # the MXU retires that lane); the VPU sum runs concurrently with
    # the MXU on the same VMEM-resident input. Worth ~7 BW pp at the
    # canonical (M=2^28, bm=524288) shape.
    row_total_col = jnp.sum(x, axis=1, keepdims=True)

    if t < 128:
        row_totals = jnp.broadcast_to(row_total_col, summed_rows.shape)
        tile_cumsum = summed_rows + _hs_exclusive_prefix(row_totals)
    else:
        b = t // 128

        # Lane-broadcast row totals so the (B, 128, 128) tensor has
        # row_total[b*128+j] at every lane l — the dot below contracts
        # on the within-block-row axis assuming this invariant.
        row_totals_full = jnp.broadcast_to(row_total_col, summed_rows.shape)
        row_totals_3d = row_totals_full.reshape(b, 128, 128)

        # ``v[k, j] = 1 iff j < k`` (lower-triangular) on LHS, broadcast
        # to a B batch dim. Contract v's lane axis (j) with row_totals_3d's
        # within-block-row axis (axis 1 = j_sub). JAX's dot_general output
        # ordering — (batch, lhs_non_contract, rhs_non_contract) —
        # places v's sublane (k_v) at output sublane and row_totals_3d's
        # lane (l) at output lane, so the exclusive prefix lands as
        # (B, k_v_sublane, l_lane) ready to add to summed_rows_3d
        # without a sublane↔lane transpose.
        v_3d = jnp.broadcast_to(v_ref[...], (b, 128, 128))
        inner_exclusive_3d = jax.lax.dot_general(
            v_3d,
            row_totals_3d,
            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        ).astype(x_ref.dtype)  # (B, k_v, l)

        # Directly recompute block_totals; it is cheaper to use the matmul for
        # exclusive prefixes + recompute than to compute the inclusive prefixes
        # and then index/concat to get the exclusive ones.
        block_totals = jnp.sum(row_totals_3d, axis=1)
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
    # Strict lower triangular to produce the exclusive prefix
    v = jnp.tril(jnp.ones((128, 128), dtype=x.dtype), k=-1)
    grid_spec = pl.GridSpec(
        grid=(m // bm,),
        in_specs=[
            pl.BlockSpec(tile_shape, lambda i: (i, 0)),
            pl.BlockSpec(u.shape, lambda i: (0, 0)),
            pl.BlockSpec(v.shape, lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec(tile_shape, lambda i: (i, 0)),
        scratch_shapes=[pltpu.VMEM((1,), x.dtype)],
    )

    return pl.pallas_call(
        _cumsum_kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x, u, v).ravel()
