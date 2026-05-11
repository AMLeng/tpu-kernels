"""Pallas segment_cumsum: VPU-only segmented Hillis-Steele.

Tile body on ``x`` laid out as ``(T, 128)``:

1. Within-row segmented HS along lanes — 7 levels of lane shifts; at
   each level add the shifted value only when its sid matches the
   current sid. Monotonic segment ids make sid equality at the shift
   offset equivalent to "same segment throughout the shift gap".
2. Across-row inclusive segmented HS on the per-row tail values, on
   1-D ``(T,)`` data — broadcast across lanes and added only where the
   entering segment id matches the current sid.

Cross-tile state in two ``(1,)`` VMEM scratches: last running sum and
last sid. First tile seeds last sid with ``int32.min`` to force a
no-match at the (0, 0) carry slot.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_BLOCK = (524288,)


def _seg_hs_lane(vals: jax.Array, sids: jax.Array) -> jax.Array:
    """Within-row segmented HS inclusive prefix along the lane axis. (N, 128)."""
    n = vals.shape[0]
    offset = 1
    while offset < 128:
        # concat-shift on vals (vs pltpu.roll) because roll lowers to
        # tpu.dynamic_rotate, which Mosaic only implements for 32-bit element
        # types — bf16 vals error out at tile heights ≥ 256. sids stays on
        # roll: it's int32 (no Mosaic limit) and wraparound matches are
        # harmless since shifted_vals is 0 in the leading ``offset`` lanes.
        shifted_vals = jnp.concatenate(
            [jnp.zeros((n, offset), vals.dtype), vals[:, :-offset]], axis=1
        )
        shifted_sids = pltpu.roll(sids, offset, axis=1)
        vals = vals + jnp.where(shifted_sids == sids, shifted_vals, 0)
        offset *= 2
    return vals


def _seg_hs_row(totals: jax.Array, sids: jax.Array) -> jax.Array:
    """Across-row segmented HS inclusive prefix on 1-D ``(T,)`` data."""
    n = totals.shape[0]
    if n == 1:
        return totals
    offset = 1
    while offset < n:
        shifted_totals = jnp.concatenate(
            [jnp.zeros((offset,), totals.dtype), totals[:-offset]], axis=0
        )
        shifted_sids = jnp.concatenate([jnp.zeros((offset,), sids.dtype), sids[:-offset]], axis=0)
        totals = totals + jnp.where(shifted_sids == sids, shifted_totals, 0)
        offset *= 2
    return totals


def _segment_cumsum_kernel(
    x_ref,
    sid_ref,
    o_ref,
    sum_scratch_ref,
    sid_scratch_ref,
):
    @pl.when(pl.program_id(0) == 0)
    def _init_scratch():
        sum_scratch_ref[...] = jnp.zeros_like(sum_scratch_ref)
        sid_scratch_ref[...] = jnp.full_like(sid_scratch_ref, jnp.iinfo(jnp.int32).min)

    t = x_ref.shape[0]
    x = x_ref[...]
    sid = sid_ref[...]

    within_row = _seg_hs_lane(x, sid)

    # prev_row_tail[i] is the within-row segment tail at row i-1 (the
    # cross-tile carry at i=0). entering_sid[i] is the sid of that tail
    # position — the segment id entering row i. For t == 1 the slice
    # ``within_row[:-1, -1]`` would be (0,) — an empty intermediate that
    # the TPU lowering rejects — so short-circuit to the carry alone.
    if t == 1:
        prev_row_tail = sum_scratch_ref[...]
        entering_sid = sid_scratch_ref[...]
    else:
        prev_row_tail = jnp.concatenate([sum_scratch_ref[...], within_row[:-1, -1]], axis=0)
        entering_sid = jnp.concatenate([sid_scratch_ref[...], sid[:-1, -1]], axis=0)
    # Inclusive seg-HS gives, at index i, the sum of tails from preceding
    # rows whose tail sid matches entering_sid[i] — the prefix to add at
    # lanes of row i still in the entering segment.
    row_carry = _seg_hs_row(prev_row_tail, entering_sid)

    entering_sid_2d = jax.lax.broadcast_in_dim(entering_sid, sid.shape, (0,))
    # Run the whole row-carry select in f32 and only cast back at the
    # final add. Two distinct Mosaic limits hit the bf16 path at tile
    # heights ≥ 256: the (t,)→(t,1) reshape inside broadcast_in_dim, and
    # the bitwidth-16 i1 mask in `where` (lane-replicated row_carry vs
    # lane-concrete sid forces an unsupported relayout). Both go away at
    # bitwidth 32, so f32 throughout sidesteps them.
    row_carry_2d = jax.lax.broadcast_in_dim(row_carry.astype(jnp.float32), sid.shape, (0,))
    delta = jnp.where(entering_sid_2d == sid, row_carry_2d, jnp.float32(0.0))
    tile_out = within_row + delta.astype(within_row.dtype)

    o_ref[...] = tile_out
    sum_scratch_ref[...] = tile_out[-1:, -1]
    sid_scratch_ref[...] = sid[-1:, -1]


def segment_cumsum(
    x: jax.Array,
    segment_ids: jax.Array,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
    interpret: bool = False,
) -> jax.Array:
    if x.ndim != 1:
        raise ValueError(f"segment_cumsum expects 1-D x, got {x.shape}")
    if segment_ids.shape != x.shape:
        raise ValueError(f"segment_ids shape {segment_ids.shape} must match x shape {x.shape}")
    if segment_ids.dtype != jnp.int32:
        raise ValueError(f"segment_ids must be int32, got {segment_ids.dtype}")
    if len(block_shape) != 1:
        raise ValueError(f"segment_cumsum expects a 1-axis block_shape, got {block_shape}")
    (m,) = x.shape
    (bm,) = block_shape
    if bm & (bm - 1):
        raise ValueError(f"block size {bm} not a power of 2")
    if m % bm:
        raise ValueError(f"shape {x.shape} not divisible by block {block_shape}")

    tile_shape = (bm // 128, 128)
    x = x.reshape(m // 128, 128)
    segment_ids = segment_ids.reshape(m // 128, 128)
    grid_spec = pl.GridSpec(
        grid=(m // bm,),
        in_specs=[
            pl.BlockSpec(tile_shape, lambda i: (i, 0)),
            pl.BlockSpec(tile_shape, lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec(tile_shape, lambda i: (i, 0)),
        scratch_shapes=[
            pltpu.VMEM((1,), x.dtype),
            pltpu.VMEM((1,), segment_ids.dtype),
        ],
    )

    return pl.pallas_call(
        _segment_cumsum_kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x, segment_ids).ravel()
