"""Pallas segment_cumsum: MXU-shift segmented Hillis-Steele.

Tile body on ``x`` laid out as ``(T, 128)``:

1. Within-row segmented HS along lanes — 7 levels. The lane shift on
   ``vals`` is ``vals @ S_k`` where ``S_k[i, j] = 1 iff j == i + 2**k``,
   issued on the MXU in parallel with the VPU sid roll + cmpi. Per
   level the critical chain is then ``MXU shift → truncf → select →
   addf`` rather than ``slice → concat → select → addf`` — one fewer
   VPU op, with the matmul itself off the VPU pipeline. Mask the add
   by sid equality; monotonic sids make sid equality at offset 2**k
   equivalent to "same segment throughout the shift gap".
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

DEFAULT_BLOCK = (262144,)


def _seg_hs_lane(vals: jax.Array, sids: jax.Array, shift_mats_ref) -> jax.Array:
    """Within-row segmented HS inclusive prefix along the lane axis.

    ``vals``: ``(N, 128)`` bf16. ``sids``: ``(N, 128)`` int32.
    ``shift_mats_ref``: ``(7, 128, 128)`` bf16 ref; level ``k`` holds the
    matrix ``S`` with ``S[i, j] = 1 iff j == i + 2**k``, so ``vals @ S``
    produces ``vals`` shifted left by ``2**k`` lanes, zero-padded.

    Each level offloads the lane shift on ``vals`` to the MXU and runs
    the sid roll + cmpi off the VPU critical path so they overlap with
    the MXU. Per level, critical chain shrinks from
    ``slice → concat → select → addf`` (4 VPU ops) to
    ``MXU shift → truncf → select → addf`` (1 MXU + 3 VPU); the truncf
    is forced by Mosaic requiring an f32 matmul accumulator on v5e.
    bf16 matmul against a 0/1 matrix is bitwise exact (every output sum
    has at most one non-zero term, no rounding).
    """
    offset = 1
    level = 0
    while offset < 128:
        # Mosaic requires f32 matmul accumulator on v5e; cast back to bf16
        # adds one truncf to the critical path. Still a small saving over
        # slice + concat (2 ops) since the matmul itself runs on the MXU
        # in parallel with the sid roll and cmpi. Keeping the HS in f32
        # to drop the per-level truncf trades 5 saved casts for 7 levels
        # of 2x-slower f32 VPU ops — measured a wash, so we stay bf16.
        shifted_vals = jnp.dot(
            vals, shift_mats_ref[level], preferred_element_type=jnp.float32
        ).astype(vals.dtype)
        shifted_sids = pltpu.roll(sids, offset, axis=1)
        vals = vals + jnp.where(shifted_sids == sids, shifted_vals, 0)
        offset *= 2
        level += 1
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
    shift_mats_ref,
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

    within_row = _seg_hs_lane(x, sid, shift_mats_ref)

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
    # 7 lane-shift matrices for the within-row HS — one per offset
    # 2**k for k in 0..6. S_k[i, j] = 1 iff j == i + 2**k.
    iota = jnp.arange(128, dtype=jnp.int32)
    shift_mats = jnp.stack(
        [(iota[:, None] + (1 << k) == iota[None, :]).astype(x.dtype) for k in range(7)]
    )
    grid_spec = pl.GridSpec(
        grid=(m // bm,),
        in_specs=[
            pl.BlockSpec(tile_shape, lambda i: (i, 0)),
            pl.BlockSpec(tile_shape, lambda i: (i, 0)),
            pl.BlockSpec(shift_mats.shape, lambda i: (0, 0, 0)),
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
    )(x, segment_ids, shift_mats).ravel()
