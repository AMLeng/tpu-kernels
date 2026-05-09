"""XLA cumsum: 3-level hierarchical scan, triangular-matmul prefix at L1.

Reshapes ``x`` as ``(T*MIDDLE, LANES)`` and computes the prefix at
three explicit levels — along the 128-lane axis, across sublane-rows
within a tile, and across tiles — broadcasting each level's exclusive
prefix back at the layer above. The implicit
``lax.cumsum(x.reshape(T, INNER), axis=1)`` form decomposes into the
same hierarchy internally but emits layout copies between stages;
explicit decomposition keeps XLA in one layout.

L1 lane and L1 sublane are *triangular matmuls*: ``cs_lane = xr @
triu(128)`` (inclusive within-row prefix in one MXU pass) and
``sublane_exclusive = row_totals @ strict_triu(MIDDLE)`` (exclusive
within-tile prefix; strict-upper diagonal absorbs the shift).

Two non-obvious choices are load-bearing for the BW the kernel hits:

1. **2-D reshape** ``(T*MIDDLE, LANES)`` rather than 3-D
   ``(T, MIDDLE, LANES)``. Two batch dims push XLA's matmul emitter to
   ``EmitAllBatchInSublanes`` with output layout ``{2,0,1}``, forcing
   a full-M ``copy`` to ``{2,1,0}`` for the final ``reshape(-1)``. One
   batch dim → ``{1,0}`` output, bitcast-equal to the 1-D output
   layout. Saves 2x M of HBM traffic.

2. **Combined exclusive prefix** in a single ``(T*MIDDLE,)`` vector
   before the lane broadcast. Reshaping ``cs_lane`` back to 3-D and
   adding ``sublane_exclusive[:, :, None]`` and
   ``tile_exclusive[:, None, None]`` separately forces XLA to
   materialise both broadcasts as full-M HBM staging tensors instead
   of fusing them into the add.

Result on v5e at M=2^28 bf16: ~54% HBM BW. The row-totals reduction
and the L1 matmul + broadcast-add can't fuse — XLA's loop fuser
refuses on account of the cross-tile ``tile_exclusive`` dep — so x is
read twice; with the single output write that's 3x M traffic vs 2x M
ideal → ~67% structural efficiency, times v5e's ~80% sustained
ceiling ≈ ~54% measured. Beyond ~67% needs the duplicate x-read
folded out, or a streaming ``lax.scan``.

INNER = 16384 chosen empirically at M=2^28 with the 2-D matmul form
(plateau spans INNER ∈ [2^13, 2^17], peak at 2^14). The old cliff at
INNER=2^13 from the ``lax.cumsum``-decomposed form is gone — the
triangular-matmul prefix doesn't trip the second rewriter level.
MIDDLE = INNER/128 follows from the 128-lane v5e VPU. For
``M < INNER`` falls back to a single-call ``lax.cumsum``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

_INNER = 16384
_LANES = 128


@jax.jit
def cumsum(x: jax.Array) -> jax.Array:
    (n,) = x.shape
    if n < _INNER:
        return lax.cumsum(x)

    pad = (-n) % _INNER
    if pad:
        x = jnp.concatenate([x, jnp.zeros((pad,), dtype=x.dtype)])
    t = x.shape[0] // _INNER
    middle = _INNER // _LANES
    xr = x.reshape(t * middle, _LANES)

    # L1 lane: within-row inclusive prefix in one MXU pass. f32 acc
    # mirrors the implicit acc TPU MXUs do for bf16 inputs.
    triu_lane = jnp.triu(jnp.ones((_LANES, _LANES), dtype=x.dtype))
    cs_lane = jnp.matmul(xr, triu_lane, preferred_element_type=jnp.float32).astype(x.dtype)

    # Fresh reduction over xr rather than slicing cs_lane[..., -1]:
    # the slice serialises on the MXU's last-lane output while the
    # reduction lets XLA fuse the read with the L1-lane matmul.
    row_totals = jnp.sum(xr, axis=1).reshape(t, middle)

    # L1 sublane: within-tile exclusive prefix; strict-upper diagonal
    # absorbs the shift, no follow-up jnp.pad needed.
    strict_triu_sub = jnp.triu(jnp.ones((middle, middle), dtype=x.dtype), k=1)
    sublane_exclusive = jnp.matmul(
        row_totals, strict_triu_sub, preferred_element_type=jnp.float32
    ).astype(x.dtype)

    # L2 tile prefix: (T,) fits VMEM, no HBM round-trip.
    tile_totals = jnp.sum(row_totals, axis=1)
    tile_inclusive = lax.cumsum(tile_totals)
    tile_exclusive = jnp.concatenate([jnp.zeros((1,), dtype=x.dtype), tile_inclusive[:-1]])

    # Fold both exclusive prefixes into one (T*MIDDLE,) vector before
    # broadcasting against the LANES axis — keeps the add inside the
    # matmul fusion. Broadcasting them separately against a 3-D
    # ``cs_lane`` reshape forces XLA to materialise full-M staging
    # tensors for each.
    combined = (sublane_exclusive + tile_exclusive[:, None]).reshape(t * middle)

    return (cs_lane + combined[:, None]).reshape(-1)[:n]
