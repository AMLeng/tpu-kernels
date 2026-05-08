"""XLA cumsum: 3-level hierarchical scan, triangular-matmul prefix at L1.

Reshapes ``x`` as ``(T, INNER/128, 128)`` and computes the prefix at
three explicit levels — along the 128-lane axis, across sublane-rows
within a tile, and across tiles — broadcasting each level's exclusive
prefix back at the layer above. Pre-decomposing this way avoids two
of the four full-M layout ``copy`` instructions XLA inserts when given
the implicit ``lax.cumsum(x.reshape(T, INNER), axis=1)`` form: XLA
decomposes the (T, INNER=4096) cumsum into the same
per-128-then-per-32 hierarchy internally, and emits layout copies
*between those stages*; in the explicit form those stages become
separate HLO nodes XLA can stay in one layout for.

L1 lane and L1 sublane are *triangular matmuls*, not ``lax.cumsum``:
``cs_lane = xr @ triu(128)`` gives the within-row inclusive prefix in
one MXU pass, and ``sublane_exclusive = row_totals @ strict_triu(32)``
(strict-upper, ones above the diagonal) gives the within-tile
exclusive prefix directly — no separate ``jnp.pad`` shift. Row totals
``jnp.sum(xr, axis=2)`` come from ``xr`` directly rather than slicing
``cs_lane[..., -1]``, so the reduction can fuse with the L1-lane
matmul on the shared ``xr`` read instead of waiting on the matmul's
last-lane output (the same VPU-parallel-row-totals trick that bought
~7 BW pp at the Pallas level).

Empirical at M=2^28 bf16 on v5e: ``cumsum_xla`` lands close to the
~25% analytical optimum for pure-JAX one-shot at this shape, fixed
by four full-M HBM-resident HLO ops the kernel is forced to emit
(every other op stays on-chip in VMEM scratch):

- two work fusions, structurally required by the hierarchy itself:
  the per-128 lane prefix that produces partial sums, then the
  broadcast + add that combines them with the cross-tile prefixes;
- two boundary layout copies: input ``{2,1,0}`` → kernel
  ``{0,1,2}`` and the inverse on output, forced by the mismatch
  between x's natural 1-D layout and the cumsum kernel's preferred
  batch-on-lanes layout.

Each op is 1 read + 1 write of M → 8x M total vs 2x M ideal, the
25% optimum. ``jax.experimental.layout.Layout`` can shift the
boundary copies out of this kernel's body but doesn't eliminate
them.

INNER = 4096 chosen empirically (re-tested with the matmul form at
M=2^28). The implicit-form sweep across M ∈ {2^26..2^29} and INNER
∈ {2^9..2^20} found a flat plateau at INNER ∈ [2^10, 2^12] and a
cliff at INNER = 2^13 where the per-INNER prefix needs a second
rewriter level. 4096 is the largest plateau value, minimising T =
M/INNER and the L2-tile cost. MIDDLE = INNER/128 = 32 follows from
the 128-lane v5e VPU.

For ``M < INNER`` falls back to a single-call ``lax.cumsum`` — at
that size the rewriter's HBM round-trips don't dominate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

# Per-tile size. 4096 is the largest INNER on the empirical plateau
# (sweep over M ∈ {2^26..2^29} and INNER ∈ {2^9..2^20}); above the
# cliff at INNER = 2^13 the per-INNER prefix needs a second rewriter
# level and BW drops ~half. LANES is fixed by hardware (v5e VPU has
# 128 lanes); MIDDLE = INNER / LANES is the within-tile sublane
# dimension.
_INNER = 4096
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
    xr = x.reshape(t, middle, _LANES)

    # L1 lane: within-row inclusive prefix via x @ triu(128). The MXU
    # collapses what would otherwise be a log2(128)=7-level rewriter
    # scan into a single matmul. ``preferred_element_type=f32`` mirrors
    # the implicit f32 accumulation TPU MXUs do for bf16 inputs (and
    # is a no-op for f32 inputs).
    triu_lane = jnp.triu(jnp.ones((_LANES, _LANES), dtype=x.dtype))
    cs_lane = jnp.matmul(xr, triu_lane, preferred_element_type=jnp.float32).astype(x.dtype)

    # Row totals come from a fresh reduction over ``xr`` rather than
    # slicing ``cs_lane[..., -1]``: the slice would serialise on the
    # MXU's last-lane output, while the reduction shares ``xr``'s read
    # with the L1-lane matmul above and lets XLA fuse them. Same trick
    # that bought ~7 BW pp at the Pallas level.
    row_totals = jnp.sum(xr, axis=2)  # (T, MIDDLE)

    # L1 sublane: within-tile *exclusive* prefix via row_totals @
    # strict-upper-triangular ones (1 strictly above diagonal). No
    # follow-up ``jnp.pad`` shift — the strict diagonal absorbs it.
    strict_triu_sub = jnp.triu(jnp.ones((middle, middle), dtype=x.dtype), k=1)
    sublane_exclusive = jnp.matmul(
        row_totals, strict_triu_sub, preferred_element_type=jnp.float32
    ).astype(x.dtype)
    cs_tile = cs_lane + sublane_exclusive[:, :, None]

    # L2 tile: cumsum across per-tile totals. (T,) fits VMEM, so the
    # rewriter's intermediates stay on-chip — no HBM round-trips here.
    tile_totals = jnp.sum(row_totals, axis=1)
    tile_inclusive = lax.cumsum(tile_totals)
    tile_exclusive = jnp.concatenate([jnp.zeros((1,), dtype=x.dtype), tile_inclusive[:-1]])

    return (cs_tile + tile_exclusive[:, None, None]).reshape(-1)[:n]
