"""XLA cumsum: 3-level hierarchical scan with explicit lane/sublane/tile decomposition.

Reshapes ``x`` as ``(T, INNER/128, 128)`` and runs ``lax.cumsum`` at
three explicit levels — along the 128-lane axis, across sublane-rows
within a tile, and across tiles — broadcasting each level's exclusive
prefix back at the layer above. Pre-decomposing the inner cumsum this
way avoids two of the four full-M layout ``copy`` instructions XLA
inserts when given the implicit ``lax.cumsum(x.reshape(T, INNER),
axis=1)`` form: XLA decomposes the (T, INNER=4096) cumsum into the
same per-128-then-per-32 hierarchy internally, and emits layout
copies *between those stages*; in the explicit form those stages
become separate HLO nodes XLA can stay in one layout for.

Empirical at M=2^28 bf16 on v5e: 18.95% HBM BW, 8.4x over the
triangular ``reduce_window`` baseline. The achieved value is 76%
of the analytical optimum for pure-JAX one-shot at this shape —
~25% — and the optimum is fixed by exactly four full-M
HBM-resident HLO ops the kernel is forced to emit (every other op
stays on-chip in VMEM scratch):

- two work fusions, structurally required by the hierarchy itself:
  the per-128 cumsum that produces partial sums, then the broadcast
  + add that combines them with the cross-tile prefixes;
- two boundary layout copies: input ``{2,1,0}`` → kernel
  ``{0,1,2}`` and the inverse on output, forced by the mismatch
  between x's natural 1-D layout and the cumsum kernel's preferred
  batch-on-lanes layout.

Each op is 1 read + 1 write of M → 8x M total vs 2x M ideal, the
25% optimum. ``jax.experimental.layout.Layout`` can shift the
boundary copies out of this kernel's body but doesn't eliminate
them.

INNER = 4096 chosen empirically. The implicit-form sweep across
M ∈ {2^26..2^29} and INNER ∈ {2^9..2^20} found a flat plateau at
INNER ∈ [2^10, 2^12] and a cliff at INNER = 2^13 where the
per-INNER cumsum needs a second rewriter level. The explicit form
re-tested at M=2^28 hits the same cliff at the same INNER: 4096 is
the largest plateau value, minimising T = M/INNER and the L2-tile
cost. MIDDLE = INNER/128 = 32 follows from the 128-lane v5e VPU.

For ``M < INNER`` falls back to a single-call ``lax.cumsum`` — at
that size the rewriter's HBM round-trips don't dominate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

# Per-tile size. 4096 is the largest INNER on the empirical plateau
# (sweep over M ∈ {2^26..2^29} and INNER ∈ {2^9..2^20}); above the
# cliff at INNER = 2^13 the per-INNER cumsum needs a second rewriter
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

    # L1 lane: 128-element cumsum along the lane axis (within one
    # sublane-row of one tile). Lane-parallel hardware scan, single
    # rewriter level.
    cs_lane = lax.cumsum(xr, axis=2)

    # L1 sublane: cumsum across sublane-rows within a tile, applied to
    # each row's last-lane total. Broadcast the exclusive prefix back
    # across the lane axis to combine with cs_lane → per-tile cumsum.
    sublane_inclusive = lax.cumsum(cs_lane[..., -1], axis=1)
    sublane_exclusive = jnp.pad(sublane_inclusive[:, :-1], ((0, 0), (1, 0)))
    cs_tile = cs_lane + sublane_exclusive[:, :, None]

    # L2 tile: cumsum across per-tile totals. (T,) fits VMEM, so the
    # rewriter's intermediates stay on-chip — no HBM round-trips here.
    tile_totals = sublane_inclusive[:, -1]
    tile_inclusive = lax.cumsum(tile_totals)
    tile_exclusive = jnp.concatenate([jnp.zeros((1,), dtype=x.dtype), tile_inclusive[:-1]])

    return (cs_tile + tile_exclusive[:, None, None]).reshape(-1)[:n]
