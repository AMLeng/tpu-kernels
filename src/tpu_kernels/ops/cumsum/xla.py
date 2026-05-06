"""XLA cumsum: explicit triangular reduce_window.

Equivalent to ``jax.jit(jnp.cumsum)``: ``jnp.cumsum`` lowers to
``cumsum_p``, whose default lowering is exactly this triangular
``reduce_window``  (``cumred_reduce_window_impl`` in
``jax/_src/lax/control_flow/loops.py`` — "an O(N^2) reduce-window
implementation"). XLA on TPU then rewrites that triangular form into a
3-level work-efficient parallel scan using the 128-lane vector unit at
each level, so the kernel that actually runs is far from O(N^2).

Spelling the call out as ``lax.reduce_window`` makes the contract
visible: the rewriter operates on the post-traced HLO and matches a
``reduce_window`` instruction with the cumsum-shaped configuration
(window=N, stride=1, padding=(N-1, 0), ``add`` reducer) — not on any
JAX-side symbol. The other two pure-JAX paths produce different HLO
graphs that don't trigger this rewriter and underperform on v5e for
unrelated reasons: a chunked ``lax.scan`` (cumsum-within-tile + scalar
carry, the only ``lax.scan`` shape worth considering on TPU) is
competitive with this variant but doesn't reach HBM-bound because it
can't pipeline async DMAs across iterations; ``lax.associative_scan``
is a Python-side recursive expansion into O(N log N) HLO that doesn't
compile at scale. See A.6's curriculum entry for the full breakdown.

The HLO emitted here is structurally identical to ``jax.jit(jnp.cumsum)``
on the same input (same ``reduce-window`` count, same fusion structure),
and benches at the same BW% — the reduce_window form is the contract,
the jnp.cumsum equivalence is the consequence.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def cumsum(x: jax.Array) -> jax.Array:
    (n,) = x.shape
    return lax.reduce_window(
        x,
        jnp.array(0, dtype=x.dtype),
        lax.add,
        window_dimensions=(n,),
        window_strides=(1,),
        padding=((n - 1, 0),),
    )
