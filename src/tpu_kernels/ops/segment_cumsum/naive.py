"""Naive segment cumsum: the obviously-correct reference. Never tuned for perf.

Three primitives — ``cumsum`` over the whole input, ``cummax`` to find
each element's segment-start index, then a gather + subtract to strip
off the prefix sum that belongs to earlier segments. All three are
ordinary scans XLA already lowers well; no custom combiner means a
much faster compile than ``lax.associative_scan`` with a segmented
combiner. Boundaries are derived once from ``segment_ids`` (a True at
every position whose id differs from its predecessor, plus the first
index) and fed into ``cummax`` as the only non-primitive step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def segment_cumsum(x: jax.Array, segment_ids: jax.Array) -> jax.Array:
    cs = jnp.cumsum(x)
    segment_starts = jnp.concatenate([jnp.array([True]), segment_ids[1:] != segment_ids[:-1]])
    idx = jnp.arange(x.shape[0])
    seg_start_idx = jnp.where(segment_starts, idx, -1)
    my_start = lax.cummax(seg_start_idx)
    prev_total = jnp.where(my_start > 0, cs[jnp.maximum(my_start - 1, 0)], 0.0)
    return cs - prev_total
