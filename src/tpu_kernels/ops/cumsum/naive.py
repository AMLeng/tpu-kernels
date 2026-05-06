"""Naive cumsum: the obviously-correct reference. Never tuned for perf.

Single call to ``jnp.cumsum``. The whole point of this op is to read
plain prefix-sum's HBM bandwidth on the same shape regime as
``segment_cumsum``, so the naive form is the simplest possible — no
segment-reset, no online recurrence, just the library primitive.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def cumsum(x: jax.Array) -> jax.Array:
    return jnp.cumsum(x)
