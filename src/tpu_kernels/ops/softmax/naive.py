"""Naive stable softmax: the obviously-correct reference. Never tuned for perf.

Standard max-subtract / log-sum-exp form along the last axis. The max and
sum reductions are accumulated in f32 regardless of input dtype — at bf16,
in-place reductions over a wide hidden dim lose enough precision that the
oracle would drift. The cast back to input dtype happens at the end, after
the divide.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def softmax(x: jax.Array) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    m = jnp.max(x_f32, axis=-1, keepdims=True)
    e = jnp.exp(x_f32 - m)
    return (e / jnp.sum(e, axis=-1, keepdims=True)).astype(x.dtype)
