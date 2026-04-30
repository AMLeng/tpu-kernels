"""Naive RMSNorm: the obviously-correct reference. Never tuned for perf.

The variance is accumulated in f32 regardless of input dtype — at bf16 the
in-place reduction loses too much precision to serve as a correctness oracle,
and Llama-style rmsnorm is conventionally written this way. The cast back
to the input dtype happens before the per-feature ``scale`` multiply, which
matches the canonical reference (e.g. Llama, MaxText).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def rmsnorm(x: jax.Array, scale: jax.Array, eps: float = 1e-6) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    ms = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    return (x_f32 * jax.lax.rsqrt(ms + eps)).astype(x.dtype) * scale
