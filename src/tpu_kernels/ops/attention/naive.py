"""Naive scaled-dot-product attention: the obviously-correct reference. Never tuned.

Single-sequence, multi-head, no cache. Q, K, V each have shape ``(T, N, H)``
— sequence length, num_heads, head_dim — and the output matches.

The score matmul and softmax accumulate in f32 regardless of input dtype.
The matmul precedent is `ops/matmul/naive.py`: a length-K inner product
loses enough bf16 precision over realistic head dims (and especially over
the full sequence-length sum in the value matmul) that the oracle would
drift. The softmax precedent is `ops/softmax/naive.py`: max/exp/sum/div
over a width-T row needs f32 to stay stable. Output is cast back to the
input dtype at the very end.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    head_dim = q.shape[-1]
    scale = jnp.float32(1.0) / jnp.sqrt(jnp.float32(head_dim))
    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)
    scores = jnp.einsum("qnh,knh->nqk", q_f32, k_f32) * scale
    m = jnp.max(scores, axis=-1, keepdims=True)
    e = jnp.exp(scores - m)
    weights = e / jnp.sum(e, axis=-1, keepdims=True)
    return jnp.einsum("nqk,knh->qnh", weights, v_f32).astype(q.dtype)
