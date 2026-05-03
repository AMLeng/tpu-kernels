"""XLA softmax: online form to fold max and sum into one reduction pass.

Naive softmax reads x twice in the reduction phase — once for max,
once for sum(exp(x - max)). Online softmax pairs them: a single
reduction over (x_i, 1) with the recurrence
``m' = max(m_a, m_b)``, ``d' = d_a * exp(m_a - m') + d_b * exp(m_b - m')``
yields (max, denom) from one pass over x. The divide step is a second
read of x, so total HBM traffic is 2 reads of x + 1 write of y — the
same shape as rmsnorm-xla.

Hypothesis: this matches rmsnorm-xla's ~55% HBM BW for the same reason
(one fewer read of x). Pallas is the next step — tiling lets the
reduction and divide phases share x while it sits in VMEM, collapsing
to 1 read of x + 1 write of y.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def softmax(x: jax.Array) -> jax.Array:
    axis = -1 % x.ndim
    denoms = jnp.ones_like(x, dtype=jnp.float32)
    x_f32 = x.astype(jnp.float32)

    def combine(a, b):
        a_m, a_d = a
        b_m, b_d = b
        m = jnp.maximum(a_m, b_m)
        d = a_d * jnp.exp(a_m - m) + b_d * jnp.exp(b_m - m)
        return (m, d)

    m, d = jax.lax.reduce((x_f32, denoms), (-jnp.inf, 0.0), combine, (axis,))
    return (jnp.exp(x_f32 - jnp.expand_dims(m, axis)) / jnp.expand_dims(d, axis)).astype(x.dtype)
