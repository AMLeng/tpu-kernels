"""XLA matmul: plain ``jnp.matmul`` under default precision.

On TPU, ``Precision.DEFAULT`` for bf16 inputs already uses the MXU at
its native rate (bf16 multiplies, f32 accumulator) — there's no
JAX-side rewrite that meaningfully outperforms this for a square
matmul. Higher precisions (``HIGH`` / ``HIGHEST``) trade speed for
accuracy by emulating wider products on the MXU; they're the
``--dump-hlo`` exercise the curriculum calls out, not a default.

This is the speed-of-light JAX baseline. The Pallas variant exists
not to beat it (it likely won't, on a square matmul) but to teach the
3-axis grid + scratch-resident f32 accumulator pattern that flash
attention reuses.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.matmul(a, b)
