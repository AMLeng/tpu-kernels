"""Naive matmul: the obviously-correct reference. Never tuned for perf.

The K reduction is performed in f32 regardless of input dtype — at bf16,
a length-K inner product over a wide K (typically 4k-16k in transformer
shapes) loses enough precision that the oracle would drift. Output is
cast back to input dtype after the matmul, mirroring the standard
"bf16 in, f32 accumulator, bf16 out" convention every variant follows.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return (a.astype(jnp.float32) @ b.astype(jnp.float32)).astype(a.dtype)
