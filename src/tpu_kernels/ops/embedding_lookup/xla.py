"""XLA embedding lookup: jit-wrapped naive.

For an op this simple there's no JAX-level tuning to do — XLA already
emits a single gather. Importing the naive body keeps the relationship
explicit and prevents the two from drifting apart. Kept here as the
placeholder slot so every op has the same three-variant shape.
"""

from __future__ import annotations

import jax

from tpu_kernels.ops.embedding_lookup.naive import embedding_lookup as _naive

embedding_lookup = jax.jit(_naive)
