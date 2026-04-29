"""XLA scale: jit-wrapped naive.

For an op this simple there's no JAX-level tuning to do — XLA already emits
a single elementwise loop. Importing the naive body keeps the relationship
explicit and prevents the two from drifting apart. Kept here as the
placeholder slot so every op has the same three-variant shape.
"""

from __future__ import annotations

import jax

from tpu_kernels.ops.scale.naive import scale as _naive

scale = jax.jit(_naive)
