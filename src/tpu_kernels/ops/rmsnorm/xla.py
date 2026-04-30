"""XLA RMSNorm: jit-wrapped naive.

There's no JAX-level reformulation that helps here — the win is XLA fusing
the (square → mean → rsqrt → mul → scale) chain into one elementwise loop.
The exercise is reading ``--dump-hlo`` to confirm the chain fused. Importing
the naive body keeps the relationship explicit and prevents the two from
drifting apart.
"""

from __future__ import annotations

import jax

from tpu_kernels.ops.rmsnorm.naive import rmsnorm as _naive

rmsnorm = jax.jit(_naive)
