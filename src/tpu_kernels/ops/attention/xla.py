"""XLA attention: jit-wrapped naive — the materialized-O(N²) baseline.

Pedagogically the *wall*: XLA can't fuse the softmax into the K and V
matmuls, so the (N, T, T) attention-scores matrix is written to HBM
between the score matmul and the value matmul, and read back. At long
sequence lengths this read+write dominates HBM bandwidth and the kernel
falls off the roofline. The Pallas (flash) variant is the answer — but
this baseline has to exist first to make the wall measurable.

Importing the naive keeps the two from drifting apart; for this op there
is no JAX-side rewrite that helps anyway (the materialization is the
problem, and rewriting the math doesn't avoid it).
"""

from __future__ import annotations

import jax

from tpu_kernels.ops.attention.naive import attention as _naive

attention = jax.jit(_naive)
