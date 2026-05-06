"""XLA embedding lookup on the packed 4-D shape: jit-wrapped naive.

On the packed layout (rows contiguous in HBM, byte-equivalent to 1-D
``T(1024)(128)(2, 1)``), XLA emits per-row reads + a bulk write back
to HBM at ~65% BW. There's no JAX-level tuning lever beyond the
layout itself.
"""

from __future__ import annotations

import jax

from tpu_kernels.ops.embedding_lookup_packed.naive import embedding_lookup as _naive

embedding_lookup = jax.jit(_naive)
