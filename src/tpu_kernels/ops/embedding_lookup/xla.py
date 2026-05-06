"""XLA embedding lookup: jit-wrapped naive.

For ``params[ids]`` on the natural 2-D layout there is no JAX-level
tuning that beats what XLA already emits — the lowered HLO contains a
single ``gather_custom_fusion`` (kind=kCustom), an XLA-internal C++
fusion that lands at ~32% HBM BW. JAX has no surface to express
anything closer to the speed-of-light here. Importing the naive body
keeps the relationship explicit and prevents drift.
"""

from __future__ import annotations

import jax

from tpu_kernels.ops.embedding_lookup.naive import embedding_lookup as _naive

embedding_lookup = jax.jit(_naive)
