"""XLA sharded matmul: jit-wrapped naive.

Scaffold-stage placeholder so every op has the same three-variant shape
even before the JAX-tuned baseline exists. Importing the naive body keeps
the two from drifting apart; once we have a non-trivial JAX rewrite
worth comparing — collective-aware reshard, manual reduce_scatter +
all_gather, etc. — this is where it lands.

``mesh`` is a ``static_argname``: it carries the sharding the caller
(harness, see ``sharding.py``) built and placed the inputs on, so it must
be a compile-time constant rather than a traced value.
"""

from __future__ import annotations

import jax

from tpu_kernels.ops.sharded_matmul.naive import matmul as _naive

matmul = jax.jit(_naive, static_argnames=("mesh",))
