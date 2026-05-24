"""XLA sharded matmul: jit-wrapped naive.

Scaffold-stage placeholder so every op has the same three-variant shape
even before the JAX-tuned baseline exists. Importing the naive body keeps
the two from drifting apart; once we have a non-trivial JAX rewrite
worth comparing — collective-aware reshard, manual reduce_scatter +
all_gather, etc. — this is where it lands.

``dp`` / ``tp`` are ``static_argnames``: they pick the mesh shape, so
they must be compile-time constants rather than traced values.
"""

from __future__ import annotations

import jax

from tpu_kernels.ops.sharded_matmul.naive import matmul as _naive

matmul = jax.jit(_naive, static_argnames=("dp", "tp"))
