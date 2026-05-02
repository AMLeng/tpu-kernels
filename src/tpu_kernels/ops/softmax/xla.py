"""XLA softmax: jit-wrapped naive.

The lesson is whether XLA fuses the stable-form chain
(max → sub → exp → sum → div) into a single elementwise loop. Confirming
that fusion happened — and identifying the breaks if it didn't — is the
exercise; run the suite with ``--dump-hlo`` to inspect. Importing the
naive body keeps the relationship explicit and prevents the two from
drifting apart.
"""

from __future__ import annotations

import jax

from tpu_kernels.ops.softmax.naive import softmax as _naive

softmax = jax.jit(_naive)
