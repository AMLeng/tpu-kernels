"""XLA segment cumsum: jit-wrapped naive.

The naive is already three primitive scans plus a gather — XLA lowers
each well and there's no JAX-level reformulation that meaningfully
outperforms it. The Pallas variant will earn its slot on the
parallel-prefix + segment-reset technique rather than on speed.
Importing the naive body keeps the relationship explicit and prevents
the two from drifting apart.
"""

from __future__ import annotations

import jax

from tpu_kernels.ops.segment_cumsum.naive import segment_cumsum as _naive

segment_cumsum = jax.jit(_naive)
