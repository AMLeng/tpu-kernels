"""Embedding lookup. Memory-bound gather: read selected rows of a parameter
table at integer indices.

Pure data movement (0 flops); speed-of-light is HBM bandwidth. The
minimum HBM traffic is M selected rows of `params` read in plus M rows
of output written out (the `M * 4` bytes of indices is negligible at
production sizes).

Pallas variant pending — A.5 in `docs/curriculum.md`. The naive + xla
pair below is the JAX-side baseline; the Pallas variant is the
teaching artifact for data-dependent indexing inside a kernel
(`pl.dynamic_slice` against runtime offsets, or a runtime-driven
`BlockSpec` `index_map`), and re-imports as `embedding_lookup_pallas`
when added.
"""

from tpu_kernels.ops.embedding_lookup.naive import (
    embedding_lookup as embedding_lookup_naive,
)
from tpu_kernels.ops.embedding_lookup.xla import (
    embedding_lookup as embedding_lookup_xla,
)

__all__ = ["embedding_lookup_naive", "embedding_lookup_xla"]
