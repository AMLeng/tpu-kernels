"""Embedding lookup on the natural 2-D bf16 layout. Memory-bound gather:
read selected rows of a parameter table at integer indices.

Pure data movement (0 flops); speed-of-light is HBM bandwidth. The
minimum HBM traffic is M selected rows of ``params`` read plus M rows
of output written.

This op takes ``params`` in the *natural* ``T(8, 128)(2, 1)`` layout —
the form XLA assigns to TensorCore-bound bf16 tensors, and what most
JAX code already has. HBM physically stores the array as 2 KiB tiles
holding 8 rows interleaved per tile. Mosaic only emits tile-aligned
DMAs against tiled refs, so a single-row read against this layout
isn't expressible from Pallas — the lowering pass rejects sub-tile
slice ops. The Pallas variant has to fetch 8-row slabs and
mask-extract, capping it at ~12.5% BW.

XLA's ``gather_custom_fusion`` (a backend-internal kCustom op compiled
by C++) hits ~32% on this layout, so on natural-2-D **XLA beats the
Pallas variant**. Beating XLA here would require a storage layout
that exposes single-row alignment, which is outside the scope of
this op.
"""

from tpu_kernels.ops.embedding_lookup.naive import (
    embedding_lookup as embedding_lookup_naive,
)
from tpu_kernels.ops.embedding_lookup.pallas import (
    embedding_lookup as embedding_lookup_pallas,
)
from tpu_kernels.ops.embedding_lookup.xla import (
    embedding_lookup as embedding_lookup_xla,
)

__all__ = [
    "embedding_lookup_naive",
    "embedding_lookup_pallas",
    "embedding_lookup_xla",
]
