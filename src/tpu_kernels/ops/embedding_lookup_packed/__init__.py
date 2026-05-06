"""Embedding lookup on a *packed* layout that exposes single-row DMAs.

The companion ``embedding_lookup`` op runs on the natural 2-D
``T(8, 128)(2, 1)`` layout and is structurally capped at ~12.5% HBM
BW in Pallas. The cap comes from how that layout meets Mosaic's DMA
emission: HBM stores the array as 2 KiB tiles holding 8 rows
interleaved, and Mosaic only emits tile-aligned DMAs against tiled
refs — so a single-row read isn't expressible from Pallas. On that
layout XLA's ``gather_custom_fusion`` (~32%) wins.

This op flips it: ``params`` arrives in a 4-D shape ``(vocab,
hidden//1024, 8, 128)`` whose natural Mosaic layout is
``T(8, 128)(2, 1)`` on the *inner* two dims. That layout is byte-
equivalent to the 1-D ``T(1024)(128)(2, 1)`` form (both correspond to
the bf16 packing pair landing on adjacent in-row positions instead of
adjacent rows), so a per-row DMA fetches exactly one useful row.

Both emitters benefit. XLA's stock gather lands at ~65% BW; the
Pallas kernel (explicit read-many / write-one) edges it out at ~69%
— the layout is doing most of the work.

Caller responsibility: pre-format ``params`` once into the packed
shape + layout. The byte reorder vs natural 2-D is a real ~1 GiB
copy at production sizes, amortized across many subsequent lookups.
"""

from tpu_kernels.ops.embedding_lookup_packed.naive import (
    embedding_lookup as embedding_lookup_packed_naive,
)
from tpu_kernels.ops.embedding_lookup_packed.pallas import (
    embedding_lookup as embedding_lookup_packed_pallas,
)
from tpu_kernels.ops.embedding_lookup_packed.xla import (
    embedding_lookup as embedding_lookup_packed_xla,
)

__all__ = [
    "embedding_lookup_packed_naive",
    "embedding_lookup_packed_pallas",
    "embedding_lookup_packed_xla",
]
