"""Cumsum: cumulative sum over a 1-D array.

Memory-bound scan (1 add/element, ~4 bytes/element of HBM traffic at
bf16) — speed-of-light is HBM bandwidth. No-segment-reset
stepping-stone for ``segment_cumsum`` (A.6 in `docs/curriculum.md`):
running this side-by-side with ``segment_cumsum_xla`` on the same
shape regime attributes any delta to the segment-reset path rather
than to the scan itself. Off the curriculum on purpose; no Pallas
variant.
"""

from tpu_kernels.ops.cumsum.naive import cumsum as cumsum_naive
from tpu_kernels.ops.cumsum.xla import cumsum as cumsum_xla

__all__ = ["cumsum_naive", "cumsum_xla"]
