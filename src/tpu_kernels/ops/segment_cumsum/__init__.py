"""Segment cumsum: cumulative sum within each segment of a segmented input.

Memory-bound scan (1 add/element, ~8 bytes/element of HBM traffic at bf16
inputs / int32 ids) — speed-of-light is HBM bandwidth.

For each ``i``, ``out[i]`` is the sum of ``x[j]`` for ``j <= i`` with
``segment_ids[j] == segment_ids[i]``: a within-segment prefix sum.
``segment_ids`` must be monotonically non-decreasing, so each segment
occupies a contiguous run of indices — boundaries are derived inside
the kernel as the positions where ``segment_ids`` changes.

Pallas variant pending — A.6 in `docs/curriculum.md`. The naive + xla
pair below is the JAX-side baseline; the Pallas variant is the teaching
artifact for the parallel-prefix scan with a carry across tiles plus
the segment-boundary reset, the pattern that returns at every Stage D
ragged kernel as "where does each segment start in the packed buffer".
"""

from tpu_kernels.ops.segment_cumsum.naive import (
    segment_cumsum as segment_cumsum_naive,
)
from tpu_kernels.ops.segment_cumsum.xla import (
    segment_cumsum as segment_cumsum_xla,
)

__all__ = ["segment_cumsum_naive", "segment_cumsum_xla"]
