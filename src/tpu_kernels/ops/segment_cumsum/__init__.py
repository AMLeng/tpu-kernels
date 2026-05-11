"""Segment cumsum: cumulative sum within each segment of a segmented input.

Memory-bound scan (1 add/element, ~8 bytes/element of HBM traffic at bf16
inputs / int32 ids) — speed-of-light is HBM bandwidth.

For each ``i``, ``out[i]`` is the sum of ``x[j]`` for ``j <= i`` with
``segment_ids[j] == segment_ids[i]``: a within-segment prefix sum.
``segment_ids`` must be monotonically non-decreasing, so each segment
occupies a contiguous run of indices — boundaries are derived inside
the kernel as the positions where ``segment_ids`` changes.

The Pallas variant is VPU-only: a two-pass segmented Hillis-Steele.
Within-row, 7 lane shifts fold the per-row prefix; across rows, a
small seg-HS on the per-row tails (1-D over T) yields the row carry,
which is broadcast and added back only where the entering segment id
matches. No MXU work; HBM bandwidth is the floor.
"""

from tpu_kernels.ops.segment_cumsum.naive import (
    segment_cumsum as segment_cumsum_naive,
)
from tpu_kernels.ops.segment_cumsum.pallas import (
    segment_cumsum as segment_cumsum_pallas,
)
from tpu_kernels.ops.segment_cumsum.xla import (
    segment_cumsum as segment_cumsum_xla,
)

__all__ = ["segment_cumsum_naive", "segment_cumsum_pallas", "segment_cumsum_xla"]
