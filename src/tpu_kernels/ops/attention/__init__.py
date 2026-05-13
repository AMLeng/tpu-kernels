"""Attention op (Stage D.1): no-cache, single-seq, fwd.

Scaled-dot-product attention on a single sequence. Q, K, V each shape
``(T, N, H)``; output matches. Pedagogically the entry point to Stage D
— the variant lineup carries the "XLA can't fuse softmax into the
matmul" wall: xla materializes the (N, T, T) attention scores in HBM
and BW% collapses; the Pallas (flash) variant streams K/V tiles and an
online softmax to keep the scores out of HBM.

The Pallas variant is not in this scaffold yet; it lands in a follow-up
once the naive + xla baseline is wired through and the wall is
measurable.
"""

from tpu_kernels.ops.attention.naive import attention as attention_naive
from tpu_kernels.ops.attention.xla import attention as attention_xla

__all__ = ["attention_naive", "attention_xla"]
