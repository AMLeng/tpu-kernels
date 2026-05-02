"""Softmax op. Memory-bound; primary lesson is fusion of the stable-form chain.

Computes ``softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`` along the
last axis, with the max and sum reductions accumulated in f32 regardless
of input dtype so the reference is numerically stable for bf16.

xla-only at scaffold time per ``docs/curriculum.md`` — a Pallas variant
joins only if XLA misses the 80% HBM BW target on v5e.
"""

from tpu_kernels.ops.softmax.naive import softmax as softmax_naive
from tpu_kernels.ops.softmax.xla import softmax as softmax_xla

__all__ = ["softmax_naive", "softmax_xla"]
