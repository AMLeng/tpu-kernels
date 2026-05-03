"""Softmax op. Memory-bound; primary lesson is fusion of the stable-form chain.

Computes ``softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`` along the
last axis, with the max and sum reductions accumulated in f32 regardless
of input dtype so the reference is numerically stable for bf16.

xla landed at ~55% HBM BW on v5e — same shortfall as rmsnorm-xla, same
cause (XLA can't keep x in VMEM across the reduction and the divide).
The Pallas variant tiles by full rows and collapses the traffic to
1 read + 1 write.
"""

from tpu_kernels.ops.softmax.naive import softmax as softmax_naive
from tpu_kernels.ops.softmax.pallas import softmax as softmax_pallas
from tpu_kernels.ops.softmax.xla import softmax as softmax_xla

__all__ = ["softmax_naive", "softmax_pallas", "softmax_xla"]
