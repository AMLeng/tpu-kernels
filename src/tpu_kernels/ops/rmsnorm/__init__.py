"""RMSNorm op. Memory-bound; primary lessons are accumulator dtype and fusion.

Computes ``y = x * rsqrt(mean(x ** 2) + eps) * scale`` along the last axis,
with the variance accumulated in f32 regardless of input dtype so the
reference is numerically stable for bf16.
"""

from tpu_kernels.ops.rmsnorm.naive import rmsnorm as rmsnorm_naive
from tpu_kernels.ops.rmsnorm.pallas import rmsnorm as rmsnorm_pallas
from tpu_kernels.ops.rmsnorm.xla import rmsnorm as rmsnorm_xla

__all__ = ["rmsnorm_naive", "rmsnorm_pallas", "rmsnorm_xla"]
