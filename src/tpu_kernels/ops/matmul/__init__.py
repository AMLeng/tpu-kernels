"""Matmul op. Compute-bound; the curriculum's first MXU lesson.

Computes ``a @ b`` for 2-D bf16 inputs ``(M, K) @ (K, N) -> (M, N)``,
with the inner-product accumulator in f32 regardless of input dtype.

xla is plain ``jnp.matmul`` — XLA already lowers this to MXU at default
precision (bf16 multiplies, f32 accumulator). Pallas tiles by
``(bm, bn, bk)`` and runs the K loop inside the kernel so the f32
accumulator lives in VMEM scratch — first hands-on exposure to
``lax.dot``/``jnp.dot`` inside ``pallas_call``.

Per CLAUDE.md / curriculum, this is the only op where both xla and
pallas variants are mandatory: even when xla wins, the Pallas
practice (3-axis grid, scratch-resident f32 accumulator) is a
prerequisite for flash attention.
"""

from tpu_kernels.ops.matmul.naive import matmul as matmul_naive
from tpu_kernels.ops.matmul.pallas import matmul as matmul_pallas
from tpu_kernels.ops.matmul.xla import matmul as matmul_xla

__all__ = ["matmul_naive", "matmul_pallas", "matmul_xla"]
