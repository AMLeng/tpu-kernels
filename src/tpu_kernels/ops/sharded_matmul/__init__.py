"""Sharded matmul op (pre-Stage-B exploratory).

``a @ w`` for the data-parallel + tensor-parallel layout
``A[B_x, D_y] @ W[D_y, F] -> Out[B_x, F]`` over a ``(dp, tp)`` mesh: the
batch dim is sharded data-parallel over ``x``, the contracting dim is
sharded tensor-parallel over ``y``. Each device runs a local matmul and
the partials are closed with a single all-reduce over the contracting
axis — the textbook "local compute + all_reduce after" row-parallel
linear that later collectives kernels (Stage B's
``reduce_scatter ∘ all_gather`` decomposition, Stage C's TP MLP) earn
their lessons against.

Not on the curriculum spine yet: this is a stepping stone that lets
``shard_map`` + ``psum`` get exercised end-to-end while the curriculum
itself stays gated on multi-host harness work (see ``docs/curriculum.md``
Stage C preamble). The Pallas slot is intentionally unfilled — the
follow-up kernels are the ones that take it.
"""

from tpu_kernels.ops.sharded_matmul.naive import matmul as sharded_matmul_naive
from tpu_kernels.ops.sharded_matmul.xla import matmul as sharded_matmul_xla

__all__ = ["sharded_matmul_naive", "sharded_matmul_xla"]
