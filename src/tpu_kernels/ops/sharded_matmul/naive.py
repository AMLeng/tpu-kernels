"""Naive sharded matmul: the obviously-correct reference. Never tuned for perf.

Computes ``a @ w`` for the data-parallel + tensor-parallel layout

    A[B_x, D_y] @ W[D_y, F] -> Out[B_x, F_y]

over a ``(dp, tp)`` mesh on axes ``("x", "y")``. The batch dim B is
sharded data-parallel over ``x``; the contracting dim D is sharded
tensor-parallel over ``y`` (and W is sharded on D over the same ``y``
axis, replicated over ``x``). Each device runs a local
``(B/dp, D/tp) @ (D/tp, F)`` matmul that yields a per-device ``(B/dp, F)``
*partial* sum, then a single ``psum_scatter`` over the ``y`` axis reduces
the partials and scatters the result along F, leaving the output sharded
on B over ``x`` and on F over ``y`` (``Out[B_x, F_y]``).

This is the reduce-scatter half of the row-parallel linear: ``all_reduce
= all_gather ∘ reduce_scatter``, so where an all-reduce would replicate the
full output across ``y``, this keeps only each chip's F-shard. The single
``psum_scatter`` is the obvious-correct oracle the hand-rolled ring in
``xla.py`` is checked against; ``shard_map`` keeps the collective explicit.

The K reduction accumulates in f32 regardless of input dtype, mirroring
``ops/matmul/naive.py``: at bf16 the length-D inner product over
realistic D (4k-16k) loses enough precision that the oracle would
drift; output is cast back to the input dtype after the scatter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from tpu_kernels.ops.sharded_matmul.sharding import input_specs, output_spec


def matmul(a: jax.Array, w: jax.Array, *, mesh: Mesh) -> jax.Array:
    """Row-parallel ``a @ w`` over the caller-supplied ``mesh``.

    ``mesh`` is passed in rather than built here: the harness owns the
    sharding source of truth (see ``sharding.py``) and hands the kernel
    inputs already placed on it, mirroring a production layer. The axis
    names and per-tensor layout come from ``mesh`` / ``sharding.py``, so the
    contracting axis the ``psum_scatter`` reduces over is whatever the mesh's
    second axis is named.
    """
    _dp_axis, tp_axis = mesh.axis_names

    def _local(a_local: jax.Array, w_local: jax.Array) -> jax.Array:
        partial = a_local.astype(jnp.float32) @ w_local.astype(jnp.float32)
        # Reduce-scatter over the contracting axis: sum the per-shard partials
        # and scatter the full-F result along F (dim 1) so chip y keeps Out[:, F_y].
        scattered = jax.lax.psum_scatter(
            partial, axis_name=tp_axis, scatter_dimension=1, tiled=True
        )
        return scattered.astype(a_local.dtype)

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=input_specs(mesh),
        out_specs=output_spec(mesh),
    )(a, w)
