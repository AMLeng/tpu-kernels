"""XLA sharded matmul: hand-rolled ring reduce-scatter matmul.

Row-parallel ``A[B_x, D_y] @ W[D_y, F] -> Out[B_x, F_y]`` under ``shard_map``,
implementing the reduce-scatter with an explicit ``ppermute`` ring rather than
the single ``psum_scatter`` the naive oracle uses. The output F dim is split
into ``axis_size`` chunks of width ``stride = F / axis_size``; over
``axis_size - 1`` ring hops each chip folds its partial for one chunk into a
travelling accumulator and rotates it one step, so after the loop the
accumulator resting on chip ``y`` is exactly chunk ``y`` minus this chip's own
contribution, which the final fold adds. Result: chip ``y`` holds ``Out[:, F_y]``.

``mesh`` is effectively a ``static_argname``: it carries the sharding the
caller (harness, see ``sharding.py``) built and placed the inputs on, so it
must be a compile-time constant rather than a traced value.
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
    names and per-tensor layout come from ``mesh`` / ``sharding.py``.
    """
    _dp_axis, tp_axis = mesh.axis_names

    def _local(a_local: jax.Array, w_local: jax.Array) -> jax.Array:
        axis_size = jax.lax.axis_size(tp_axis)
        rank = jax.lax.axis_index(tp_axis)
        stride = w.shape[1] // axis_size

        accumulator_type = jnp.float32
        # accumulator_type = a_local.dtype
        result = jax.lax.pcast(
            jnp.zeros((a_local.shape[0], stride), dtype=accumulator_type),
            (_dp_axis, tp_axis),
            to="varying",
        )

        perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]

        def loop_body(i, result):
            # Fold this chip's partial for chunk (rank + axis_size-1-i) % axis_size
            # into the travelling accumulator, then rotate one hop. After axis_size-1 hops
            # the buffer landing on chip `rank` is the (incomplete) sum for chunk
            # `rank` — missing only this chip's own term, added below.
            chunk = (rank + axis_size - 1 - i) % axis_size
            w_slice = jax.lax.dynamic_slice_in_dim(w_local, chunk * stride, stride, axis=1)
            local_result = jnp.dot(a_local, w_slice, preferred_element_type=accumulator_type)
            result = result + local_result
            result = jax.lax.ppermute(result, axis_name=tp_axis, perm=perm)
            return result

        # result = jax.lax.fori_loop(0, axis_size - 1, loop_body, result)
        for i in range(axis_size - 1):
            result = loop_body(i, result)

        w_slice = jax.lax.dynamic_slice_in_dim(w_local, rank * stride, stride, axis=1)
        result = result + jnp.dot(a_local, w_slice, preferred_element_type=accumulator_type)
        return result.astype(a_local.dtype)

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=input_specs(mesh),
        out_specs=output_spec(mesh),
    )(a, w)
