"""XLA sharded matmul: manual all-reduce collective matmul.

Row-parallel ``A[B_x, D_y] @ W[D_y, F] -> Out[B_x, F]`` under ``shard_map``.
Each device computes a local ``(B/dp, D/tp) @ (D/tp, F_block)`` partial over
its contracting-dim shard, then ``psum``s that partial over the ``tp`` axis to
close the contraction. The output ``F`` dim is walked in ``stride``-wide
blocks via ``fori_loop`` — one all-reduce per block — rather than a single
monolithic collective over the whole output.

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
    names and per-tensor layout come from ``mesh`` / ``sharding.py``, so the
    contracting axis the ``psum`` reduces over is whatever the mesh's second
    axis is named.
    """
    _dp_axis, tp_axis = mesh.axis_names

    stride = min(1024, w.shape[1])

    def _local(a_local: jax.Array, w_local: jax.Array) -> jax.Array:
        result = jax.lax.pcast(
            jnp.empty((a_local.shape[0], w_local.shape[1]), dtype=a_local.dtype),
            (_dp_axis,),
            to="varying",
        )

        def loop_body(i, result):
            w_slice = jax.lax.dynamic_slice_in_dim(w_local, i * stride, stride, axis=1)
            local_result = a_local.astype(jnp.float32) @ w_slice
            result_slice = jax.lax.psum(local_result, axis_name=tp_axis).astype(result.dtype)
            slice_loc = (jnp.int32(0), i * stride)
            result = jax.lax.dynamic_update_slice(result, result_slice, slice_loc)
            return result

        result = jax.lax.fori_loop(0, w_local.shape[1] // stride, loop_body, result)
        return result

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=input_specs(mesh),
        out_specs=output_spec(mesh),
    )(a, w)
