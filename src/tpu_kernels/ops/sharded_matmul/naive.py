"""Naive sharded matmul: the obviously-correct reference. Never tuned for perf.

Computes ``a @ w`` for the data-parallel + tensor-parallel layout

    A[B_x, D_y] @ W[D_y, F] -> Out[B_x, F]

over a ``(dp, tp)`` mesh on axes ``("x", "y")``. The batch dim B is
sharded data-parallel over ``x``; the contracting dim D is sharded
tensor-parallel over ``y`` (and W is sharded on D over the same ``y``
axis, replicated over ``x``). Each device runs a local
``(B/dp, D/tp) @ (D/tp, F)`` matmul that yields a per-device ``(B/dp, F)``
*partial* sum, then a single ``psum`` over the ``y`` axis closes the
partials into the full output (sharded on B over ``x``, replicated over
``y``).

The all-reduce is over the contracting axis ``y`` only — that's the
textbook "local compute + all_reduce after" row-parallel linear the
Stage B ``reduce_scatter ∘ all_gather`` decomposition and the Stage C TP
MLP earn their lessons against. ``shard_map`` keeps the collective
explicit.

The K reduction accumulates in f32 regardless of input dtype, mirroring
``ops/matmul/naive.py``: at bf16 the length-D inner product over
realistic D (4k-16k) loses enough precision that the oracle would
drift; output is cast back to the input dtype after the psum.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

# x = data-parallel (batch) axis; y = tensor-parallel (contracting) axis.
MESH_AXES: tuple[str, str] = ("x", "y")
DEFAULT_DP = 2
DEFAULT_TP = 2


def make_mesh(dp: int, tp: int) -> Mesh:
    """Build the ``(dp, tp)`` device mesh on ``MESH_AXES``.

    ``dp * tp`` must equal the number of available devices;
    ``create_device_mesh`` raises otherwise.
    """
    return Mesh(mesh_utils.create_device_mesh((dp, tp)), MESH_AXES)


def matmul(a: jax.Array, w: jax.Array, *, dp: int = DEFAULT_DP, tp: int = DEFAULT_TP) -> jax.Array:
    dp_axis, tp_axis = MESH_AXES
    mesh = make_mesh(dp, tp)

    def _local(a_local: jax.Array, w_local: jax.Array) -> jax.Array:
        partial = a_local.astype(jnp.float32) @ w_local.astype(jnp.float32)
        return jax.lax.psum(partial, axis_name=tp_axis).astype(a_local.dtype)

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(dp_axis, tp_axis), P(tp_axis, None)),
        out_specs=P(dp_axis, None),
    )(a, w)
