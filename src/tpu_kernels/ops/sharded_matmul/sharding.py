"""Sharding source of truth for the sharded_matmul op.

The mesh, its axis naming, and the per-tensor partition layout live here —
*not* inside the kernel functions — so the caller (bench suite, correctness
test) owns them the way a production driver does: build the mesh once,
``device_put`` the inputs onto it, then pass the mesh into the layer. The
kernels (``naive.matmul`` / ``xla.matmul``) take the mesh and operate on
already-sharded inputs, mirroring how a real model serves a sharded
activation/weight rather than resharding on every call.

That separation is also what keeps the bench honest: if inputs arrive
unsharded (e.g. a single replicated array on device 0), the timed call pays
a per-call scatter to lay them out for ``shard_map`` — overhead that a
production layer never sees. ``shard_inputs`` removes it by placing each
tensor on exactly the ``NamedSharding`` the kernel's ``in_specs`` expect.

Layout for ``A[B_x, D_y] @ W[D_y, F] -> Out[B_x, F]``:

* ``A``  — ``P(x, y)``: batch sharded data-parallel over ``x``, contracting
  dim sharded tensor-parallel over ``y``.
* ``W``  — ``P(y, None)``: contracting dim sharded over ``y``, replicated
  over ``x``.
* ``Out``— ``P(x, None)``: batch sharded over ``x``, replicated over ``y``
  (the ``psum`` over ``y`` closes the partials).
"""

from __future__ import annotations

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
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


def input_specs(mesh: Mesh) -> tuple[P, P]:
    """``(a_spec, w_spec)`` partition specs for the op's two inputs.

    Derived from ``mesh.axis_names`` so the op's layout follows whatever
    axes the passed mesh declares. This is the single definition both the
    kernel's ``shard_map`` ``in_specs`` and ``shard_inputs`` read, so the
    placement the harness performs can never drift from what the kernel
    expects.
    """
    dp_axis, tp_axis = mesh.axis_names
    return P(dp_axis, tp_axis), P(tp_axis, None)


def output_spec(mesh: Mesh) -> P:
    """Partition spec for the output: batch sharded over ``dp``, output feature
    dim sharded over ``tp`` — the reduce-scatter (``psum_scatter`` in naive, a
    ring in xla) reduces the partials and scatters the result along F."""
    dp_axis, tp_axis = mesh.axis_names
    return P(dp_axis, tp_axis)


def shard_inputs(mesh: Mesh, a: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Place ``a`` and ``w`` on the ``NamedSharding`` the kernel expects.

    Returns the inputs committed to ``input_specs(mesh)`` via ``device_put``
    so the timed call sees them already laid out — no per-call reshard. Use
    this in the harness before handing inputs to a kernel variant.
    """
    a_spec, w_spec = input_specs(mesh)
    a_sharded = jax.device_put(a, NamedSharding(mesh, a_spec))
    w_sharded = jax.device_put(w, NamedSharding(mesh, w_spec))
    return a_sharded, w_sharded
