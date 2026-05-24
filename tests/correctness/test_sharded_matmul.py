"""Correctness tests for sharded_matmul variants.

Runs on any host that exposes >=4 devices (enough for the (dp, tp) combos
below). The cheap way to satisfy that on CPU is
`XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run pytest
tests/correctness/test_sharded_matmul.py`; the CI default of 1 CPU device
skips the module cleanly via the marker below.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_kernels.ops.sharded_matmul import sharded_matmul_naive, sharded_matmul_xla
from tpu_kernels.ops.sharded_matmul.sharding import (
    MESH_AXES,
    input_specs,
    make_mesh,
    output_spec,
    shard_inputs,
)

MIN_DEVICES = 4  # largest dp*tp product exercised below

pytestmark = pytest.mark.skipif(
    len(jax.devices()) < MIN_DEVICES,
    reason=(
        f"needs >={MIN_DEVICES} devices for the mesh combos; "
        "re-run with XLA_FLAGS=--xla_force_host_platform_device_count=4 on CPU"
    ),
)

VARIANTS: dict[str, Callable[..., jax.Array]] = {
    "xla": sharded_matmul_xla,
}

# (dp, tp) meshes that multiply to MIN_DEVICES: pure DP (tp=1, no all_reduce),
# the 2x2 default, and pure TP (dp=1, all_reduce over every device).
MESHES: list[tuple[int, int]] = [(4, 1), (2, 2), (1, 4)]

# Mirrors tests/correctness/test_matmul.py: bf16 tolerance loose enough to
# absorb a length-D bf16-multiply / f32-accumulate inner product (the MXU's
# default precision), tight on f32. Shapes are small (D=256) so the
# accumulator's bf16 drift stays well inside rtol.
DTYPES: dict[str, tuple[Any, dict[str, float]]] = {
    "bf16": (jnp.bfloat16, {"rtol": 2e-2, "atol": 1e-2}),
    "f32": (jnp.float32, {"rtol": 5e-5, "atol": 5e-5}),
}


def _inputs(dtype: Any, batch: int = 64, d: int = 256, f: int = 64) -> tuple[jax.Array, jax.Array]:
    a = jax.random.normal(jax.random.key(0), (batch, d), dtype=dtype)  # A[B, D]
    w = jax.random.normal(jax.random.key(1), (d, f), dtype=dtype)  # W[D, F]
    return a, w


@pytest.mark.parametrize("mesh_shape", MESHES, ids=[f"dp{dp}_tp{tp}" for dp, tp in MESHES])
@pytest.mark.parametrize("dtype_name", list(DTYPES.keys()))
def test_naive_matches_unsharded_matmul(dtype_name: str, mesh_shape: tuple[int, int]) -> None:
    """Pin the oracle against the unsharded f32-accumulated reference across
    mesh shapes. The reduce-scatter output is sharded on F over tp; out_specs
    reassembles the per-chip F-shards into the full ``[B, F]``, which must equal
    the unsharded matmul. A regression that broke the reduce-scatter (dropped
    the psum_scatter, scattered the wrong dim, or reduced the wrong axis) lands
    here as a tp-fold or wrong-shard error on every output element."""
    dp, tp = mesh_shape
    dtype, tol = DTYPES[dtype_name]
    a, w = _inputs(dtype)
    ref = (a.astype(jnp.float32) @ w.astype(jnp.float32)).astype(dtype)
    mesh = make_mesh(dp, tp)
    a_s, w_s = shard_inputs(mesh, a, w)
    np.testing.assert_allclose(
        np.asarray(sharded_matmul_naive(a_s, w_s, mesh=mesh)),
        np.asarray(ref),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )


@pytest.mark.parametrize("mesh_shape", MESHES, ids=[f"dp{dp}_tp{tp}" for dp, tp in MESHES])
@pytest.mark.parametrize("dtype_name", list(DTYPES.keys()))
@pytest.mark.parametrize("variant", VARIANTS.values(), ids=list(VARIANTS.keys()))
def test_matches_naive(
    variant: Callable[..., jax.Array],
    dtype_name: str,
    mesh_shape: tuple[int, int],
) -> None:
    dp, tp = mesh_shape
    dtype, tol = DTYPES[dtype_name]
    a, w = _inputs(dtype)
    mesh = make_mesh(dp, tp)
    a_s, w_s = shard_inputs(mesh, a, w)
    np.testing.assert_allclose(
        np.asarray(variant(a_s, w_s, mesh=mesh)),
        np.asarray(sharded_matmul_naive(a_s, w_s, mesh=mesh)),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )


# ---- sharding source-of-truth helpers -----------------------------------


def test_make_mesh_builds_named_dp_tp_grid() -> None:
    """make_mesh is the single source of truth for the (dp, tp) topology and
    the ("x", "y") axis naming the kernels read back via mesh.axis_names."""
    mesh = make_mesh(2, 2)
    assert mesh.axis_names == MESH_AXES
    assert mesh.devices.shape == (2, 2)


def test_shard_inputs_places_a_and_w_on_op_specs() -> None:
    """The harness must hand the kernel inputs already laid out the way its
    shard_map expects — A over (dp, tp), W over (tp, replicated) — so no
    reshard happens on the timed path. Pin that shard_inputs produces exactly
    the NamedSharding input_specs declares for the same mesh."""
    mesh = make_mesh(2, 2)
    dp_axis, tp_axis = mesh.axis_names
    a, w = _inputs(jnp.bfloat16)
    a_s, w_s = shard_inputs(mesh, a, w)
    assert a_s.sharding == NamedSharding(mesh, P(dp_axis, tp_axis))
    assert w_s.sharding == NamedSharding(mesh, P(tp_axis, None))
    # input_specs is the contract shard_map consumes; shard_inputs must match.
    a_spec, w_spec = input_specs(mesh)
    assert a_s.sharding == NamedSharding(mesh, a_spec)
    assert w_s.sharding == NamedSharding(mesh, w_spec)


def test_output_spec_is_batch_and_feature_sharded() -> None:
    """Output is sharded on batch over dp and on the output feature dim over tp
    — the reduce-scatter keeps each chip its F-shard rather than replicating the
    full output. Pin it so a shard_map out_specs regression is caught here."""
    mesh = make_mesh(2, 2)
    dp_axis, tp_axis = mesh.axis_names
    assert output_spec(mesh) == P(dp_axis, tp_axis)
