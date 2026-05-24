"""Correctness tests for sharded_matmul variants.

Runs on any host that exposes >=4 devices (enough for the (dp, tp) combos
below). The cheap way to satisfy that on CPU is
`XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run pytest
tests/correctness/test_sharded_matmul.py`; the CI default of 1 CPU device
skips the module cleanly via the marker below.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_kernels.ops.sharded_matmul import sharded_matmul_naive, sharded_matmul_xla

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


@pytest.mark.parametrize("mesh", MESHES, ids=[f"dp{dp}_tp{tp}" for dp, tp in MESHES])
@pytest.mark.parametrize("dtype_name", list(DTYPES.keys()))
def test_naive_matches_unsharded_matmul(dtype_name: str, mesh: tuple[int, int]) -> None:
    """Pin the oracle against the unsharded f32-accumulated reference across
    mesh shapes. A regression that broke the all_reduce (dropped the psum,
    reduced over the wrong axis, or psummed the wrong dtype) would land here
    as a tp-fold error on every output element."""
    dp, tp = mesh
    dtype, tol = DTYPES[dtype_name]
    a, w = _inputs(dtype)
    ref = (a.astype(jnp.float32) @ w.astype(jnp.float32)).astype(dtype)
    np.testing.assert_allclose(
        np.asarray(sharded_matmul_naive(a, w, dp=dp, tp=tp)),
        np.asarray(ref),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )


@pytest.mark.parametrize("mesh", MESHES, ids=[f"dp{dp}_tp{tp}" for dp, tp in MESHES])
@pytest.mark.parametrize("dtype_name", list(DTYPES.keys()))
@pytest.mark.parametrize("variant", VARIANTS.values(), ids=list(VARIANTS.keys()))
def test_matches_naive(
    variant: Callable[..., jax.Array],
    dtype_name: str,
    mesh: tuple[int, int],
) -> None:
    dp, tp = mesh
    dtype, tol = DTYPES[dtype_name]
    a, w = _inputs(dtype)
    np.testing.assert_allclose(
        np.asarray(partial(variant, dp=dp, tp=tp)(a, w)),
        np.asarray(sharded_matmul_naive(a, w, dp=dp, tp=tp)),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )
