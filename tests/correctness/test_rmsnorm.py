"""Correctness tests for rmsnorm variants."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_kernels.ops.rmsnorm import rmsnorm_naive, rmsnorm_pallas, rmsnorm_xla

VARIANTS: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = {
    "xla": rmsnorm_xla,
    # Two block sizes to cover both the single-block (grid of 1) and the
    # multi-block (grid of >1) lowering paths. Pallas runs on CPU via interpret=True.
    "pallas_b4": partial(rmsnorm_pallas, block_shape=(4,), interpret=True),
    "pallas_b32": partial(rmsnorm_pallas, block_shape=(32,), interpret=True),
}

# bf16 is the project default working dtype (CLAUDE.md). It's also the only
# dtype where the f32 accumulator inside naive does work — at f32 the internal
# astype is a no-op, so an f32-only suite would silently accept a regression
# that dropped the f32 accumulation. f32 stays as a tight-tolerance check on
# the rest of the chain (cast → rsqrt → scale multiply).
DTYPES: dict[str, tuple[Any, dict[str, float]]] = {
    "bf16": (jnp.bfloat16, {"rtol": 1e-2, "atol": 1e-3}),
    "f32": (jnp.float32, {"rtol": 1e-5, "atol": 1e-6}),
}


def _inputs(dtype: Any) -> tuple[jax.Array, jax.Array]:
    # Leading dim is 32 so the pallas variant can sweep block_size ∈ {4, 32}.
    x = jax.random.normal(jax.random.key(0), (32, 256), dtype=dtype)
    scale = jax.random.normal(jax.random.key(1), (256,), dtype=dtype)
    return x, scale


@pytest.mark.parametrize("dtype_name", list(DTYPES.keys()))
@pytest.mark.parametrize("variant", VARIANTS.values(), ids=list(VARIANTS.keys()))
def test_matches_naive(
    variant: Callable[[jax.Array, jax.Array], jax.Array],
    dtype_name: str,
) -> None:
    dtype, tol = DTYPES[dtype_name]
    x, scale = _inputs(dtype)
    np.testing.assert_allclose(
        np.asarray(variant(x, scale)),
        np.asarray(rmsnorm_naive(x, scale)),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )


@pytest.mark.parametrize("dtype_name", list(DTYPES.keys()))
def test_unit_variance_input_normalizes_to_scale(dtype_name: str) -> None:
    # If x already has rms ≈ 1 along the last axis, rmsnorm reduces to ``x * scale``.
    # This pins the math so a regression that mis-orders the chain (e.g. forgetting
    # to multiply by scale, or applying it before the cast) gets caught here, not
    # by allclose-vs-naive (which would also be wrong in the same way).
    dtype, _ = DTYPES[dtype_name]
    h = 256
    x = jnp.ones((4, h), dtype=dtype)
    scale = jax.random.normal(jax.random.key(2), (h,), dtype=dtype)
    y = rmsnorm_naive(x, scale, eps=0.0)
    np.testing.assert_allclose(np.asarray(y), np.broadcast_to(np.asarray(scale), y.shape))
