"""Correctness tests for softmax variants."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_kernels.ops.softmax import softmax_naive, softmax_xla

VARIANTS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "xla": softmax_xla,
}

# bf16 is the project default working dtype (CLAUDE.md). It's also the only
# dtype where the f32 accumulator inside naive does work — at f32 the internal
# astype is a no-op, so an f32-only suite would silently accept a regression
# that dropped the f32 accumulation. f32 stays as a tight-tolerance check on
# the rest of the chain (max → sub → exp → div).
DTYPES: dict[str, tuple[Any, dict[str, float]]] = {
    "bf16": (jnp.bfloat16, {"rtol": 1e-2, "atol": 1e-3}),
    "f32": (jnp.float32, {"rtol": 1e-5, "atol": 1e-6}),
}


def _input(dtype: Any) -> jax.Array:
    return jax.random.normal(jax.random.key(0), (32, 256), dtype=dtype)


@pytest.mark.parametrize("dtype_name", list(DTYPES.keys()))
def test_naive_matches_jax_nn_softmax(dtype_name: str) -> None:
    """Pin the oracle against jax.nn.softmax. naive is what every variant is
    checked against; if naive itself drifted, the entire correctness suite
    would silently accept a wrong implementation."""
    dtype, tol = DTYPES[dtype_name]
    x = _input(dtype)
    np.testing.assert_allclose(
        np.asarray(softmax_naive(x)),
        np.asarray(jax.nn.softmax(x, axis=-1)),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )


@pytest.mark.parametrize("dtype_name", list(DTYPES.keys()))
@pytest.mark.parametrize("variant", VARIANTS.values(), ids=list(VARIANTS.keys()))
def test_matches_naive(
    variant: Callable[[jax.Array], jax.Array],
    dtype_name: str,
) -> None:
    dtype, tol = DTYPES[dtype_name]
    x = _input(dtype)
    np.testing.assert_allclose(
        np.asarray(variant(x)),
        np.asarray(softmax_naive(x)),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )


@pytest.mark.parametrize("dtype_name", list(DTYPES.keys()))
def test_uniform_input_yields_uniform_output(dtype_name: str) -> None:
    """All-equal logits map to 1/N along the last axis. Pins normalization
    independently of the naive-vs-jax.nn check (which would also be wrong
    if the divide step were wrong in the same way)."""
    dtype, _ = DTYPES[dtype_name]
    n = 256
    x = jnp.ones((4, n), dtype=dtype)
    y = softmax_naive(x)
    expected = jnp.full((4, n), 1.0 / n, dtype=dtype)
    np.testing.assert_allclose(np.asarray(y), np.asarray(expected), rtol=1e-2, atol=1e-3)


def test_large_logits_dont_overflow() -> None:
    """The max-subtract step is what keeps exp() finite for large inputs.
    A regression that drops it (i.e. computes exp(x) / sum(exp(x)) directly)
    overflows here and produces NaN rather than a valid distribution."""
    x = jnp.full((4, 16), 1000.0, dtype=jnp.float32)
    y = softmax_naive(x)
    assert jnp.all(jnp.isfinite(y)), "softmax overflowed; max-subtract is missing"
    np.testing.assert_allclose(np.asarray(y.sum(axis=-1)), 1.0, rtol=1e-5)
