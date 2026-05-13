"""Correctness tests for attention (no-cache, single-seq, fwd) variants."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_kernels.ops.attention import attention_naive, attention_xla

VARIANTS: dict[str, Callable[[jax.Array, jax.Array, jax.Array], jax.Array]] = {
    "xla": attention_xla,
}

# bf16 is the project default working dtype (CLAUDE.md). The bf16 atol is
# loose because the naive vs jax.nn.dot_product_attention pin compares two
# legitimate precision choices on length-T inner products: naive upcasts to
# f32 (matching the matmul-naive precedent and what MXU defaults give on
# TPU), while jax.nn on CPU stays bf16 throughout. The resulting per-element
# drift is small (~bf16 ulp scaled by sqrt(T)) and the pin still catches
# anything structural — wrong scale, missing transpose, wrong softmax axis,
# missing max-subtract. f32 stays tight.
DTYPES: dict[str, tuple[Any, dict[str, float]]] = {
    "bf16": (jnp.bfloat16, {"rtol": 1e-2, "atol": 5e-3}),
    "f32": (jnp.float32, {"rtol": 1e-5, "atol": 1e-6}),
}


def _inputs(dtype: Any, t: int = 64, n: int = 4, h: int = 32) -> tuple[jax.Array, ...]:
    keys = jax.random.split(jax.random.key(0), 3)
    return tuple(jax.random.normal(k, (t, n, h), dtype=dtype) for k in keys)


@pytest.mark.parametrize("dtype_name", list(DTYPES.keys()))
def test_naive_matches_jax_nn_dot_product_attention(dtype_name: str) -> None:
    """Pin the oracle against jax.nn.dot_product_attention. naive is what
    every variant is checked against; if naive itself drifted, the entire
    correctness suite would silently accept a wrong implementation.

    jax.nn expects a (B, T, N, H) layout, so wrap with a length-1 batch
    dim and unwrap on the way out. Default scale (None) is 1/sqrt(H),
    matching naive.
    """
    dtype, tol = DTYPES[dtype_name]
    q, k, v = _inputs(dtype)
    expected = jax.nn.dot_product_attention(q[None], k[None], v[None])[0]
    np.testing.assert_allclose(
        np.asarray(attention_naive(q, k, v)),
        np.asarray(expected),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )


@pytest.mark.parametrize("dtype_name", list(DTYPES.keys()))
@pytest.mark.parametrize("variant", VARIANTS.values(), ids=list(VARIANTS.keys()))
def test_matches_naive(
    variant: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    dtype_name: str,
) -> None:
    dtype, tol = DTYPES[dtype_name]
    q, k, v = _inputs(dtype)
    np.testing.assert_allclose(
        np.asarray(variant(q, k, v)),
        np.asarray(attention_naive(q, k, v)),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )


def test_large_logits_dont_overflow() -> None:
    """The max-subtract inside the softmax keeps exp() finite for large
    pre-softmax scores. A regression that drops the stable form (computes
    exp(s) / sum(exp(s)) directly) overflows and produces NaN."""
    t, n, h = 8, 2, 16
    big = jnp.full((t, n, h), 100.0, dtype=jnp.float32)
    out = attention_naive(big, big, big)
    assert jnp.all(jnp.isfinite(out)), "attention overflowed; softmax max-subtract is missing"
