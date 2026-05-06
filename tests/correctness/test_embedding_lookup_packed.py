"""Correctness tests for embedding_lookup_packed variants on the 4-D
packed layout.

The packed shape ``(vocab, hidden//1024, 8, 128)`` is byte-equivalent
to the 1-D ``T(1024)(128)(2, 1)`` form. Tests use small shapes that
satisfy the divisibility constraint and run Pallas under
``interpret=True`` so they pass on CPU.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_kernels.ops.embedding_lookup_packed import (
    embedding_lookup_packed_naive,
    embedding_lookup_packed_pallas,
    embedding_lookup_packed_xla,
)

VARIANTS: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = {
    "xla": embedding_lookup_packed_xla,
    "pallas": partial(embedding_lookup_packed_pallas, block_shape=(8,), interpret=True),
}


@pytest.fixture
def params() -> jax.Array:
    # Packed shape: (vocab=64, hidden//1024=1, 8, 128) — the smallest 4-D
    # shape that satisfies the layout's constraints.
    return jax.random.normal(jax.random.key(0), (64, 1, 8, 128), dtype=jnp.bfloat16)


@pytest.fixture
def ids() -> jax.Array:
    return jax.random.randint(jax.random.key(1), (32,), 0, 64, dtype=jnp.int32)


@pytest.mark.parametrize("variant", VARIANTS.values(), ids=list(VARIANTS.keys()))
def test_matches_naive(
    params: jax.Array,
    ids: jax.Array,
    variant: Callable[[jax.Array, jax.Array], jax.Array],
) -> None:
    np.testing.assert_array_equal(
        np.asarray(variant(params, ids)),
        np.asarray(embedding_lookup_packed_naive(params, ids)),
    )


def test_naive_returns_correct_rows() -> None:
    """Pin the oracle independently of any variant — same rationale as
    the natural-layout op's pinned naive test."""
    params = jnp.arange(40.0, dtype=jnp.float32).reshape(5, 1, 8, 1)
    ids = jnp.array([0, 2, 4, 1], dtype=jnp.int32)
    expected = jnp.stack([params[0], params[2], params[4], params[1]])
    np.testing.assert_array_equal(
        np.asarray(embedding_lookup_packed_naive(params, ids)),
        np.asarray(expected),
    )
