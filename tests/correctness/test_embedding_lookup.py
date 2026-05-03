"""Correctness tests for embedding_lookup variants.

Pallas variant pending (A.5 in `docs/curriculum.md`); for now VARIANTS
holds only the xla path. When the Pallas variant lands it joins the
parametrize list with `interpret=True` so it runs on CPU here.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_kernels.ops.embedding_lookup import (
    embedding_lookup_naive,
    embedding_lookup_xla,
)

VARIANTS: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = {
    "xla": embedding_lookup_xla,
}


@pytest.fixture
def params() -> jax.Array:
    return jax.random.normal(jax.random.key(0), (1024, 256), dtype=jnp.float32)


@pytest.fixture
def ids() -> jax.Array:
    return jax.random.randint(jax.random.key(1), (128,), 0, 1024, dtype=jnp.int32)


@pytest.mark.parametrize("variant", VARIANTS.values(), ids=list(VARIANTS.keys()))
def test_matches_naive(
    params: jax.Array,
    ids: jax.Array,
    variant: Callable[[jax.Array, jax.Array], jax.Array],
) -> None:
    np.testing.assert_allclose(
        np.asarray(variant(params, ids)),
        np.asarray(embedding_lookup_naive(params, ids)),
    )


def test_naive_returns_correct_rows() -> None:
    """Pin the oracle independently of any variant. If naive itself drifted
    (e.g. from `params[ids]` to `params[:, ids]` or a transposed gather),
    every variant comparison would silently agree with a wrong oracle.
    """
    params = jnp.arange(20.0, dtype=jnp.float32).reshape(5, 4)
    ids = jnp.array([0, 2, 4, 1], dtype=jnp.int32)
    expected = jnp.stack([params[0], params[2], params[4], params[1]])
    np.testing.assert_array_equal(
        np.asarray(embedding_lookup_naive(params, ids)),
        np.asarray(expected),
    )
