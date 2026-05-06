"""Correctness tests for segment_cumsum variants.

Pallas variant pending (A.6 in `docs/curriculum.md`); for now VARIANTS
holds only xla. When the Pallas variant lands it joins the parametrize
list with `interpret=True` so it runs on CPU here.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_kernels.ops.segment_cumsum import (
    segment_cumsum_naive,
    segment_cumsum_xla,
)

VARIANTS: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = {
    "xla": segment_cumsum_xla,
}


@pytest.fixture
def x() -> jax.Array:
    return jax.random.normal(jax.random.key(0), (256,), dtype=jnp.float32)


@pytest.fixture
def segment_ids() -> jax.Array:
    # 8 equal-sized segments of 32 elements each; monotonically non-decreasing
    # so the segmented combiner is associative on this input.
    return jnp.repeat(jnp.arange(8, dtype=jnp.int32), 32)


@pytest.mark.parametrize("variant", VARIANTS.values(), ids=list(VARIANTS.keys()))
def test_matches_naive(
    x: jax.Array,
    segment_ids: jax.Array,
    variant: Callable[[jax.Array, jax.Array], jax.Array],
) -> None:
    np.testing.assert_allclose(
        np.asarray(variant(x, segment_ids)),
        np.asarray(segment_cumsum_naive(x, segment_ids)),
    )


def test_naive_returns_segmented_cumsum() -> None:
    """Pin the oracle independently of any variant. If naive itself
    drifted, every variant test would still pass against the wrong reference."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float32)
    segment_ids = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
    expected = jnp.array([1.0, 3.0, 6.0, 4.0, 9.0, 15.0], dtype=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(segment_cumsum_naive(x, segment_ids)),
        np.asarray(expected),
    )


def test_naive_handles_uneven_segments() -> None:
    """Segment widths needn't be equal — pin a case with a 1-element
    segment and a 4-element segment in the same input."""
    x = jnp.array([10.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    segment_ids = jnp.array([0, 1, 1, 1, 1], dtype=jnp.int32)
    expected = jnp.array([10.0, 1.0, 3.0, 6.0, 10.0], dtype=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(segment_cumsum_naive(x, segment_ids)),
        np.asarray(expected),
    )
