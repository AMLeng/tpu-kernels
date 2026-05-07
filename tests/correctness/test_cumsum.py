"""Correctness tests for cumsum variants.

cumsum is the no-segment-reset stepping-stone for segment_cumsum (A.6);
no Pallas variant exists, so VARIANTS holds only xla.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_kernels.ops.cumsum import cumsum_naive, cumsum_xla

VARIANTS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "xla": cumsum_xla,
}


@pytest.mark.parametrize("variant", VARIANTS.values(), ids=list(VARIANTS.keys()))
@pytest.mark.parametrize(
    "n",
    [
        # < xla's INNER=4096: hits the single-call `lax.cumsum` fallback.
        256,
        # 2x INNER + an unaligned remainder: exercises the (T, INNER)
        # reshape, the L2 cumsum-of-totals path, and the zero-pad-and-crop.
        8192 + 17,
    ],
    ids=lambda n: f"n={n}",
)
def test_matches_naive(
    n: int,
    variant: Callable[[jax.Array], jax.Array],
) -> None:
    x = jax.random.normal(jax.random.key(0), (n,), dtype=jnp.float32)
    # The hierarchy accumulates in a different order than naive's linear
    # scan, so f32 sums diverge at noise level (~sqrt(N) ulp). atol absorbs
    # near-zero crossings where rtol blows up.
    np.testing.assert_allclose(
        np.asarray(variant(x)),
        np.asarray(cumsum_naive(x)),
        atol=1e-4,
        rtol=1e-4,
    )


def test_naive_returns_prefix_sum() -> None:
    """Pin the oracle independently of any variant. If naive itself
    drifted, every variant test would still pass against the wrong reference."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    expected = jnp.array([1.0, 3.0, 6.0, 10.0], dtype=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(cumsum_naive(x)),
        np.asarray(expected),
    )
