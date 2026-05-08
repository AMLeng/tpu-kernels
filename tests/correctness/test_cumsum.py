"""Correctness tests for cumsum variants.

xla and pallas have different shape constraints — xla falls back to a
single ``lax.cumsum`` for ``n < INNER`` and zero-pad-and-crops
unaligned ``n``; pallas requires ``n`` divisible by ``bm`` (no
fallback). Shared tests use ``n`` values divisible by all pallas
``bm`` choices; the xla-only zero-pad path has its own pin.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_kernels.ops.cumsum import cumsum_naive, cumsum_pallas, cumsum_xla

# pallas runs on CPU via interpret=True. Two block sizes to cover the
# kernel's two distinct paths: bm=128 → T=1 per tile (no within-tile
# Hillis-Steele, only cross-tile carry); bm=512 → T=4 per tile
# (log2(4)=2 Hillis-Steele iterations on the row totals before the
# cross-tile carry).
VARIANTS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "xla": cumsum_xla,
    "pallas_b128": partial(cumsum_pallas, block_shape=(128,), interpret=True),
    "pallas_b512": partial(cumsum_pallas, block_shape=(512,), interpret=True),
}


@pytest.mark.parametrize("variant", VARIANTS.values(), ids=list(VARIANTS.keys()))
@pytest.mark.parametrize(
    "n",
    [
        # < xla's INNER=4096 → hits xla's lax.cumsum fallback. Multiple
        # of both pallas bms (128, 512): 8 tiles of T=1 / 2 tiles of T=4.
        1024,
        # 2x INNER, multiple of both pallas bms. Exercises xla's
        # multi-tile cumsum-of-totals path with no zero-pad remainder.
        8192,
    ],
    ids=lambda n: f"n={n}",
)
def test_matches_naive(
    n: int,
    variant: Callable[[jax.Array], jax.Array],
) -> None:
    x = jax.random.normal(jax.random.key(0), (n,), dtype=jnp.float32)
    # Reduction order differs from naive's linear scan, so f32 sums
    # diverge at noise level (~sqrt(N) ulp). atol absorbs near-zero
    # crossings where rtol blows up.
    np.testing.assert_allclose(
        np.asarray(variant(x)),
        np.asarray(cumsum_naive(x)),
        atol=1e-4,
        rtol=1e-4,
    )


def test_xla_handles_unaligned_remainder() -> None:
    """xla's zero-pad-and-crop path: n not a multiple of INNER. pallas
    requires divisibility, so this case is xla-only."""
    n = 8192 + 17  # 2*INNER + an unaligned remainder
    x = jax.random.normal(jax.random.key(0), (n,), dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(cumsum_xla(x)),
        np.asarray(cumsum_naive(x)),
        atol=1e-4,
        rtol=1e-4,
    )


def test_pallas_recursive_mxu_path() -> None:
    """bm=16384 → T=128, B=1: exercises the within-block recursive-MXU
    path (the T>=128 branch). The shared parametrization above uses
    bm in {128, 512} (T<128) so this branch is otherwise unexercised
    on CPU."""
    n = 2 * 16384
    x = jax.random.normal(jax.random.key(0), (n,), dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(cumsum_pallas(x, block_shape=(16384,), interpret=True)),
        np.asarray(cumsum_naive(x)),
        atol=1e-3,
        rtol=1e-3,
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
