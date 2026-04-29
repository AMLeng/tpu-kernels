"""Correctness tests for scale variants. Pallas paths run on CPU via interpret=True."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_kernels.ops.scale import scale_naive, scale_pallas, scale_xla

VARIANTS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "xla": scale_xla,
    "pallas_64x64": partial(scale_pallas, block_shape=(64, 64), interpret=True),
    "pallas_128x128": partial(scale_pallas, block_shape=(128, 128), interpret=True),
    "pallas_256x256": partial(scale_pallas, block_shape=(256, 256), interpret=True),
}


@pytest.fixture
def x() -> jax.Array:
    return jax.random.normal(jax.random.key(0), (256, 256), dtype=jnp.float32)


@pytest.mark.parametrize("variant", VARIANTS.values(), ids=list(VARIANTS.keys()))
def test_matches_naive(x: jax.Array, variant: Callable[[jax.Array], jax.Array]) -> None:
    np.testing.assert_allclose(np.asarray(variant(x)), np.asarray(scale_naive(x)))


def test_pallas_rejects_wrong_ndim() -> None:
    x_3d = jnp.zeros((2, 4, 8), dtype=jnp.float32)
    with pytest.raises(ValueError, match="2-D"):
        scale_pallas(x_3d, interpret=True)


def test_pallas_rejects_unaligned_shape() -> None:
    x_misaligned = jnp.zeros((100, 100), dtype=jnp.float32)
    with pytest.raises(ValueError, match="not divisible"):
        scale_pallas(x_misaligned, block_shape=(128, 128), interpret=True)
