"""Correctness tests for segment_cumsum variants.

Two pallas block sizes exercise the seg-HS structure: bm=128 covers
T=1 (no cross-row scan, only the cross-tile carry path); bm=512 covers
T=4 (small cross-row Hillis-Steele). Both run under interpret=True on
CPU. Larger block sizes share the same kernel logic — they only differ
in Mosaic-lowered tile heights, which interpret=True bypasses.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_kernels.ops.segment_cumsum import (
    segment_cumsum_naive,
    segment_cumsum_pallas,
    segment_cumsum_xla,
)

# `interpret=True` evaluates the kernel via standard `jnp.*` ops, but
# those still dispatch to whatever the default device is. On a TPU box
# the matmul then takes the MXU's bf16-input default and drifts beyond
# this test's f32 tolerance, so we pin to CPU explicitly. CPU-only CI
# already lands here naturally.
_CPU_DEVICE = jax.devices("cpu")[0]


def _on_cpu(fn: Callable[..., jax.Array]) -> Callable[..., jax.Array]:
    def wrapper(*args, **kwargs):
        with jax.default_device(_CPU_DEVICE):
            return fn(*args, **kwargs)

    return wrapper


VARIANTS: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = {
    "xla": segment_cumsum_xla,
    "pallas_b128": _on_cpu(partial(segment_cumsum_pallas, block_shape=(128,), interpret=True)),
    "pallas_b512": _on_cpu(partial(segment_cumsum_pallas, block_shape=(512,), interpret=True)),
}


@pytest.mark.parametrize("variant", VARIANTS.values(), ids=list(VARIANTS.keys()))
@pytest.mark.parametrize(
    "n,segment_length",
    [
        # Tile-aligned boundaries: at bm=128 every tile is exactly one
        # segment, so every tile boundary is a segment boundary —
        # stresses the cross-tile carry's "new segment" path. At bm=512,
        # T=4, each tile contains 4 segments — every row a fresh segment.
        (1024, 128),
        # Many within-row boundaries: 32-element segments mean 4
        # boundaries inside every 128-lane row. Exercises the within-row
        # PFB scan past the trivial "one boundary per row" case.
        (1024, 32),
        # Larger n exercises xla's full hierarchy (n >= INNER=4096) and
        # multiple cross-tile carries on pallas.
        (8192, 256),
    ],
    ids=["n1024_seg128", "n1024_seg32", "n8192_seg256"],
)
def test_matches_naive(
    n: int,
    segment_length: int,
    variant: Callable[[jax.Array, jax.Array], jax.Array],
) -> None:
    x = jax.random.normal(jax.random.key(0), (n,), dtype=jnp.float32)
    num_segments = n // segment_length
    segment_ids = jnp.repeat(jnp.arange(num_segments, dtype=jnp.int32), segment_length)
    # Reduction order differs from naive's linear scan, so f32 sums
    # diverge at noise level (~sqrt(N) ulp). atol absorbs near-zero
    # crossings where rtol blows up.
    np.testing.assert_allclose(
        np.asarray(variant(x, segment_ids)),
        np.asarray(segment_cumsum_naive(x, segment_ids)),
        atol=1e-4,
        rtol=1e-4,
    )


def test_pallas_handles_single_segment() -> None:
    """Single segment spanning the whole input — every lane and every
    cross-tile carry sees the same sid, so the seg-HS masked-add
    fires unconditionally and the entering-sid match against the carry
    is always true. Exercises the cross-tile continuation path."""
    n = 1024
    x = jax.random.normal(jax.random.key(1), (n,), dtype=jnp.float32)
    segment_ids = jnp.zeros((n,), dtype=jnp.int32)
    np.testing.assert_allclose(
        np.asarray(segment_cumsum_pallas(x, segment_ids, block_shape=(128,), interpret=True)),
        np.asarray(segment_cumsum_naive(x, segment_ids)),
        atol=1e-4,
        rtol=1e-4,
    )


def test_pallas_handles_uneven_segments() -> None:
    """Segment widths needn't be equal — pin a case with a 1-element
    segment next to a long one. Tests the seg-HS scan against an
    irregular boundary pattern, not just power-of-two strides."""
    n = 128
    # 1, 7, 100, 20 element segments
    segment_ids = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.ones((7,), dtype=jnp.int32),
            jnp.full((100,), 2, dtype=jnp.int32),
            jnp.full((20,), 3, dtype=jnp.int32),
        ]
    )
    x = jax.random.normal(jax.random.key(2), (n,), dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(segment_cumsum_pallas(x, segment_ids, block_shape=(128,), interpret=True)),
        np.asarray(segment_cumsum_naive(x, segment_ids)),
        atol=1e-4,
        rtol=1e-4,
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
