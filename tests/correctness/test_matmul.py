"""Correctness tests for matmul variants."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_kernels.ops.matmul import matmul_naive, matmul_pallas, matmul_xla

# Block cubes that divide both narrow (M=64, K=256, N=64) and wide-K bf16
# (M=8, K=8192, N=8) shapes used below — the cubes exercise single-grid-step
# (bm=M) and multi-step grid lowerings across the two tests. Pallas runs on
# CPU via interpret=True.
VARIANTS: dict[str, Callable[..., jax.Array]] = {
    "xla": matmul_xla,
    "pallas_b16x16x32": partial(matmul_pallas, block_shape=(16, 16, 32), interpret=True),
    "pallas_b32x32x64": partial(matmul_pallas, block_shape=(32, 32, 64), interpret=True),
}

# bf16 is the project default working dtype (CLAUDE.md). It's also the only
# dtype where the f32 K-accumulator inside naive does work — at f32 the
# internal astype is a no-op, so an f32-only suite would silently accept a
# regression that dropped the f32 accumulation. f32 stays as a tight-tolerance
# check on the rest of the chain (dot + cast); 5e-5 leaves room for
# reduction-tree differences between naive's single length-K dot and a
# Pallas variant's K/bk partial sums (worst-case forward error scales with
# K·eps_f32 ≈ 3e-5 at K=256), which differ in the last few ulps depending
# on how the host LLVM build orders the summation.
DTYPES: dict[str, tuple[Any, dict[str, float]]] = {
    "bf16": (jnp.bfloat16, {"rtol": 2e-2, "atol": 1e-2}),
    "f32": (jnp.float32, {"rtol": 5e-5, "atol": 5e-5}),
}


def _inputs(dtype: Any, m: int = 64, k: int = 256, n: int = 64) -> tuple[jax.Array, jax.Array]:
    a = jax.random.normal(jax.random.key(0), (m, k), dtype=dtype)
    b = jax.random.normal(jax.random.key(1), (k, n), dtype=dtype)
    return a, b


@pytest.mark.parametrize("dtype_name", list(DTYPES.keys()))
def test_naive_matches_jnp_matmul_in_f32(dtype_name: str) -> None:
    """Pin the oracle against ``jnp.matmul`` lifted to f32. naive is what
    every variant is checked against; if naive itself drifted (e.g. someone
    dropped the ``.astype(jnp.float32)`` and silently went bf16-acc), the
    entire correctness suite would silently accept a wrong implementation."""
    dtype, tol = DTYPES[dtype_name]
    a, b = _inputs(dtype)
    ref = jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32)).astype(dtype)
    np.testing.assert_allclose(
        np.asarray(matmul_naive(a, b)),
        np.asarray(ref),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )


@pytest.mark.parametrize("dtype_name", list(DTYPES.keys()))
@pytest.mark.parametrize("variant", VARIANTS.values(), ids=list(VARIANTS.keys()))
def test_matches_naive(
    variant: Callable[..., jax.Array],
    dtype_name: str,
) -> None:
    dtype, tol = DTYPES[dtype_name]
    a, b = _inputs(dtype)
    np.testing.assert_allclose(
        np.asarray(variant(a, b)),
        np.asarray(matmul_naive(a, b)),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )


def test_pallas_matches_naive_wide_k_bf16() -> None:
    """Wide-K bf16 case for the f32-accumulator regression. The narrow
    (64, 256, 64) test_matches_naive shape is too short for a bf16 K-sum
    to drift past rtol — at K=256 the partial sum stays well-resolved
    in bf16, so a variant that accumulated in bf16 instead of f32 would
    still pass. K=8192 is a realistic transformer hidden dim and reliably
    trips the failure mode: the partial sum grows past the bf16 ulp at
    which O(1/N) terms round away."""
    dtype, tol = DTYPES["bf16"]
    a, b = _inputs(dtype, m=8, k=8192, n=8)
    np.testing.assert_allclose(
        np.asarray(matmul_pallas(a, b, block_shape=(8, 8, 128), interpret=True)),
        np.asarray(matmul_naive(a, b)),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )


def test_identity_left_returns_b() -> None:
    """``I @ b == b`` — pins basic semantics independently of the
    naive-vs-jnp.matmul check, which would also be wrong if the matmul
    op were transposed (returning ``b.T @ a.T`` etc.)."""
    b = jax.random.normal(jax.random.key(0), (32, 16), dtype=jnp.float32)
    eye = jnp.eye(32, dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(matmul_naive(eye, b)), np.asarray(b), rtol=1e-5)


def test_identity_right_returns_a() -> None:
    """``a @ I == a`` — symmetric pin to test_identity_left_returns_b."""
    a = jax.random.normal(jax.random.key(0), (16, 32), dtype=jnp.float32)
    eye = jnp.eye(32, dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(matmul_naive(a, eye)), np.asarray(a), rtol=1e-5)


def test_pallas_rejects_wrong_ndim() -> None:
    a_3d = jnp.zeros((2, 4, 8), dtype=jnp.float32)
    b = jnp.zeros((8, 4), dtype=jnp.float32)
    with pytest.raises(ValueError, match="2-D"):
        matmul_pallas(a_3d, b, interpret=True)


def test_pallas_rejects_shape_mismatch() -> None:
    a = jnp.zeros((32, 64), dtype=jnp.float32)
    b = jnp.zeros((128, 32), dtype=jnp.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        matmul_pallas(a, b, interpret=True)


def test_pallas_rejects_wrong_block_ndim() -> None:
    a = jnp.zeros((32, 64), dtype=jnp.float32)
    b = jnp.zeros((64, 32), dtype=jnp.float32)
    with pytest.raises(ValueError, match="3-axis block_shape"):
        matmul_pallas(a, b, block_shape=(16, 16), interpret=True)


def test_pallas_rejects_unaligned_shape() -> None:
    a = jnp.zeros((30, 64), dtype=jnp.float32)
    b = jnp.zeros((64, 32), dtype=jnp.float32)
    with pytest.raises(ValueError, match="not divisible"):
        matmul_pallas(a, b, block_shape=(16, 16, 32), interpret=True)
