"""Performance regression test for scale. TPU-only.

Threshold-gated: fails if `scale_pallas` hits less than `BW_UTIL_FLOOR`
of HBM bandwidth on the canonical (8192, 8192) bf16 input. Skipped on
CPU-only runs via the `tpu` marker.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from benchmarks.roofline import analyze, v5e
from benchmarks.runner import bench

from tpu_kernels.ops.scale import scale_pallas

pytestmark = [pytest.mark.tpu, pytest.mark.perf]

BW_UTIL_FLOOR = 0.85  # ≥ 85% HBM BW per ops/scale/PERF.md


def test_scale_pallas_hits_target_bw() -> None:
    m, n = 8192, 8192
    dtype = jnp.bfloat16
    bytes_per_elem = jnp.dtype(dtype).itemsize
    x = jax.random.normal(jax.random.key(0), (m, n), dtype=dtype)

    @jax.jit
    def fn(y: jax.Array) -> jax.Array:
        return scale_pallas(y, block_shape=(256, 256))

    result = bench("scale::pallas_256x256", fn, args=(x,))
    roof = analyze(
        flops=m * n,
        nbytes=2 * m * n * bytes_per_elem,  # read + write
        seconds=result.median_s,
        hw=v5e(),
    )
    assert roof.bw_util >= BW_UTIL_FLOOR, (
        f"scale_pallas hit {roof.bw_util:.1%} HBM BW, below floor {BW_UTIL_FLOOR:.0%}"
    )
