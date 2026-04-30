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

# Regression floor against the current naive single-buffered kernel,
# which plateaus at ~63% HBM BW across every sane (bm, bn) on v5e (see
# bench_history/scale/). 0.60 leaves ~3pp headroom for run-to-run noise.
# Raise this once a pipelined / multi-buffered variant lands and pushes
# Current up.
BW_UTIL_FLOOR = 0.60


def test_scale_pallas_hits_target_bw() -> None:
    m, n = 8192, 8192
    dtype = jnp.bfloat16
    bytes_per_elem = jnp.dtype(dtype).itemsize
    x = jax.random.normal(jax.random.key(0), (m, n), dtype=dtype)

    @jax.jit
    def fn(y: jax.Array) -> jax.Array:
        return scale_pallas(y, block_shape=(512, 1024))

    result = bench("scale::pallas_512x1024", fn, args=(x,))
    roof = analyze(
        flops=m * n,
        nbytes=2 * m * n * bytes_per_elem,  # read + write
        seconds=result.median_s,
        hw=v5e(),
    )
    assert roof.bw_util >= BW_UTIL_FLOOR, (
        f"scale_pallas hit {roof.bw_util:.1%} HBM BW, below floor {BW_UTIL_FLOOR:.0%}"
    )


def test_scale_pallas_device_timing_higher_bw_than_unroll() -> None:
    """device-clock numbers must come out higher than unroll-amortized
    numbers for the same kernel: unroll still carries residual host-side
    overhead even after amortization, while device reads the TPU
    hardware clock directly. If the two are equal, the parse path isn't
    actually reading device time.

    Also bounds device BW% below 110% — catches a ns-vs-µs units bug
    that would otherwise go silent — and asserts the cluster gap
    heuristic split iters cleanly.
    """
    m, n = 8192, 8192
    dtype = jnp.bfloat16
    bytes_per_elem = jnp.dtype(dtype).itemsize
    x = jax.random.normal(jax.random.key(0), (m, n), dtype=dtype)

    @jax.jit
    def fn(y: jax.Array) -> jax.Array:
        return scale_pallas(y, block_shape=(512, 1024))

    unroll = bench("scale::pallas_512x1024_unroll", fn, args=(x,), timing="unroll")
    device = bench("scale::pallas_512x1024_device", fn, args=(x,), timing="device")

    assert device.timing == "device"
    assert device.unroll == 1
    assert not device.cluster_mismatch, (
        "XPlane parse split timed iters into a different number of clusters "
        "than expected; gap_ns may need bumping for this kernel."
    )

    nbytes = 2 * m * n * bytes_per_elem
    unroll_bw = analyze(flops=m * n, nbytes=nbytes, seconds=unroll.median_s, hw=v5e()).bw_util
    device_bw = analyze(flops=m * n, nbytes=nbytes, seconds=device.median_s, hw=v5e()).bw_util

    assert device_bw <= 1.10, (
        f"device-timed BW% {device_bw:.1%} > 110%; suspect a units bug in the XPlane parse"
    )
    assert device_bw > unroll_bw, (
        f"device BW% ({device_bw:.1%}) should exceed unroll BW% ({unroll_bw:.1%}) "
        "— device reads kernel time directly off the TPU clock, unroll carries "
        "residual host overhead."
    )
