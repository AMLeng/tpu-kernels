"""Performance regression test for rmsnorm. TPU-only.

Threshold-gated: fails if ``rmsnorm_pallas`` hits less than
``BW_UTIL_FLOOR`` of HBM bandwidth on the canonical (8192, 8192) bf16
input. Skipped on CPU-only runs via the ``tpu`` marker.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from benchmarks.roofline import analyze, v5e
from benchmarks.runner import bench

from tpu_kernels.ops.rmsnorm import rmsnorm_pallas
from tpu_kernels.ops.rmsnorm.pallas import DEFAULT_BLOCK

pytestmark = [pytest.mark.tpu, pytest.mark.perf]

# Regression floor against the current Pallas kernel under timing="device"
# (kernel clock read directly from the TPU XPlane). PERF.md records 80.2%
# on (8192, 8192) bf16 with bm=DEFAULT_BLOCK (128) on v5e. Floor pinned at
# the target — both 0.80 — so a below-floor reading is also a missed target.
# Sibling to tests/perf/test_scale.py.
BW_UTIL_FLOOR = 0.80

_BS = 8192
_H = 8192


def _inputs() -> tuple[jax.Array, jax.Array]:
    dtype = jnp.bfloat16
    x = jax.random.normal(jax.random.key(0), (_BS, _H), dtype=dtype)
    scale = jax.random.normal(jax.random.key(1), (_H,), dtype=dtype)
    return x, scale


def _bytes_and_flops() -> tuple[int, int]:
    bytes_per_elem = jnp.dtype(jnp.bfloat16).itemsize
    nbytes = 2 * _BS * _H * bytes_per_elem + _H * bytes_per_elem
    flops = 4 * _BS * _H
    return nbytes, flops


def test_rmsnorm_pallas_hits_target_bw() -> None:
    x, scale = _inputs()

    @jax.jit
    def fn(y: jax.Array, s: jax.Array) -> jax.Array:
        return rmsnorm_pallas(y, s, block_size=DEFAULT_BLOCK)

    result = bench(f"rmsnorm::pallas_b{DEFAULT_BLOCK}", fn, args=(x, scale), timing="device")
    nbytes, flops = _bytes_and_flops()
    roof = analyze(flops=flops, nbytes=nbytes, seconds=result.median_s, hw=v5e())
    assert roof.bw_util >= BW_UTIL_FLOOR, (
        f"rmsnorm_pallas hit {roof.bw_util:.1%} HBM BW, below floor {BW_UTIL_FLOOR:.0%}"
    )


def test_rmsnorm_pallas_device_timing_higher_bw_than_unroll() -> None:
    """device-clock numbers must come out higher than unroll-amortized
    numbers for the same kernel: unroll still carries residual host-side
    overhead even after amortization, while device reads the TPU hardware
    clock directly. If the two are equal, the parse path isn't actually
    reading device time.

    Also bounds device BW% below 110% — catches a ns-vs-µs units bug that
    would otherwise go silent — and asserts the cluster gap heuristic
    split iters cleanly. Mirror of the scale-side check.
    """
    x, scale = _inputs()

    @jax.jit
    def fn(y: jax.Array, s: jax.Array) -> jax.Array:
        return rmsnorm_pallas(y, s, block_size=DEFAULT_BLOCK)

    unroll = bench(f"rmsnorm::pallas_b{DEFAULT_BLOCK}_unroll", fn, args=(x, scale), timing="unroll")
    device = bench(f"rmsnorm::pallas_b{DEFAULT_BLOCK}_device", fn, args=(x, scale), timing="device")

    assert device.timing == "device"
    assert device.unroll == 1
    assert not device.cluster_mismatch, (
        "XPlane parse split timed iters into a different number of clusters "
        "than expected; gap_ns may need bumping for this kernel."
    )

    nbytes, flops = _bytes_and_flops()
    unroll_bw = analyze(flops=flops, nbytes=nbytes, seconds=unroll.median_s, hw=v5e()).bw_util
    device_bw = analyze(flops=flops, nbytes=nbytes, seconds=device.median_s, hw=v5e()).bw_util

    assert device_bw <= 1.10, (
        f"device-timed BW% {device_bw:.1%} > 110%; suspect a units bug in the XPlane parse"
    )
    assert device_bw > unroll_bw, (
        f"device BW% ({device_bw:.1%}) should exceed unroll BW% ({unroll_bw:.1%}) "
        "— device reads kernel time directly off the TPU clock, unroll carries "
        "residual host overhead."
    )
