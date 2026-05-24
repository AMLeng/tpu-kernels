"""Unit tests for benchmarks/roofline.py — pure logic, no JAX needed.

The TPU-marked test at the bottom is the exception: it cross-checks our
hardcoded constants against ``pltpu.get_tpu_info()`` so any drift between
this file and JAX's own canonical numbers shows up loudly. CPU runs
auto-skip it via conftest.
"""

from __future__ import annotations

from typing import Any

import pytest
from benchmarks.roofline import (
    V5E_BF16_PEAK_FLOPS,
    V5E_F32_PEAK_FLOPS,
    V5E_HBM_BANDWIDTH,
    V5E_ICI_LINKS_PER_CHIP,
    V5E_ICI_PER_LINK,
    V5E_INT8_PEAK_OPS,
    V5E_SMEM_CAPACITY,
    V5E_VMEM_CAPACITY,
    HardwarePeak,
    analyze,
    binding_regime,
    check_supported_hardware,
    peak_flops_for,
    regime,
    v5e,
)


def test_v5e_constants_match_announcement() -> None:
    hw = v5e()
    assert hw.kind == "v5e"
    assert hw.bf16_flops == V5E_BF16_PEAK_FLOPS == 197e12
    assert hw.f32_flops == V5E_F32_PEAK_FLOPS == 98e12
    assert hw.int8_ops == V5E_INT8_PEAK_OPS == 393e12
    assert hw.hbm_bw == V5E_HBM_BANDWIDTH == 819e9
    # Bidirectional per-link ICI: 90 GB/s (45 GB/s/direction), per the jax-ml
    # scaling book. 4 links → 360 GB/s/chip bidirectional aggregate.
    assert hw.ici_bw_per_link == V5E_ICI_PER_LINK == 90e9
    assert hw.ici_links_per_chip == V5E_ICI_LINKS_PER_CHIP == 4
    assert hw.num_chips == 1


def test_v5e_aggregate_ici_is_well_below_hbm() -> None:
    """Regression: ICI was mislabelled at 200 GB/s *per link* (x4 -> 800 GB/s/chip),
    making the ICI roof rival HBM. The real per-chip aggregate is 360 GB/s
    bidirectional — well under half the 819 GB/s HBM, so cross-chip traffic
    binds far sooner than HBM does. Pin that ordering so the double-count
    can't return.
    """
    hw = v5e()
    assert hw.total_ici_bw == 360e9
    assert hw.total_ici_bw < hw.total_hbm_bw / 2


def test_v5e_scales_with_num_chips() -> None:
    hw = v5e(num_chips=8)
    assert hw.total_bf16_flops == V5E_BF16_PEAK_FLOPS * 8
    assert hw.total_f32_flops == V5E_F32_PEAK_FLOPS * 8
    assert hw.total_hbm_bw == V5E_HBM_BANDWIDTH * 8
    assert hw.total_int8_ops == V5E_INT8_PEAK_OPS * 8
    assert hw.total_ici_bw == V5E_ICI_PER_LINK * V5E_ICI_LINKS_PER_CHIP * 8


def test_compute_bound_kernel_binds_compute() -> None:
    # Intensity 1000 F/B, well above v5e ridge ~240 → compute-bound.
    flops = 10**9
    nbytes = 10**6
    seconds = flops / V5E_BF16_PEAK_FLOPS
    roof = analyze(flops=flops, nbytes=nbytes, seconds=seconds)
    assert roof.binds == "compute"
    assert roof.mfu == pytest.approx(1.0)


def test_memory_bound_kernel_binds_memory() -> None:
    # Intensity 1e-6 F/B, far below ridge → memory-bound.
    flops = 10**3
    nbytes = 10**9
    seconds = nbytes / V5E_HBM_BANDWIDTH
    roof = analyze(flops=flops, nbytes=nbytes, seconds=seconds)
    assert roof.binds == "memory"
    assert roof.bw_util == pytest.approx(1.0)


def test_sol_picks_binding_floor() -> None:
    compute_bound = analyze(flops=10**12, nbytes=1, seconds=1.0)
    assert compute_bound.sol_s == compute_bound.compute_floor_s

    memory_bound = analyze(flops=1, nbytes=10**12, seconds=1.0)
    assert memory_bound.sol_s == memory_bound.memory_floor_s


def test_int8_uses_int8_peak() -> None:
    # nbytes is small but non-zero so memory_floor_s isn't degenerate; the
    # assertion is about which compute peak `analyze` picked, not memory.
    flops = 10**11
    roof = analyze(flops=flops, nbytes=1, seconds=1.0, flop_dtype="int8")
    assert roof.peak_flops == V5E_INT8_PEAK_OPS
    assert roof.mfu == pytest.approx(flops / V5E_INT8_PEAK_OPS)


def test_arithmetic_intensity_handles_zero_bytes() -> None:
    roof = analyze(flops=10, nbytes=0, seconds=1.0)
    assert roof.arithmetic_intensity == float("inf")


def test_arithmetic_intensity_positive_case() -> None:
    roof = analyze(flops=10, nbytes=2, seconds=1.0)
    assert roof.arithmetic_intensity == pytest.approx(5.0)


def test_peak_flops_for_each_dtype() -> None:
    hw = v5e()
    assert peak_flops_for(hw, "bf16") == V5E_BF16_PEAK_FLOPS
    assert peak_flops_for(hw, "f32") == V5E_F32_PEAK_FLOPS
    assert peak_flops_for(hw, "int8") == V5E_INT8_PEAK_OPS
    # Scales with num_chips like the rest of the totals.
    hw8 = v5e(num_chips=8)
    assert peak_flops_for(hw8, "bf16") == V5E_BF16_PEAK_FLOPS * 8
    assert peak_flops_for(hw8, "int8") == V5E_INT8_PEAK_OPS * 8


def test_binds_ties_resolve_to_compute() -> None:
    # Pin the documented tie-break: compute_floor >= memory_floor → "compute".
    # If a future refactor flips this to `>`, this test will catch it.
    flops = 10**9
    nbytes = round(flops * V5E_HBM_BANDWIDTH / V5E_BF16_PEAK_FLOPS)
    roof = analyze(flops=flops, nbytes=nbytes, seconds=1.0)
    assert roof.compute_floor_s == pytest.approx(roof.memory_floor_s, rel=1e-9)
    assert roof.binds == "compute"


def test_multi_chip_scales_peak_for_mfu() -> None:
    # Same total work on 8 chips → 1/8 the peak time → 100% MFU.
    hw = v5e(num_chips=8)
    flops = int(V5E_BF16_PEAK_FLOPS)
    roof = analyze(flops=flops, nbytes=0, seconds=1.0 / 8, hw=hw)
    assert roof.mfu == pytest.approx(1.0)


def test_sol_pct_at_speed_of_light() -> None:
    flops = int(V5E_BF16_PEAK_FLOPS)  # 1 sec at peak compute
    roof = analyze(flops=flops, nbytes=1, seconds=1.0)
    assert roof.sol_pct == pytest.approx(1.0)


def test_f32_uses_f32_peak() -> None:
    # See test_int8_uses_int8_peak for the rationale on nbytes=1.
    flops = 10**11
    roof = analyze(flops=flops, nbytes=1, seconds=1.0, flop_dtype="f32")
    assert roof.peak_flops == V5E_F32_PEAK_FLOPS
    assert roof.mfu == pytest.approx(flops / V5E_F32_PEAK_FLOPS)


def test_ici_floor_zero_when_no_ici_bytes() -> None:
    # Default single-chip path: ici_bytes=0 → ici_floor and ici_util are 0,
    # and ici doesn't enter the sol_s max.
    roof = analyze(flops=10, nbytes=10, seconds=1.0)
    assert roof.ici_floor_s == 0.0
    assert roof.ici_util == 0.0
    assert roof.binds in {"compute", "memory"}


def test_ici_bound_kernel_binds_ici() -> None:
    # All-to-all-ish pattern: small flops, small HBM, dominant ICI traffic.
    hw = v5e(num_chips=8)
    ici_bytes = 10**11  # 100 GB across the slice
    seconds = ici_bytes / hw.total_ici_bw
    roof = analyze(
        flops=10**6,
        nbytes=10**6,
        seconds=seconds,
        hw=hw,
        ici_bytes=ici_bytes,
    )
    assert roof.binds == "ici"
    assert roof.ici_util == pytest.approx(1.0)
    assert roof.sol_s == pytest.approx(roof.ici_floor_s)


def test_binds_compute_wins_over_ici_in_tie() -> None:
    # Pin tie-break: compute >= ici → "compute" (preserves the original
    # compute >= memory rule extended to three terms).
    hw = v5e()
    flops = int(hw.total_bf16_flops)  # compute_floor_s = 1.0
    ici_bytes = int(hw.total_ici_bw)  # ici_floor_s   = 1.0
    roof = analyze(flops=flops, nbytes=0, seconds=1.0, hw=hw, ici_bytes=ici_bytes)
    assert roof.compute_floor_s == pytest.approx(roof.ici_floor_s, rel=1e-9)
    assert roof.binds == "compute"


def test_binds_memory_wins_over_ici_in_tie() -> None:
    # When compute is the smallest floor, memory >= ici → "memory".
    hw = v5e()
    nbytes = int(hw.total_hbm_bw)  # memory_floor_s = 1.0
    ici_bytes = int(hw.total_ici_bw)  # ici_floor_s   = 1.0
    roof = analyze(flops=1, nbytes=nbytes, seconds=1.0, hw=hw, ici_bytes=ici_bytes)
    assert roof.memory_floor_s == pytest.approx(roof.ici_floor_s, rel=1e-9)
    assert roof.binds == "memory"


def test_regime_above_ridge_is_compute_bound() -> None:
    assert regime(arithmetic_intensity=300.0, ridge_point=240.0) == "compute-bound"


def test_regime_below_ridge_is_memory_bound() -> None:
    assert regime(arithmetic_intensity=100.0, ridge_point=240.0) == "memory-bound"


def test_regime_tie_resolves_to_compute_bound() -> None:
    # Mirrors `Roofline.binds` tie-break: equal → compute.
    assert regime(arithmetic_intensity=240.0, ridge_point=240.0) == "compute-bound"


def test_regime_handles_infinite_intensity() -> None:
    # Zero-byte kernels (compute-only on registers) report inf intensity;
    # they should still classify as compute-bound, not blow up.
    assert regime(arithmetic_intensity=float("inf"), ridge_point=240.0) == "compute-bound"


def test_binding_regime_compute_bound() -> None:
    # High intensity against both HBM and ICI → compute floor dominates.
    hw = v5e(num_chips=4)
    assert binding_regime(flops=10**12, nbytes=10**6, ici_bytes=10**6, hw=hw) == "compute-bound"


def test_binding_regime_memory_bound() -> None:
    # Tiny flops, large HBM traffic, negligible ICI → HBM floor dominates.
    hw = v5e(num_chips=4)
    assert binding_regime(flops=1, nbytes=10**11, ici_bytes=1, hw=hw) == "memory-bound"


def test_binding_regime_ici_bound() -> None:
    # Cross-chip traffic dwarfs both compute and HBM floors → ICI binds.
    # This is the case the single-resource `regime()` could never surface.
    hw = v5e(num_chips=4)
    assert binding_regime(flops=10**9, nbytes=10**6, ici_bytes=10**11, hw=hw) == "ici-bound"


def test_binding_regime_single_chip_never_ici() -> None:
    # ici_bytes=0 (the default single-chip path) collapses to the 2-way
    # compute/memory split and agrees with `regime()`.
    hw = v5e()
    assert binding_regime(flops=10**12, nbytes=10**6, ici_bytes=0, hw=hw) == "compute-bound"
    assert binding_regime(flops=1, nbytes=10**9, ici_bytes=0, hw=hw) == "memory-bound"


def test_binding_regime_respects_flop_dtype() -> None:
    # int8 ridge is ~2x the bf16 ridge; at intensity 300 F/B the same workload
    # is compute-bound in bf16 but memory-bound in int8.
    hw = v5e()
    flops, nbytes = 300_000, 1_000
    assert binding_regime(flops, nbytes, ici_bytes=0, hw=hw, flop_dtype="bf16") == "compute-bound"
    assert binding_regime(flops, nbytes, ici_bytes=0, hw=hw, flop_dtype="int8") == "memory-bound"


class _FakeDevice:
    """Stand-in for ``jax.devices()[0]`` — only ``platform`` and ``device_kind``
    matter for the supported-hardware check. Built per test so each test
    can express its own scenario without polluting module state."""

    def __init__(self, platform: str, device_kind: str) -> None:
        self.platform = platform
        self.device_kind = device_kind


def _patch_devices(monkeypatch: pytest.MonkeyPatch, devices: list[Any]) -> None:
    """Replace ``jax.devices`` with one that returns ``devices``."""
    monkeypatch.setattr("jax.devices", lambda *args, **kwargs: devices)


def test_check_supported_hardware_accepts_v5e_on_cpu_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default path: developer running tests on a CPU machine. The
    hw kind is v5e (the project target), no TPU device to cross-check
    against, so the check passes silently."""
    _patch_devices(monkeypatch, [_FakeDevice(platform="cpu", device_kind="cpu")])
    check_supported_hardware(v5e())  # no raise


def test_check_supported_hardware_accepts_v5e_on_v5e_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical happy path: hw=v5e and device is "TPU v5 lite" (the JAX
    string for v5e). Both agree → no raise."""
    _patch_devices(monkeypatch, [_FakeDevice(platform="tpu", device_kind="TPU v5 lite")])
    check_supported_hardware(v5e())  # no raise


def test_check_supported_hardware_rejects_non_v5e_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The roofline constants in roofline.py are v5e-specific. A
    HardwarePeak with a different ``kind`` would pass numbers from
    another generation through the math and silently produce wrong
    BW%/MFU% — refuse outright."""
    _patch_devices(monkeypatch, [_FakeDevice(platform="cpu", device_kind="cpu")])
    fake = HardwarePeak(
        kind="v6e",
        bf16_flops=918e12,
        f32_flops=459e12,
        int8_ops=1836e12,
        hbm_bw=1640e9,
        ici_bw_per_link=400e9,
        ici_links_per_chip=4,
    )
    with pytest.raises(NotImplementedError, match="v5e"):
        check_supported_hardware(fake)


def test_check_supported_hardware_rejects_non_v5e_tpu_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller passed hw=v5e() but the host is actually a v4 (or v5p, or
    v6e). The math would compile and run, but BW%/MFU% would be lies.
    Refuse — the suite must be re-pointed at the right hw."""
    _patch_devices(monkeypatch, [_FakeDevice(platform="tpu", device_kind="TPU v4")])
    with pytest.raises(NotImplementedError, match="v5e"):
        check_supported_hardware(v5e())


def test_check_supported_hardware_skips_tpu_check_on_gpu_or_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On non-TPU platforms (CPU, GPU) the device-kind check doesn't apply
    — the suite is in interpret/correctness mode, hw constants are just
    the model. Only the kind-field check fires."""
    _patch_devices(monkeypatch, [_FakeDevice(platform="gpu", device_kind="NVIDIA A100")])
    check_supported_hardware(v5e())  # no raise


@pytest.mark.tpu
def test_v5e_vmem_capacity_matches_runtime() -> None:
    """Catch drift between V5E_VMEM_CAPACITY and JAX's own per-generation
    constant in ``pltpu.get_tpu_info()``. Running on a real v5e is the
    authoritative cross-check — the runtime answer reflects whatever the
    XLA:TPU compiler will actually allow you to allocate."""
    from jax.experimental.pallas import tpu as pltpu

    info = pltpu.get_tpu_info()
    chip = info.chip_version.value  # pyright: ignore[reportAttributeAccessIssue]
    assert chip == "v5e", f"this test must run on v5e; got {chip}"
    assert info.vmem_capacity_bytes == V5E_VMEM_CAPACITY, (
        f"V5E_VMEM_CAPACITY={V5E_VMEM_CAPACITY} ({V5E_VMEM_CAPACITY / 2**20:.0f} MiB) "
        f"but runtime reports {info.vmem_capacity_bytes} "
        f"({info.vmem_capacity_bytes / 2**20:.0f} MiB). Update roofline.py."
    )


@pytest.mark.tpu
def test_v5e_smem_capacity_matches_runtime() -> None:
    """Same idea as the VMEM cross-check but for SMEM (the 32-bit scalar
    scratchpad). Used today only by ``benchmarks/probe_hardware.py``;
    pinning it here so a future v5p/v6e port catches the change."""
    from jax.experimental.pallas import tpu as pltpu

    info = pltpu.get_tpu_info()
    chip = info.chip_version.value  # pyright: ignore[reportAttributeAccessIssue]
    assert chip == "v5e", f"this test must run on v5e; got {chip}"
    assert info.smem_capacity_bytes == V5E_SMEM_CAPACITY, (
        f"V5E_SMEM_CAPACITY={V5E_SMEM_CAPACITY} ({V5E_SMEM_CAPACITY / 2**10:.0f} KiB) "
        f"but runtime reports {info.smem_capacity_bytes} "
        f"({info.smem_capacity_bytes / 2**10:.0f} KiB). Update roofline.py."
    )
