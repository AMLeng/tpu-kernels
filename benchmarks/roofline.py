"""Roofline math for TPU v5e.

A roofline tells you the floor on execution time given physical limits:
the compute peak, the HBM bandwidth peak, and (for multi-chip kernels)
the ICI bandwidth peak. Speed-of-light is whichever limit binds. MFU%,
HBM BW%, and ICI BW% are how close the actual run got to those peaks.

All numbers are per-chip. For multi-chip runs, peaks are multiplied by
`num_chips` when computing utilization (the implementation does this).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FlopDtype = Literal["bf16", "f32", "int8"]

# v5e per-chip peaks. Sources: Google Cloud TPU v5e announcement + jax-ml/maxtext
# constants. Tune in one place if Google revises figures.
V5E_BF16_PEAK_FLOPS = 197e12  # 197 TFLOP/s
# f32 has no separately published v5e peak; the MXU is bf16-native and emulates
# f32 matmul at roughly half rate. VPU elementwise f32 is lower again. Use this
# as a rough upper bound; override per-kernel if you have a measured number.
V5E_F32_PEAK_FLOPS = 98e12
V5E_INT8_PEAK_OPS = 393e12  # 393 TOPS, for int8 matmul
V5E_HBM_BANDWIDTH = 819e9  # 819 GB/s
V5E_HBM_CAPACITY = 16 * 1024**3  # 16 GiB
V5E_VMEM_CAPACITY = 32 * 1024**2  # 32 MiB (per TensorCore)
V5E_ICI_PER_LINK = 200e9  # 200 GB/s per link (1600 Gbps), torus topology
V5E_ICI_LINKS_PER_CHIP = 4  # 2D torus: 2 links per axis, 2 axes


@dataclass(frozen=True)
class HardwarePeak:
    """Per-chip peak rates for a TPU generation, scaled by `num_chips`."""

    bf16_flops: float
    f32_flops: float
    int8_ops: float
    hbm_bw: float
    ici_bw_per_link: float
    ici_links_per_chip: int
    num_chips: int = 1

    @property
    def total_bf16_flops(self) -> float:
        return self.bf16_flops * self.num_chips

    @property
    def total_f32_flops(self) -> float:
        return self.f32_flops * self.num_chips

    @property
    def total_int8_ops(self) -> float:
        return self.int8_ops * self.num_chips

    @property
    def total_hbm_bw(self) -> float:
        return self.hbm_bw * self.num_chips

    @property
    def total_ici_bw(self) -> float:
        """Aggregate ICI bandwidth across all links on all chips."""
        return self.ici_bw_per_link * self.ici_links_per_chip * self.num_chips


def v5e(num_chips: int = 1) -> HardwarePeak:
    return HardwarePeak(
        bf16_flops=V5E_BF16_PEAK_FLOPS,
        f32_flops=V5E_F32_PEAK_FLOPS,
        int8_ops=V5E_INT8_PEAK_OPS,
        hbm_bw=V5E_HBM_BANDWIDTH,
        ici_bw_per_link=V5E_ICI_PER_LINK,
        ici_links_per_chip=V5E_ICI_LINKS_PER_CHIP,
        num_chips=num_chips,
    )


def peak_flops_for(hw: HardwarePeak, flop_dtype: FlopDtype) -> float:
    """Resolve the total compute peak (across `num_chips`) for a given dtype.

    Single source of truth: both `Roofline.peak_flops` and the ridge / regime
    classification in `compare._print_table` route through this so they can't
    drift out of sync.
    """
    if flop_dtype == "int8":
        return hw.total_int8_ops
    if flop_dtype == "f32":
        return hw.total_f32_flops
    return hw.total_bf16_flops


def arithmetic_intensity(flops: int, nbytes: int) -> float:
    """FLOPs per byte. Returns inf when nbytes == 0 (compute-only kernels)."""
    return flops / nbytes if nbytes else float("inf")


Regime = Literal["compute-bound", "memory-bound"]


def regime(arithmetic_intensity: float, ridge_point: float) -> Regime:
    """Classify a kernel as compute- or memory-bound on the roofline.

    `ridge_point` is peak_flops / peak_bw — the arithmetic intensity at which
    a kernel transitions from memory-bound to compute-bound. Tie-break to
    "compute-bound" matches the `Roofline.binds` rule for consistency.
    """
    return "compute-bound" if arithmetic_intensity >= ridge_point else "memory-bound"


@dataclass(frozen=True)
class Roofline:
    """Result of roofline analysis for one kernel invocation."""

    flops: int
    nbytes: int
    seconds: float
    hw: HardwarePeak
    flop_dtype: FlopDtype = "bf16"
    ici_bytes: int = 0  # ICI bytes traversed per call; 0 for single-chip kernels

    @property
    def peak_flops(self) -> float:
        return peak_flops_for(self.hw, self.flop_dtype)

    @property
    def compute_floor_s(self) -> float:
        """Lower bound on runtime if compute is the only limit."""
        return self.flops / self.peak_flops

    @property
    def memory_floor_s(self) -> float:
        """Lower bound on runtime if HBM bandwidth is the only limit."""
        return self.nbytes / self.hw.total_hbm_bw

    @property
    def ici_floor_s(self) -> float:
        """Lower bound on runtime if ICI bandwidth is the only limit."""
        if self.ici_bytes == 0 or self.hw.total_ici_bw == 0:
            return 0.0
        return self.ici_bytes / self.hw.total_ici_bw

    @property
    def sol_s(self) -> float:
        """Speed-of-light: whichever floor binds."""
        return max(self.compute_floor_s, self.memory_floor_s, self.ici_floor_s)

    @property
    def binds(self) -> str:
        """Which limit binds. Tie-break order: compute → memory → ici."""
        cf, mf, icf = self.compute_floor_s, self.memory_floor_s, self.ici_floor_s
        if cf >= mf and cf >= icf:
            return "compute"
        if mf >= icf:
            return "memory"
        return "ici"

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte. Compare to peak_flops/peak_bw to see compute vs memory regime."""
        return arithmetic_intensity(self.flops, self.nbytes)

    @property
    def mfu(self) -> float:
        """Achieved FLOPs / peak FLOPs."""
        return (self.flops / self.seconds) / self.peak_flops

    @property
    def bw_util(self) -> float:
        """Achieved HBM bytes/s / peak HBM bandwidth."""
        return (self.nbytes / self.seconds) / self.hw.total_hbm_bw

    @property
    def ici_util(self) -> float:
        """Achieved ICI bytes/s / peak ICI bandwidth. 0 when ici_bytes is 0."""
        if self.ici_bytes == 0 or self.hw.total_ici_bw == 0:
            return 0.0
        return (self.ici_bytes / self.seconds) / self.hw.total_ici_bw

    @property
    def sol_pct(self) -> float:
        """Fraction of speed-of-light hit. 1.0 means kernel is on the roofline."""
        return self.sol_s / self.seconds


def analyze(
    flops: int,
    nbytes: int,
    seconds: float,
    hw: HardwarePeak | None = None,
    flop_dtype: FlopDtype = "bf16",
    ici_bytes: int = 0,
) -> Roofline:
    return Roofline(
        flops=flops,
        nbytes=nbytes,
        seconds=seconds,
        hw=hw if hw is not None else v5e(),
        flop_dtype=flop_dtype,
        ici_bytes=ici_bytes,
    )
