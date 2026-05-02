"""Regression tests for benchmarks/suites/rmsnorm.py.

Sibling to test_scale_suite.py — pins the rmsnorm CLI's contract with
its kernel. When a Pallas variant lands and brings tunable knobs, this
file is where their defaults get pinned against the kernel's preferred
values.
"""

from __future__ import annotations

from benchmarks.roofline import V5E_VMEM_CAPACITY
from benchmarks.suites.rmsnorm import _make_parser


def test_timing_default_matches_bench_default() -> None:
    """No-flag run uses bench()'s default mode (unroll)."""
    args = _make_parser().parse_args([])
    assert args.timing == "unroll"


def test_timing_flag_accepts_device() -> None:
    args = _make_parser().parse_args(["--timing", "device"])
    assert args.timing == "device"


def test_default_input_size_clears_vmem_floor() -> None:
    """No-flag input must be >=4x v5e VMEM (CLAUDE.md / Bench inputs).

    Smaller inputs fit on chip and let XLA tile-pipeline chained calls
    under unroll mode; per-call BW% then inflates by ~k. The unroll
    harness wraps chained values in optimization_barrier to block the
    fusion, but the sane-shape floor is the cheaper insurance — pipelining
    can't help once the working set genuinely doesn't fit.
    """
    args = _make_parser().parse_args([])
    bf16_bytes = 2  # default --dtype bf16
    input_bytes = args.bs * args.hidden * bf16_bytes
    floor = 4 * V5E_VMEM_CAPACITY
    assert input_bytes >= floor, (
        f"default rmsnorm input is {input_bytes / 2**20:.0f} MiB; "
        f"floor is 4x v5e VMEM = {floor / 2**20:.0f} MiB. Bump --bs / --hidden."
    )
