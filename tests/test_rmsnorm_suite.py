"""Regression tests for benchmarks/suites/rmsnorm.py.

Sibling to test_scale_suite.py — pins the rmsnorm CLI's contract with
its kernel. Today the only real contract worth pinning is the timing
flag (the rmsnorm kernel has no per-block tuning yet); when a Pallas
variant lands and brings tunable knobs, this file is where their
defaults get pinned against the kernel's preferred values.
"""

from __future__ import annotations

from benchmarks.suites.rmsnorm import _make_parser


def test_timing_default_matches_bench_default() -> None:
    """No-flag run uses bench()'s default mode (unroll)."""
    args = _make_parser().parse_args([])
    assert args.timing == "unroll"


def test_timing_flag_accepts_device() -> None:
    args = _make_parser().parse_args(["--timing", "device"])
    assert args.timing == "device"
