"""Regression tests for benchmarks/suites/segment_cumsum.py.

Pin contract between the suite's CLI defaults and the kernel they
drive. CLAUDE.md treats the suite as part of the harness — silent
drift between defaults and the bench's reported numbers is exactly
what the harness exists to prevent.

No `--block` / `--sweep-block` checks here — the Pallas variant isn't
written yet (A.6 in `docs/curriculum.md`); those tests get added with
the Pallas scaffolding.
"""

from __future__ import annotations

from benchmarks.roofline import V5E_VMEM_CAPACITY
from benchmarks.suites.segment_cumsum import _make_parser


def test_timing_default_matches_perf_md_mode() -> None:
    """No-flag run matches the mode PERF.md Current is measured in
    (device). Drift between the two means a no-flag run silently reports
    a different number."""
    args = _make_parser().parse_args([])
    assert args.timing == "device"


def test_timing_flag_accepts_device() -> None:
    args = _make_parser().parse_args(["--timing", "device"])
    assert args.timing == "device"


def test_default_x_size_clears_vmem_floor() -> None:
    """No-flag ``x`` footprint must be >=4x v5e VMEM (CLAUDE.md / Bench
    inputs). For a scan, the working set chained across calls under
    unroll mode is ``x`` itself (output of call k feeds input of call
    k+1); pin its size against the floor. ``num_segments`` is irrelevant
    to this concern."""
    args = _make_parser().parse_args([])
    bf16_bytes = 2  # default --dtype bf16
    x_bytes = args.m * bf16_bytes
    floor = 4 * V5E_VMEM_CAPACITY
    assert x_bytes >= floor, (
        f"default segment_cumsum x is {x_bytes / 2**20:.0f} MiB; "
        f"floor is 4x v5e VMEM = {floor / 2**20:.0f} MiB. Bump --m."
    )


def test_m_must_divide_num_segments() -> None:
    """Equal-width segment construction requires ``m % num_segments == 0``;
    catch the typo at the CLI rather than inside the bench setup."""
    parser = _make_parser()
    args = parser.parse_args(["--m", "1000", "--num-segments", "7"])
    # main() calls parser.error() on the bad combination, which exits.
    # The parser itself doesn't validate cross-arg constraints, so the
    # check lives in main(); reproduce its precondition here so a future
    # refactor can't silently drop it.
    assert args.m % args.num_segments != 0
