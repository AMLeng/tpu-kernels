"""Regression tests for benchmarks/suites/embedding_lookup.py.

Pin contract between the suite's CLI defaults and the kernel they
drive. CLAUDE.md treats the suite as part of the harness — silent
drift between defaults and the bench's reported numbers is exactly
what the harness exists to prevent.

No `--block` / `--sweep-block` checks here — the Pallas variant isn't
written yet (A.5 in `docs/curriculum.md`); those tests get added with
the Pallas scaffolding.
"""

from __future__ import annotations

from benchmarks.roofline import V5E_VMEM_CAPACITY
from benchmarks.suites.embedding_lookup import _make_parser


def test_timing_default_matches_perf_md_mode() -> None:
    """No-flag run matches the mode PERF.md Current is measured in
    (device). Drift between the two means a no-flag run silently reports
    a different number than PERF.md claims."""
    args = _make_parser().parse_args([])
    assert args.timing == "device"


def test_timing_flag_accepts_device() -> None:
    args = _make_parser().parse_args(["--timing", "device"])
    assert args.timing == "device"


def test_default_output_size_clears_vmem_floor() -> None:
    """No-flag output footprint must be >=4x v5e VMEM (CLAUDE.md / Bench
    inputs). For a gather, `params` is huge by construction
    (`vocab * hidden`) so it can't be cached in VMEM regardless; the
    footprint that *could* be cached across chained calls is the
    `M * hidden` output. Pin that against the floor.
    """
    args = _make_parser().parse_args([])
    bf16_bytes = 2  # default --dtype bf16
    output_bytes = args.m * args.hidden * bf16_bytes
    floor = 4 * V5E_VMEM_CAPACITY
    assert output_bytes >= floor, (
        f"default embedding_lookup output is {output_bytes / 2**20:.0f} MiB; "
        f"floor is 4x v5e VMEM = {floor / 2**20:.0f} MiB. Bump --m / --hidden."
    )
