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


def test_default_params_size_clears_vmem_floor() -> None:
    """No-flag ``params`` footprint must be >=4x v5e VMEM (CLAUDE.md / Bench
    inputs). For a random-index gather, the working set that could be
    VMEM-resident across chained calls is ``params`` — same tensor every
    call — not the output, since fresh ``ids`` mean each call selects
    different rows. Pin ``vocab * hidden`` against the floor; ``m`` is
    irrelevant to this concern.
    """
    args = _make_parser().parse_args([])
    bf16_bytes = 2  # default --dtype bf16
    params_bytes = args.vocab * args.hidden * bf16_bytes
    floor = 4 * V5E_VMEM_CAPACITY
    assert params_bytes >= floor, (
        f"default embedding_lookup params is {params_bytes / 2**20:.0f} MiB; "
        f"floor is 4x v5e VMEM = {floor / 2**20:.0f} MiB. Bump --vocab / --hidden."
    )
