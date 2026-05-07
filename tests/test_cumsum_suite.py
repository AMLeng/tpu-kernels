"""Regression tests for benchmarks/suites/cumsum.py.

Pin contract between the suite's CLI defaults and the kernel they
drive. CLAUDE.md treats the suite as part of the harness — silent
drift between defaults and the bench's reported numbers is exactly
what the harness exists to prevent.
"""

from __future__ import annotations

import pytest
from benchmarks.roofline import V5E_VMEM_CAPACITY
from benchmarks.suites._common import validate_block_shapes
from benchmarks.suites.cumsum import _make_parser

from tpu_kernels.ops.cumsum.pallas import DEFAULT_BLOCK


def test_block_default_matches_kernel_default_block() -> None:
    """No-flag bench must use the kernel's ``DEFAULT_BLOCK``. Drift means
    the suite's PERF.md ``Current`` would be measured at a different bm
    than the bench actually picks at no-flag."""
    args = _make_parser().parse_args([])
    assert tuple(args.block) == DEFAULT_BLOCK, (
        f"suite --block default {tuple(args.block)} != kernel DEFAULT_BLOCK {DEFAULT_BLOCK}"
    )


def test_timing_default_matches_perf_md_mode() -> None:
    """No-flag run matches the mode PERF.md Current is measured in
    (device). Drift between the two means a no-flag run silently reports
    a different number."""
    args = _make_parser().parse_args([])
    assert args.timing == "device"


def test_timing_flag_accepts_device() -> None:
    args = _make_parser().parse_args(["--timing", "device"])
    assert args.timing == "device"


def test_block_rejects_wrong_axis_count() -> None:
    """cumsum tiles a 1-D buffer; a 2-axis ``--block`` is a typo against
    a 2-D op's flag pattern. Catch it before it reaches the kernel."""
    parser = _make_parser()
    args = parser.parse_args(["--block", "128", "256"])
    with pytest.raises(SystemExit):
        validate_block_shapes(args, expected_axes=1, parser=parser)


def test_sweep_block_rejects_wrong_axis_count() -> None:
    """A 2-axis sweep would otherwise blow up the cartesian product and
    crash deep inside ``cumsum_pallas``."""
    parser = _make_parser()
    args = parser.parse_args(["--sweep-block", "128,256", "512,1024"])
    with pytest.raises(SystemExit):
        validate_block_shapes(args, expected_axes=1, parser=parser)


def test_default_x_size_clears_vmem_floor() -> None:
    """No-flag ``x`` footprint must be >=4x v5e VMEM (CLAUDE.md / Bench
    inputs). For a scan, the working set chained across calls under
    unroll mode is ``x`` itself (output of call k feeds input of call
    k+1); pin its size against the floor."""
    args = _make_parser().parse_args([])
    bf16_bytes = 2  # default --dtype bf16
    x_bytes = args.m * bf16_bytes
    floor = 4 * V5E_VMEM_CAPACITY
    assert x_bytes >= floor, (
        f"default cumsum x is {x_bytes / 2**20:.0f} MiB; "
        f"floor is 4x v5e VMEM = {floor / 2**20:.0f} MiB. Bump --m."
    )
