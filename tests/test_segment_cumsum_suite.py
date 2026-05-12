"""Regression tests for benchmarks/suites/segment_cumsum.py.

Pin contract between the suite's CLI defaults and the kernel they
drive. CLAUDE.md treats the suite as part of the harness — silent
drift between defaults and the bench's reported numbers is exactly
what the harness exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.roofline import V5E_VMEM_CAPACITY
from benchmarks.suites._common import validate_block_shapes
from benchmarks.suites.segment_cumsum import _build_inputs, _make_parser

from tpu_kernels.ops.segment_cumsum.pallas import DEFAULT_BLOCK


def test_block_default_matches_kernel_default_block() -> None:
    """No-flag bench must use the kernel's ``DEFAULT_BLOCK``. Drift means
    the suite's PERF.md ``Current`` would be measured at a different bm
    than the bench actually picks at no-flag."""
    args = _make_parser().parse_args([])
    assert tuple(args.block) == DEFAULT_BLOCK, (
        f"suite --block default {tuple(args.block)} != kernel DEFAULT_BLOCK {DEFAULT_BLOCK}"
    )


def test_block_rejects_wrong_axis_count() -> None:
    """segment_cumsum tiles a 1-D buffer; a 2-axis ``--block`` is a typo
    against a 2-D op's flag pattern. Catch it before it reaches the kernel."""
    parser = _make_parser()
    args = parser.parse_args(["--block", "128", "256"])
    with pytest.raises(SystemExit):
        validate_block_shapes(args, expected_axes=1, parser=parser)


def test_sweep_block_rejects_wrong_axis_count() -> None:
    """A 2-axis sweep would otherwise blow up the cartesian product and
    crash deep inside ``segment_cumsum_pallas``."""
    parser = _make_parser()
    args = parser.parse_args(["--sweep-block", "128,256", "512,1024"])
    with pytest.raises(SystemExit):
        validate_block_shapes(args, expected_axes=1, parser=parser)


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
    k+1); pin its size against the floor. ``mean_length`` is irrelevant
    to this concern."""
    args = _make_parser().parse_args([])
    bf16_bytes = 2  # default --dtype bf16
    x_bytes = args.m * bf16_bytes
    floor = 4 * V5E_VMEM_CAPACITY
    assert x_bytes >= floor, (
        f"default segment_cumsum x is {x_bytes / 2**20:.0f} MiB; "
        f"floor is 4x v5e VMEM = {floor / 2**20:.0f} MiB. Bump --m."
    )


def test_default_mean_length_stresses_boundary_path() -> None:
    """Pin the default mean_length below the threshold where boundary
    rows become rare. With 128-lane rows, mean_length > ~32K means <0.4%
    of rows have a within-row boundary and segment_cumsum reduces to
    plain cumsum + a correction so sparse it doesn't matter — at which
    point the bench measures cumsum, not segment_cumsum. Cap it well
    below that so the kernel actually exercises the segment-reset path."""
    args = _make_parser().parse_args([])
    assert args.mean_length <= 4096, (
        f"default mean_length={args.mean_length} is too sparse — boundary "
        f"rows would be <3% of all rows and the bench effectively measures "
        f"plain cumsum."
    )


def test_default_construction_has_within_row_boundaries() -> None:
    """The kernel's within-row segment-reset path is what distinguishes
    ``segment_cumsum`` from plain ``cumsum``. Any construction that lands
    every boundary at a 128-lane row start (uniform widths divisible by
    128 being the canonical failure) skips that path and the bench then
    measures the wrong thing. Pin that at least some boundaries fall
    inside rows at a representative size."""
    args = _make_parser().parse_args(["--m", str(128 * 1024)])
    _, segment_ids, _ = _build_inputs(args)
    sids = np.asarray(segment_ids)
    transitions = np.where(np.diff(sids) != 0)[0] + 1
    assert len(transitions) > 0, "no segment boundaries at all"
    within_row = (transitions % 128) != 0
    assert within_row.any(), (
        "no within-row boundaries — every transition lands at a 128-lane "
        "row start, so the kernel runs on its plain-cumsum fast path."
    )


def test_build_inputs_segment_ids_cover_m() -> None:
    """The constructed ``segment_ids`` must have one entry per element of
    ``x`` and be non-decreasing — pin the contract the kernel relies on so
    a future refactor of the (now non-divisible) construction can't drop
    or duplicate elements."""
    args = _make_parser().parse_args(["--m", "8192"])
    x, segment_ids, num_segments = _build_inputs(args)
    assert segment_ids.shape == x.shape
    sids = np.asarray(segment_ids)
    assert np.all(np.diff(sids) >= 0)
    assert sids[0] == 0
    assert sids[-1] == num_segments - 1
