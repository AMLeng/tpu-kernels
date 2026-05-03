"""Regression tests for benchmarks/suites/matmul.py.

Sibling to test_softmax_suite.py / test_rmsnorm_suite.py — pins the
matmul CLI's contract with its kernel.
"""

from __future__ import annotations

import pytest
from benchmarks.roofline import V5E_VMEM_CAPACITY
from benchmarks.suites._common import validate_block_shapes
from benchmarks.suites.matmul import _make_parser

from tpu_kernels.ops.matmul.pallas import DEFAULT_BLOCK


def test_block_default_matches_kernel_default_block() -> None:
    """`uv run python -m benchmarks.suites.matmul` (no flags) must use the
    kernel's ``DEFAULT_BLOCK``. The suite stores it as a list under the base
    parser's ``nargs="+"`` shape; ``tuple(args.block)`` rebuilds the
    canonical tuple form the kernel expects."""
    args = _make_parser().parse_args([])
    assert tuple(args.block) == DEFAULT_BLOCK, (
        f"suite --block default {tuple(args.block)} != kernel DEFAULT_BLOCK "
        f"{DEFAULT_BLOCK}; running the suite with no flags would report a "
        f"different number than PERF.md claims as Current."
    )


def test_timing_default_matches_perf_md_mode() -> None:
    """No-flag run matches the mode PERF.md Current is measured in (device)."""
    args = _make_parser().parse_args([])
    assert args.timing == "device"


def test_timing_flag_accepts_device() -> None:
    args = _make_parser().parse_args(["--timing", "device"])
    assert args.timing == "device"


def test_block_rejects_wrong_axis_count() -> None:
    """matmul tiles in 3-D (bm, bn, bk); a 1-axis ``--block`` is a typo
    against a row-reduction op's flag pattern. Catch it before it reaches
    the kernel."""
    parser = _make_parser()
    args = parser.parse_args(["--block", "128"])
    with pytest.raises(SystemExit):
        validate_block_shapes(args, expected_axes=3, parser=parser)


def test_sweep_block_rejects_wrong_axis_count() -> None:
    """A 1-axis sweep would otherwise blow up the cartesian product and
    crash deep inside ``matmul_pallas``."""
    parser = _make_parser()
    args = parser.parse_args(["--sweep-block", "128,256"])
    with pytest.raises(SystemExit):
        validate_block_shapes(args, expected_axes=3, parser=parser)


def test_default_input_size_clears_vmem_floor() -> None:
    """No-flag inputs must be >=4x v5e VMEM each (CLAUDE.md / Bench inputs).

    Smaller inputs fit on chip and let XLA tile-pipeline chained calls
    under unroll mode; per-call BW% then inflates by ~k. The unroll
    harness wraps chained values in optimization_barrier to block the
    fusion, but the sane-shape floor is the cheaper insurance — pipelining
    can't help once the working set genuinely doesn't fit. matmul has
    two inputs, so both must clear the floor; an undersized A or B is
    enough to let the pipelining trick re-engage.
    """
    args = _make_parser().parse_args([])
    bf16_bytes = 2  # default --dtype bf16
    a_bytes = args.m * args.k * bf16_bytes
    b_bytes = args.k * args.n * bf16_bytes
    floor = 4 * V5E_VMEM_CAPACITY
    assert a_bytes >= floor, (
        f"default matmul A is {a_bytes / 2**20:.0f} MiB; "
        f"floor is 4x v5e VMEM = {floor / 2**20:.0f} MiB. Bump --m / --k."
    )
    assert b_bytes >= floor, (
        f"default matmul B is {b_bytes / 2**20:.0f} MiB; "
        f"floor is 4x v5e VMEM = {floor / 2**20:.0f} MiB. Bump --k / --n."
    )
