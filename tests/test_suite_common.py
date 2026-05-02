"""Regression tests for benchmarks/suites/_common.py.

The base parser is what every op's bench suite layers its shape flags
onto. These tests pin the parts the suites lean on: the shared flag set,
their no-flag defaults (which are what PERF.md's Current is measured
under), the variable-axis ``--block`` / ``--sweep-block`` plumbing that
lets the same flags serve 1-D and 2-D ops, and the wrong-axis-count
rejector each suite calls before destructuring. Per CLAUDE.md, anything
the harness uses to judge kernels is TDD-only — drift here would silently
change every suite at once, so the regression test lives here, not under
each op.
"""

from __future__ import annotations

import argparse

import pytest
from benchmarks.suites._common import (
    base_parser,
    csv_ints,
    validate_block_shapes,
)


def _child_parser(*, block_default: list[int] | None = None) -> argparse.ArgumentParser:
    """Wrap base_parser in a child the way every suite does. ``block_default``
    mimics each suite's ``set_defaults(block=...)`` call; tests that don't
    set it leave ``args.block`` at the parent's ``None`` so they exercise
    only the parts of the contract they care about."""
    parser = argparse.ArgumentParser(parents=[base_parser()])
    if block_default is not None:
        parser.set_defaults(block=block_default)
    return parser


def test_csv_ints_parses_csv() -> None:
    assert csv_ints("8,16,32") == (8, 16, 32)


def test_csv_ints_single_value() -> None:
    assert csv_ints("64") == (64,)


def test_base_parser_no_flag_defaults() -> None:
    """No-flag run reproduces the mode PERF.md's Current is measured in.
    ``--block`` has no parent default — each suite must call
    ``set_defaults(block=...)`` to pin its kernel's ``DEFAULT_BLOCK``."""
    args = _child_parser().parse_args([])
    assert args.dtype == "bf16"
    assert args.dump_hlo is False
    assert args.profile_dir is None
    assert args.timing == "unroll"
    assert args.block is None
    assert args.sweep_block is None


def test_base_parser_timing_accepts_device() -> None:
    args = _child_parser().parse_args(["--timing", "device"])
    assert args.timing == "device"


def test_base_parser_dtype_accepts_f32() -> None:
    args = _child_parser().parse_args(["--dtype", "f32"])
    assert args.dtype == "f32"


def test_block_accepts_one_axis() -> None:
    """1-D ops (rmsnorm-style) pass a single int."""
    args = _child_parser().parse_args(["--block", "128"])
    assert args.block == [128]


def test_block_accepts_two_axes() -> None:
    """2-D ops (scale-style) pass two ints."""
    args = _child_parser().parse_args(["--block", "512", "1024"])
    assert args.block == [512, 1024]


def test_sweep_block_accepts_one_axis() -> None:
    args = _child_parser().parse_args(["--sweep-block", "8,16,32"])
    assert args.sweep_block == [(8, 16, 32)]


def test_sweep_block_accepts_two_axes() -> None:
    args = _child_parser().parse_args(["--sweep-block", "8,16", "128,256"])
    assert args.sweep_block == [(8, 16), (128, 256)]


def test_sweep_block_accepts_three_axes() -> None:
    """Future ops with 3-D block geometry just work; per-op count check
    is what catches misuse, not the parser."""
    args = _child_parser().parse_args(["--sweep-block", "1,2", "3,4", "5,6"])
    assert args.sweep_block == [(1, 2), (3, 4), (5, 6)]


def test_validate_passes_when_both_match() -> None:
    parser = _child_parser(block_default=[8, 16])
    args = parser.parse_args(["--sweep-block", "8,16", "128,256"])
    validate_block_shapes(args, expected_axes=2, parser=parser)  # no raise


def test_validate_passes_when_sweep_unset() -> None:
    """No --sweep-block means single-config compare(); only --block is checked."""
    parser = _child_parser(block_default=[8, 16])
    args = parser.parse_args([])
    validate_block_shapes(args, expected_axes=2, parser=parser)  # no raise


def test_validate_rejects_wrong_block_count() -> None:
    """A 2-D op given a 1-axis --block should refuse rather than crash
    inside the kernel on a shape mismatch."""
    parser = _child_parser(block_default=[8, 16])
    args = parser.parse_args(["--block", "128"])
    with pytest.raises(SystemExit):
        validate_block_shapes(args, expected_axes=2, parser=parser)


def test_validate_rejects_wrong_sweep_block_count() -> None:
    """A 1-D op given a 2-axis sweep would silently multiply the runs and
    crash deep inside the kernel."""
    parser = _child_parser(block_default=[128])
    args = parser.parse_args(["--sweep-block", "8,16", "128,256"])
    with pytest.raises(SystemExit):
        validate_block_shapes(args, expected_axes=1, parser=parser)


def test_validate_rejects_block_default_unset() -> None:
    """A suite that forgets ``parser.set_defaults(block=...)`` leaves
    ``args.block`` at the parent's ``None``. The bare ``len(None)`` would
    crash with a TypeError deep inside ``validate_block_shapes`` — useless
    to a new-op author. Surface it as an actionable parser.error so the
    fix points at the suite's missing ``set_defaults`` line.
    """
    parser = _child_parser()  # no block_default
    args = parser.parse_args([])
    with pytest.raises(SystemExit):
        validate_block_shapes(args, expected_axes=1, parser=parser)
