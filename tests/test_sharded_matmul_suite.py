"""Regression tests for benchmarks/suites/sharded_matmul.py.

Bench suites are part of the harness as far as CLAUDE.md is concerned —
silent drift between a no-flag run and what PERF.md depends on is the
kernel-correctness equivalent of a silent regression. The scaffold is
naive + xla today, so the pins are minimal: the mesh defaults, the
shape/mesh divisibility a no-flag run relies on, and the non-default
`--timing unroll` (non-obvious enough to be worth a regression net).
"""

from __future__ import annotations

from benchmarks.suites.sharded_matmul import _make_parser

from tpu_kernels.ops.sharded_matmul.naive import DEFAULT_DP, DEFAULT_TP


def test_mesh_defaults_match_kernel_defaults() -> None:
    """Suite --dp/--tp defaults track the kernel's DEFAULT_DP/DEFAULT_TP so a
    no-flag run reproduces the mesh PERF.md's Current is measured against."""
    args = _make_parser().parse_args([])
    assert (args.dp, args.tp) == (DEFAULT_DP, DEFAULT_TP)


def test_timing_default_is_unroll() -> None:
    """The runner's device-timing path doesn't yet coalesce events across
    TPU planes (runner.py `_events_from_line` raises on >1 plane), so this
    suite defaults to unroll. Once the harness extension lands the default
    flips back to device and this pin updates with it.
    """
    args = _make_parser().parse_args([])
    assert args.timing == "unroll"


def test_default_shape_divides_mesh() -> None:
    """batch must divide dp and the contracting dim must divide tp, or
    shard_map raises deep in the lowering. Pin the no-flag combination."""
    args = _make_parser().parse_args([])
    assert args.batch % args.dp == 0
    assert args.d % args.tp == 0


def test_shape_and_mesh_flags_parse() -> None:
    argv = ["--batch", "4096", "--d", "2048", "--f", "1024", "--dp", "4", "--tp", "2"]
    args = _make_parser().parse_args(argv)
    assert (args.batch, args.d, args.f, args.dp, args.tp) == (4096, 2048, 1024, 4, 2)
