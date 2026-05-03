"""Shared CLI scaffolding for op bench suites.

Each suite under ``benchmarks/suites/`` owns its op's shape, inputs, and
flop/byte accounting — the parts that say something about the kernel.
The mechanical bits that don't (timing mode, dtype plumbing, hlo/profile
capture, block / sweep-block parsing) live here so a new op only writes
the interesting half.

Both ``--block`` and ``--sweep-block`` use ``nargs="+"`` / ``nargs="*"``
so the same flags drive 1-D and 2-D (and beyond) ops; each suite calls
``validate_block_shapes`` after parsing to reject the wrong axis count
for its kernel before destructuring.

Bench suites are part of the harness as far as CLAUDE.md is concerned,
so changes here are TDD-only; the regression test is
``tests/test_suite_common.py``.
"""

from __future__ import annotations

import argparse


def csv_ints(s: str) -> tuple[int, ...]:
    """Parse ``"8,16,32"`` → ``(8, 16, 32)``. argparse hook for sweep axes."""
    return tuple(int(x) for x in s.split(","))


def base_parser() -> argparse.ArgumentParser:
    """argparse parent with the flags every suite uses identically.

    Each op's ``_make_parser()`` builds on this with shape flags and
    sets the kernel's ``DEFAULT_BLOCK`` via ``parser.set_defaults(...)``;
    ``add_help=False`` is required so the child parser owns ``-h``
    without colliding.

    Defaults pinned here also pin the contract a no-flag
    ``uv run python -m benchmarks.suites.<op>`` run depends on (PERF.md's
    Current is measured under these); the per-op suite tests verify those
    defaults still hold for each op.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--dump-hlo", action="store_true")
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument(
        "--timing",
        choices=["unroll", "device"],
        default="device",
        help=(
            "Timing mode forwarded to bench(). Default `device` matches "
            "PERF.md and requires a TPU; pass `--timing unroll` for CPU."
        ),
    )
    # Single block geometry, one int per axis. Each suite sets its
    # kernel's DEFAULT_BLOCK as the no-flag value via set_defaults.
    parser.add_argument(
        "--block",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Block shape (one int per axis), e.g. '--block 512 1024' for a 2-D op "
            "or '--block 128' for a 1-D op. Ignored when --sweep-block is set."
        ),
    )
    # nargs="*" lets the same flag drive 1-D ops (one CSV) and 2-D ops
    # (two CSVs); the per-op count check lives in validate_block_shapes().
    parser.add_argument(
        "--sweep-block",
        type=csv_ints,
        nargs="*",
        default=None,
        metavar="AXIS_LIST",
        help=(
            "Cartesian sweep over block-shape axes, one CSV per axis. "
            "1-D op example: --sweep-block 8,16,32. "
            "2-D op example: --sweep-block 8,16,32 128,256."
        ),
    )
    return parser


def validate_block_shapes(
    args: argparse.Namespace,
    *,
    expected_axes: int,
    parser: argparse.ArgumentParser,
) -> None:
    """Hard error if ``--block`` or ``--sweep-block`` were given the wrong
    number of axes for this op.

    Called by each suite after ``parse_args()`` and before destructuring.
    A mismatch surfaces as an argparse-style error rather than a deep
    crash inside the kernel or an unintended cartesian product.
    ``parser.error`` exits with status 2; tests assert ``SystemExit``.

    The ``args.block is None`` branch catches a different mistake: a suite
    that forgot ``parser.set_defaults(block=...)``. ``base_parser`` leaves
    the parent default at None on purpose so each suite must pin its
    kernel's DEFAULT_BLOCK; without this guard the omission surfaces as a
    bare ``TypeError`` from ``len(None)`` — useless to a new-op author.
    """
    if args.block is None:
        parser.error(
            "--block has no default; the suite must call "
            "parser.set_defaults(block=list(DEFAULT_BLOCK)) so a no-flag run "
            "reproduces the kernel's pinned shape."
        )
    if len(args.block) != expected_axes:
        block_example = " ".join(["128"] * expected_axes)
        parser.error(
            f"--block expects {expected_axes} value(s) for this op, got {len(args.block)}. "
            f"Example: --block {block_example}"
        )
    if args.sweep_block is not None and len(args.sweep_block) != expected_axes:
        sweep_example = " ".join(["8,16"] * expected_axes)
        parser.error(
            f"--sweep-block expects {expected_axes} axis list(s) for this op, "
            f"got {len(args.sweep_block)}. Example: --sweep-block {sweep_example}"
        )
