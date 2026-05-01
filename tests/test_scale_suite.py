"""Regression tests for benchmarks/suites/scale.py.

These pin contract between the suite's CLI defaults and the kernel they
drive. CLAUDE.md treats the suite as part of the harness — silent drift
between ``--block`` and the kernel's preferred shape is exactly the kind
of foot-gun the harness exists to prevent (a no-flag run would otherwise
report a different ``BW%`` than ``PERF.md`` claims as Current).
"""

from __future__ import annotations

from benchmarks.suites.scale import _make_parser

from tpu_kernels.ops.scale.pallas import DEFAULT_BLOCK


def test_block_default_matches_kernel_default_block() -> None:
    """`uv run python -m benchmarks.suites.scale` (no flags) must use the
    kernel's ``DEFAULT_BLOCK`` so the documented ``Current %`` in
    ``PERF.md`` is reproducible without remembering ``--block ...``.
    """
    args = _make_parser().parse_args([])
    assert tuple(args.block) == DEFAULT_BLOCK, (
        f"suite --block default {tuple(args.block)} != kernel DEFAULT_BLOCK "
        f"{DEFAULT_BLOCK}; running the suite with no flags would report a "
        f"different number than PERF.md claims as Current."
    )


def test_timing_default_matches_bench_default() -> None:
    """No-flag run uses bench()'s default mode (unroll). ``PERF.md`` Current
    is measured under that mode; switching defaults silently would make a
    no-flag run report a different number."""
    args = _make_parser().parse_args([])
    assert args.timing == "unroll"


def test_timing_flag_accepts_device() -> None:
    args = _make_parser().parse_args(["--timing", "device"])
    assert args.timing == "device"
