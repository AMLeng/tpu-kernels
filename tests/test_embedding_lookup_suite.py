"""Regression tests for benchmarks/suites/embedding_lookup.py.

Pin contract between the suite's CLI defaults and the kernel they
drive. CLAUDE.md treats the suite as part of the harness — silent
drift between defaults and the bench's reported numbers is exactly
what the harness exists to prevent.
"""

from __future__ import annotations

import pytest
from benchmarks.roofline import V5E_VMEM_CAPACITY
from benchmarks.suites._common import validate_block_shapes
from benchmarks.suites.embedding_lookup import _make_parser

from tpu_kernels.ops.embedding_lookup.pallas import DEFAULT_BLOCK


def test_block_default_matches_kernel_default_block() -> None:
    """`uv run python -m benchmarks.suites.embedding_lookup` (no flags)
    must use the kernel's ``DEFAULT_BLOCK`` so the documented
    ``Current %`` in ``PERF.md`` is reproducible without remembering
    ``--block ...``."""
    args = _make_parser().parse_args([])
    assert tuple(args.block) == DEFAULT_BLOCK, (
        f"suite --block default {tuple(args.block)} != kernel DEFAULT_BLOCK "
        f"{DEFAULT_BLOCK}; running the suite with no flags would report a "
        f"different number than PERF.md claims as Current."
    )


def test_timing_default_matches_perf_md_mode() -> None:
    """No-flag run matches the mode PERF.md Current is measured in
    (device). Drift between the two means a no-flag run silently reports
    a different number than PERF.md claims."""
    args = _make_parser().parse_args([])
    assert args.timing == "device"


def test_timing_flag_accepts_device() -> None:
    args = _make_parser().parse_args(["--timing", "device"])
    assert args.timing == "device"


def test_block_rejects_wrong_axis_count() -> None:
    """The kernel takes a 1-axis ``block_shape``; a 2-axis ``--block``
    would crash inside the kernel. Catching it at the CLI keeps the
    error close to the typo."""
    parser = _make_parser()
    args = parser.parse_args(["--block", "128", "128"])
    with pytest.raises(SystemExit):
        validate_block_shapes(args, expected_axes=1, parser=parser)


def test_default_params_size_clears_vmem_floor() -> None:
    """No-flag ``params`` footprint must be >=4x v5e VMEM (CLAUDE.md /
    Bench inputs). For a random-index gather, the working set that
    could be VMEM-resident across chained calls is ``params`` — same
    tensor every call — not the output, since fresh ``ids`` mean each
    call selects different rows. Pin ``vocab * hidden`` against the
    floor; ``m`` is irrelevant to this concern.
    """
    args = _make_parser().parse_args([])
    bf16_bytes = 2  # default --dtype bf16
    params_bytes = args.vocab * args.hidden * bf16_bytes
    floor = 4 * V5E_VMEM_CAPACITY
    assert params_bytes >= floor, (
        f"default embedding_lookup params is {params_bytes / 2**20:.0f} MiB; "
        f"floor is 4x v5e VMEM = {floor / 2**20:.0f} MiB. Bump --vocab / --hidden."
    )
