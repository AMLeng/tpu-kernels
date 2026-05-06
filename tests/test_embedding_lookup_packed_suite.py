"""Regression tests for benchmarks/suites/embedding_lookup_packed.py."""

from __future__ import annotations

import pytest
from benchmarks.roofline import V5E_VMEM_CAPACITY
from benchmarks.suites._common import validate_block_shapes
from benchmarks.suites.embedding_lookup_packed import _make_parser

from tpu_kernels.ops.embedding_lookup_packed.pallas import DEFAULT_BLOCK


def test_block_default_matches_kernel_default_block() -> None:
    """`uv run python -m benchmarks.suites.embedding_lookup_packed` (no
    flags) must use the kernel's ``DEFAULT_BLOCK`` so the documented
    ``Current %`` in ``PERF.md`` is reproducible."""
    args = _make_parser().parse_args([])
    assert tuple(args.block) == DEFAULT_BLOCK, (
        f"suite --block default {tuple(args.block)} != kernel DEFAULT_BLOCK "
        f"{DEFAULT_BLOCK}; running the suite with no flags would report a "
        f"different number than PERF.md claims as Current."
    )


def test_timing_default_matches_perf_md_mode() -> None:
    args = _make_parser().parse_args([])
    assert args.timing == "device"


def test_timing_flag_accepts_device() -> None:
    args = _make_parser().parse_args(["--timing", "device"])
    assert args.timing == "device"


def test_block_rejects_wrong_axis_count() -> None:
    parser = _make_parser()
    args = parser.parse_args(["--block", "128", "128"])
    with pytest.raises(SystemExit):
        validate_block_shapes(args, expected_axes=1, parser=parser)


def test_default_params_size_clears_vmem_floor() -> None:
    """Same VMEM-floor pin as the natural-2-D suite; ``params`` must be
    >=4x v5e VMEM so chained-call pipelining can't fake a higher BW%."""
    args = _make_parser().parse_args([])
    bf16_bytes = 2  # default --dtype bf16
    params_bytes = args.vocab * args.hidden * bf16_bytes
    floor = 4 * V5E_VMEM_CAPACITY
    assert params_bytes >= floor, (
        f"default embedding_lookup_packed params is {params_bytes / 2**20:.0f} MiB; "
        f"floor is 4x v5e VMEM = {floor / 2**20:.0f} MiB. Bump --vocab / --hidden."
    )
