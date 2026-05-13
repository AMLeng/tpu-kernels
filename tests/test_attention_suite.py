"""Regression tests for benchmarks/suites/attention.py.

CLAUDE.md treats the bench suite as part of the harness — silent drift
between the no-flag run and the contract ``PERF.md`` assumes is the
kernel-correctness equivalent of a silent regression. Tests are minimal
while the op is naive + xla only; the ``--block`` / Pallas-default
checks land when the Pallas variant does.
"""

from __future__ import annotations

from benchmarks.roofline import V5E_VMEM_CAPACITY
from benchmarks.suites.attention import _make_parser


def test_timing_default_matches_perf_md_mode() -> None:
    """No-flag run matches the mode PERF.md Current is measured in
    (device). Drift between the two means a no-flag run silently reports
    a different number than what PERF.md claims."""
    args = _make_parser().parse_args([])
    assert args.timing == "device"


def test_timing_flag_accepts_device() -> None:
    args = _make_parser().parse_args(["--timing", "device"])
    assert args.timing == "device"


def test_default_scores_intermediate_clears_vmem_floor() -> None:
    """The (N, T, T) scores tensor xla materializes must exceed 4x v5e
    VMEM so the wall is structurally visible at defaults.

    The standard suite-default floor (CLAUDE.md, Bench inputs) is sized
    on input tensors to prevent XLA from keeping the inputs in VMEM
    across chained calls and inflating per-call BW%. Attention's input
    footprint (Q+K+V+O) is small — at T=4096, N=16, H=128, bf16 that's
    only 64 MiB. The wall narrative depends instead on the (N, T, T)
    intermediate being too large to live in VMEM, forcing the
    score-then-softmax-then-value flow through HBM. Pin that here. f32
    is the worst case for the intermediate dtype: naive upcasts before
    the matmul and softmax, so xla's HLO writes f32 scores even when
    inputs are bf16.
    """
    args = _make_parser().parse_args([])
    scores_bytes = args.n * args.t * args.t * 4  # f32 intermediate
    floor = 4 * V5E_VMEM_CAPACITY
    assert scores_bytes >= floor, (
        f"default scores intermediate is {scores_bytes / 2**20:.0f} MiB; "
        f"floor is 4x v5e VMEM = {floor / 2**20:.0f} MiB. Bump --t or --n."
    )
