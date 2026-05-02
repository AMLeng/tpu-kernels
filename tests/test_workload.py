"""Sanity tests for ``benchmarks/workload.py``.

The dataclass itself is mostly trivial, but pinning the shape here means
a future field rename or default change has to update a test alongside,
which is the cheap insurance CLAUDE.md asks for on harness changes.
"""

from __future__ import annotations

import pytest
from benchmarks.workload import Workload


def test_workload_required_fields() -> None:
    """``op``, ``flops``, ``nbytes`` are required — there's no sensible
    default for any of them per-call."""
    w = Workload(op="scale", flops=100, nbytes=200)
    assert w.op == "scale"
    assert w.flops == 100
    assert w.nbytes == 200


def test_workload_optional_field_defaults() -> None:
    """``args=()`` and ``flop_dtype="bf16"`` mirror the project defaults
    (no extra inputs, bf16 working dtype per CLAUDE.md)."""
    w = Workload(op="x", flops=1, nbytes=1)
    assert w.args == ()
    assert w.flop_dtype == "bf16"


def test_workload_is_frozen() -> None:
    """Workload is a value object: rebuilding-with-changes is cheap and the
    immutability prevents a sweep loop from accidentally mutating shared
    state across configs."""
    w = Workload(op="x", flops=1, nbytes=1)
    with pytest.raises((AttributeError, TypeError)):
        w.op = "y"  # type: ignore[misc]
