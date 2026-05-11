"""Regression tests for the ``--dump-mosaic`` wrapper.

JAX 0.10.0 exposes no env-var equivalent for dumping Pallas's lowered
Mosaic IR; the only knob is the ``debug`` kwarg on ``pl.pallas_call``.
``force_pallas_debug()`` monkey-patches ``pl.pallas_call`` to force
``debug=True`` for the duration of a ``.lower(...).compile()`` pass so
``--dump-mosaic`` works without threading the kwarg through every
kernel signature.

These CPU-runnable tests pin the wrapper's contract; the TPU smoke
(``tests/perf/test_mosaic_dump.py``) is what catches a JAX-side change
to the dump mechanism itself. Both layers are needed because Mosaic
lowering only fires under the TPU lowering rule — CPU runs (including
``interpret=True``) bypass it.
"""

from __future__ import annotations

from typing import Any

import jax.experimental.pallas as pl
import pytest
from benchmarks._mosaic_dump import force_pallas_debug


def test_force_pallas_debug_injects_debug_true_when_caller_omits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside the context, calls to pl.pallas_call land debug=True even
    when the caller didn't pass it. This is the wrapper's whole job."""
    captured: list[dict[str, Any]] = []

    def spy(*_args: Any, **kwargs: Any) -> str:
        captured.append(dict(kwargs))
        return "ok"

    monkeypatch.setattr(pl, "pallas_call", spy)

    with force_pallas_debug():
        pl.pallas_call(lambda: None, None)
    assert captured == [{"debug": True}]


def test_force_pallas_debug_overrides_explicit_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even if the kernel author hardcoded debug=False, the wrapper must
    win — the whole point of --dump-mosaic is to force the dump on."""
    captured: list[dict[str, Any]] = []

    def spy(*_args: Any, **kwargs: Any) -> str:
        captured.append(dict(kwargs))
        return "ok"

    monkeypatch.setattr(pl, "pallas_call", spy)

    with force_pallas_debug():
        pl.pallas_call(lambda: None, None, debug=False)
    assert captured == [{"debug": True}]


def test_force_pallas_debug_restores_original_on_normal_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch is reverted on clean exit so subsequent compare() variants
    don't keep paying for the dump (or, worse, see a stale closure)."""

    def spy(*_args: Any, **_kwargs: Any) -> str:
        return "ok"

    monkeypatch.setattr(pl, "pallas_call", spy)
    before = pl.pallas_call

    with force_pallas_debug():
        assert pl.pallas_call is not before
    assert pl.pallas_call is before


def test_force_pallas_debug_restores_original_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure inside .lower() must not leak the patch into the rest of
    the variants loop — otherwise one broken Pallas kernel would
    silently force-debug every subsequent one."""

    def spy(*_args: Any, **_kwargs: Any) -> str:
        return "ok"

    monkeypatch.setattr(pl, "pallas_call", spy)
    before = pl.pallas_call

    with pytest.raises(RuntimeError, match="boom"), force_pallas_debug():
        raise RuntimeError("boom")
    assert pl.pallas_call is before


def test_force_pallas_debug_targets_real_pallas_module() -> None:
    """The wrapper must patch the actual ``jax.experimental.pallas`` module
    used by kernels — not a re-export or a vendored copy. Kernels do
    ``import jax.experimental.pallas as pl`` and call ``pl.pallas_call``;
    attribute lookup happens at call time, so patching this module is
    what makes the injection visible to the kernel.

    If JAX moves ``pallas_call`` out from under that import path, this
    test fails loudly rather than producing an empty dump."""
    assert hasattr(pl, "pallas_call")
    original = pl.pallas_call
    with force_pallas_debug():
        # Inside the context, pl.pallas_call has been replaced.
        assert pl.pallas_call is not original
    assert pl.pallas_call is original
