"""``--dump-mosaic`` plumbing: force ``pl.pallas_call(debug=True)`` for
the duration of a ``.lower(...).compile()`` pass.

JAX 0.10.0 prints the kernel jaxpr and lowered Mosaic module to stdout
from inside the TPU lowering rule, gated on the ``debug`` kwarg of
``pl.pallas_call`` (no env-var equivalent). Every Pallas variant in
this repo calls ``pl.pallas_call(...)`` at call time, so monkey-patching
``jax.experimental.pallas.pallas_call`` to inject ``debug=True`` makes
the dump fire without threading the kwarg through every kernel signature.

The TPU smoke test under ``tests/perf/`` is what catches a JAX-side
change to this mechanism — if Mosaic stops printing on ``debug=True``,
or the marker string changes, the smoke test fails and points here.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import jax.experimental.pallas as pl


@contextlib.contextmanager
def force_pallas_debug() -> Iterator[None]:
    """Within the ``with`` block, every ``pl.pallas_call`` invocation
    receives ``debug=True`` regardless of what the caller passed.

    Restores the original on both clean exit and exception so a single
    broken variant in a compare() loop doesn't poison subsequent ones.
    """
    original = pl.pallas_call

    def patched(*args: Any, **kwargs: Any) -> Any:
        kwargs["debug"] = True
        return original(*args, **kwargs)

    pl.pallas_call = patched  # type: ignore[assignment]
    try:
        yield
    finally:
        pl.pallas_call = original  # type: ignore[assignment]
