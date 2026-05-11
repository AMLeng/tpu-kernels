"""Pallas-variant builder shared by ``compare()`` and ``sweep()``.

Both call sites bake a kernel's tile geometry in via
``functools.partial(pallas_fn, block_shape=...)``. The helpers here keep
that one rule in one place: every Pallas kernel benched in this repo
declares its tile geometry under the same parameter name
(``block_shape``), and the harness validates the spelling at
construction time so a typo (``block_size``, ``tile``, …) can't produce
a silent mis-bench.

Underscore-prefixed because this is harness-internal scaffolding;
``pallas_variant`` is re-exported from ``benchmarks.compare`` for the
import path suites already use.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

import jax


def _validate_block_shape_kwarg(pallas_fn: Callable[..., Any]) -> None:
    """Raise if ``pallas_fn`` has no ``block_shape`` parameter.

    Every Pallas kernel benched by ``compare()`` / ``sweep()`` declares its
    tile geometry under the same parameter name; the harness bakes the
    value in via ``functools.partial(pallas_fn, block_shape=...)``. A
    typo (``block_size``, ``tile``, etc.) would otherwise produce a silent
    mis-bench: a ``TypeError`` deep inside jax, or — worse — the kwarg
    landing in a kernel that happens to accept extras while the kernel
    quietly uses its own default. Surfacing the mismatch at call time
    keeps the diagnostic close to the typo.
    """
    params = inspect.signature(pallas_fn).parameters
    if "block_shape" not in params:
        raise ValueError(
            f"{getattr(pallas_fn, '__name__', repr(pallas_fn))} has no "
            "`block_shape` parameter; every Pallas kernel benched by "
            "compare()/sweep() must declare its tile geometry as "
            "`block_shape: tuple[int, ...]`."
        )


def pallas_variant(
    pallas_fn: Callable[..., Any],
    *,
    block_shape: tuple[int, ...],
) -> jax.stages.Wrapped:
    """Build a jit-wrapped, validated Pallas variant for a ``compare()``
    variants dict.

    Suites place the result directly alongside any other variants — there
    is no special parameter on ``compare()`` for Pallas, so a single
    compare() can bench arbitrarily many Pallas variants (e.g. ``pallas``
    vs ``pipelined`` for matmul) against any number of XLA baselines:

        compare(
            variants={
                "xla": matmul_xla,
                "pallas": pallas_variant(matmul_pallas, block_shape=(128, 128)),
                "pipelined": pallas_variant(matmul_pipelined, block_shape=(128, 128)),
            },
            ...
        )

    ``pallas_fn`` must declare ``block_shape`` (the repo-wide convention
    for Pallas kernels) — a typo would otherwise produce a silent
    mis-bench, either a TypeError deep inside jax or, worse, the kwarg
    landing in a kernel that happens to accept extras while it quietly
    uses its own default. The validation runs eagerly at the suite line
    that names the kernel, so the error points right at the typo.

    The returned callable is ``jax.jit(functools.partial(pallas_fn,
    block_shape=...))``: ``functools.partial`` binds ``block_shape`` as
    a closed-over Python static so it is not a traced input, and jax
    sees a fresh callable with a unique cache key.
    """
    _validate_block_shape_kwarg(pallas_fn)
    jitted = jax.jit(functools.partial(pallas_fn, block_shape=block_shape))
    # Marker read by compare() to gate the --dump-mosaic pass: only
    # Pallas variants produce Mosaic IR (the TPU lowering rule is what
    # emits it), so XLA entries are skipped to avoid printing a header
    # with no body.
    jitted._is_pallas_variant = True  # type: ignore[attr-defined]
    return jitted
