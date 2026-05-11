"""End-to-end smoke for ``--dump-mosaic``. TPU-only.

This is the real regression catcher for the dump mechanism. The CPU
unit tests in ``tests/test_mosaic_dump.py`` verify our monkey-patch
forces ``debug=True`` on ``pl.pallas_call``, but they don't exercise
JAX's lowering — Mosaic IR only prints under the TPU lowering rule,
which CPU runs (including ``interpret=True``) bypass.

If JAX renames the ``debug`` kwarg, drops the print, or changes the
marker string, this test fails and points at the dump path.
"""

from __future__ import annotations

import contextlib
import io

import jax
import jax.numpy as jnp
import pytest
from benchmarks.compare import compare, pallas_variant
from benchmarks.workload import Workload

from tpu_kernels.ops.scale import scale_pallas

pytestmark = [pytest.mark.tpu, pytest.mark.perf]

# Marker string JAX prints from the TPU lowering rule when debug=True.
# See jax/_src/pallas/mosaic/pallas_call_registration.py:433 (jax 0.10.0).
# If JAX renames or removes this, the test fails and we know the dump
# mechanism needs revisiting.
MOSAIC_MARKER = "The Mosaic module for pallas_call"


def test_dump_mosaic_prints_lowered_mosaic_ir() -> None:
    """compare(..., dump_mosaic=True) must produce stdout containing the
    Mosaic-module marker for a real Pallas variant."""
    x = jax.random.normal(jax.random.key(0), (256, 256), dtype=jnp.bfloat16)
    workload = Workload(op="scale_dump_smoke", flops=1, nbytes=1, args=(x,), flop_dtype="bf16")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compare(
            workload,
            variants={"pallas": pallas_variant(scale_pallas, block_shape=(128, 128))},
            dump_mosaic=True,
            write_history=False,
        )
    output = buf.getvalue()
    assert MOSAIC_MARKER in output, (
        f"expected {MOSAIC_MARKER!r} in --dump-mosaic stdout but did not find it; "
        "JAX's debug=True dump path likely changed. Captured tail:\n" + output[-2000:]
    )
