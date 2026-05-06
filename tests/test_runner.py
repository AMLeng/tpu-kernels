"""Tests for bench() runner: timing modes, chain semantics, k sizing.

Per CLAUDE.md, harness changes are TDD-only — these pin the new behavior
of bench() before the implementation lands. The XPlane parse used by
timing="device" needs a real profile artifact, so its tests live under
tests/perf/ (TPU-marked); on CPU we only assert the entry-level rejection.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from benchmarks.runner import (
    BenchResult,
    _build_unrolled,
    _choose_k,
    _find_xplane,
    _xla_module_events,
    bench,
)

# ---- _choose_k: pure sizing rule ----------------------------------------


def test_choose_k_picks_next_pow2_for_target() -> None:
    """t1 = 1ms, target = 10ms → ceil(10) = 10 → next_pow2 = 16."""
    assert _choose_k(t1_s=0.001, target_s=0.010, k_max=1024) == 16


def test_choose_k_clamps_to_k_max_for_very_fast_kernel() -> None:
    """t1 = 1µs against 10ms target wants k=10000; clamped to k_max=1024.

    Hitting the cap is a signal the kernel is too cheap to bench reliably
    via unroll — caller should switch to timing='device'. Clamp silently;
    the unroll field on BenchResult lets callers see they hit the cap.
    """
    assert _choose_k(t1_s=1e-6, target_s=0.010, k_max=1024) == 1024


def test_choose_k_returns_one_when_t1_exceeds_target() -> None:
    """One call already over the floor — no point amortizing further."""
    assert _choose_k(t1_s=0.020, target_s=0.010, k_max=1024) == 1


def test_choose_k_returns_one_when_t1_equals_target() -> None:
    """ceil(target/t1) = 1 at exact equality → k = 1."""
    assert _choose_k(t1_s=0.010, target_s=0.010, k_max=1024) == 1


def test_choose_k_picks_two_when_t1_is_half_target() -> None:
    """5ms call, 10ms target → ceil = 2 → next_pow2 = 2."""
    assert _choose_k(t1_s=0.005, target_s=0.010, k_max=1024) == 2


def test_choose_k_uses_documented_defaults() -> None:
    """Default target = 10ms, default k_max = 1024.

    t1 = 100µs → ceil(0.010 / 0.0001) = 100 → next_pow2 = 128.
    """
    assert _choose_k(t1_s=1e-4) == 128


# ---- _build_unrolled: chain semantics -----------------------------------


def test_build_unrolled_chain_int_zero_threads_output_to_first_arg() -> None:
    def add_one(x):
        return x + 1

    unrolled = _build_unrolled(add_one, k=4, chain=0)
    # 0 -> 1 -> 2 -> 3 -> 4
    assert float(unrolled(jnp.array(0.0))) == 4.0


def test_build_unrolled_chain_int_zero_keeps_remaining_args_constant() -> None:
    """With chain=0, only arg 0 gets replaced; arg 1 stays the same each iter."""

    def mul(a, b):
        return a * b

    unrolled = _build_unrolled(mul, k=3, chain=0)
    # iter 1: 2*3=6  iter 2: 6*3=18  iter 3: 18*3=54
    assert float(unrolled(jnp.array(2.0), jnp.array(3.0))) == 54.0


def test_build_unrolled_chain_int_picks_specified_arg() -> None:
    """chain=1 replaces arg 1; chain=0 replaces arg 0. Index-by-position."""

    def sub(a, b):
        return a - b

    unrolled = _build_unrolled(sub, k=3, chain=1)
    # iter 1: 10-1=9  iter 2: b=9 -> 10-9=1  iter 3: b=1 -> 10-1=9
    assert float(unrolled(jnp.array(10.0), jnp.array(1.0))) == 9.0


def test_build_unrolled_chain_str_threads_kwarg() -> None:
    def add(*, x, y):
        return x + y

    unrolled = _build_unrolled(add, k=3, chain="x")
    # iter 1: 0+2=2  iter 2: x=2 -> 4  iter 3: x=4 -> 6
    assert float(unrolled(x=jnp.array(0.0), y=jnp.array(2.0))) == 6.0


def test_build_unrolled_chain_callable_full_control() -> None:
    """Callable chain gets (prev_out, *args, **kwargs); returns (new_args, new_kwargs)."""

    def fn(a, b):
        return a + b

    def swap_b_with_prev(prev, *args, **kwargs):
        a, _b = args
        return (a, prev), kwargs

    unrolled = _build_unrolled(fn, k=3, chain=swap_b_with_prev)
    # iter 1: 1+0=1  iter 2: b<-1 -> 1+1=2  iter 3: b<-2 -> 1+2=3
    assert float(unrolled(jnp.array(1.0), jnp.array(0.0))) == 3.0


def test_build_unrolled_k_one_is_single_call() -> None:
    """k=1 reduces to fn(*args, **kwargs); chain unused."""

    def add_one(x):
        return x + 1

    unrolled = _build_unrolled(add_one, k=1, chain=0)
    assert float(unrolled(jnp.array(5.0))) == 6.0


def test_build_unrolled_does_not_collapse_chained_calls_in_compiled_hlo() -> None:
    """XLA must not fold k chained calls into fewer ops at compile time.

    Without an optimization barrier between chained calls, a chain like
    ``(((x+1)+1)+1)+1`` collapses to ``x+4`` in optimized HLO — one add op
    for k=4 calls. The unroll harness divides total wallclock by k to get
    per-call time, so a folded program reports per-call time as 1/k of the
    true single-call cost. Memory-bound kernels see this as BW% > 100%.

    Note this only pins the math-fusion case. The barrier doesn't force
    materialization through HBM, so an op whose working set fits in VMEM
    can still surface the same per-call accounting bug even with the
    barrier in place. The complementary mitigation is sizing inputs to
    several times VMEM; the suite-default tests in tests/test_*_suite.py
    pin that floor.
    """

    @jax.jit
    def add_one(x: jax.Array) -> jax.Array:
        return x + 1.0

    x = jnp.zeros((1024,), dtype=jnp.float32)
    k = 4
    unrolled = _build_unrolled(add_one, k=k, chain=0)
    hlo = jax.jit(unrolled).lower(x).compile().as_text()
    assert hlo is not None  # CPU backend returns text; defensive for pyright
    # Compiled HLO uses ``add(...)`` for f32 element-wise add. With the
    # optimization barrier between chained calls, each call's add survives
    # as a distinct op — at least k of them remain.
    add_ops = hlo.count(" add(")
    assert add_ops >= k, (
        f"compiled HLO has {add_ops} ' add(' op(s); expected >= {k} for "
        f"k={k} chained add_one calls. XLA folded the chain — per-call "
        "timing under unroll mode would be inflated by ~k times."
    )


# ---- bench() entry-level validation -------------------------------------


def test_bench_chain_shape_mismatch_raises_before_loop() -> None:
    """fn output shape ≠ chained slot shape → fail-fast at bench entry.

    Trace-time check via jax.eval_shape so we never enter the timed loop
    with a misconfigured chain.
    """

    def reduce_to_scalar(x):
        return jnp.sum(x)

    with pytest.raises(ValueError, match=r"(?i)chain.*shape|shape.*chain"):
        bench(
            "test",
            reduce_to_scalar,
            [jnp.zeros(10)],
            timing="unroll",
            warmup=0,
            iters=1,
        )


def test_bench_chain_int_rejects_tuple_output() -> None:
    """Multi-output kernels need a callable chain; int can't disambiguate."""

    def two_outputs(x):
        return x, x + 1

    with pytest.raises(ValueError, match=r"(?i)tuple|multi.*output|callable"):
        bench(
            "test",
            two_outputs,
            [jnp.zeros(10)],
            timing="unroll",
            warmup=0,
            iters=1,
        )


def test_bench_device_timing_with_explicit_chain_raises() -> None:
    """chain has no meaning under device timing — surface intent mismatch."""
    with pytest.raises(ValueError, match=r"(?i)device.*chain|chain.*device"):
        bench(
            "test",
            lambda x: x,
            [jnp.zeros(10)],
            timing="device",
            chain=0,
            warmup=0,
            iters=1,
        )


def test_bench_device_timing_on_cpu_raises() -> None:
    """timing='device' relies on TPU XPlane parsing; refuse on CPU rather than fall back silently."""  # noqa: E501
    with pytest.raises(RuntimeError, match=r"(?i)device.*tpu|tpu.*device|requires.*tpu"):
        bench(
            "test",
            lambda x: x,
            [jnp.zeros(10)],
            timing="device",
            warmup=0,
            iters=1,
        )


# ---- bench() output: BenchResult shape ----------------------------------


def test_bench_unroll_smoke_cpu_returns_populated_result() -> None:
    """End-to-end on CPU: timing='unroll' returns a BenchResult with new fields set."""

    def add_one(x):
        return x + 1

    result = bench(
        "smoke",
        add_one,
        [jnp.zeros(100)],
        timing="unroll",
        warmup=1,
        iters=3,
    )
    assert result.timing == "unroll"
    assert result.unroll >= 1
    assert len(result.times_s) == 3
    assert all(t >= 0 for t in result.times_s)


def test_bench_result_back_compat_defaults() -> None:
    """BenchResult(name, times_s, warmup_iters, timed_iters) still works.

    Existing test fixtures (test_compare.py, test_sweep.py) build
    BenchResult directly with the original four fields. Defaults on the
    new fields keep those call sites working without churn.
    """
    br = BenchResult(name="x", times_s=[1.0], warmup_iters=1, timed_iters=1)
    assert br.unroll == 1
    assert br.timing == "unroll"
    assert br.cluster_mismatch is False


def test_bench_unroll_chooses_k_above_one_for_fast_kernel_on_cpu() -> None:
    """Fast kernel (small array, simple op) → sizing rule should pick k > 1.

    Asserts the sizing path is actually wired into bench(), not just
    living in _choose_k untouched.
    """

    def cheap(x):
        return x + 1

    result = bench(
        "cheap",
        cheap,
        [jnp.zeros(8)],
        timing="unroll",
        warmup=1,
        iters=3,
    )
    assert result.unroll > 1


# ---- device-timing parse helpers (CPU-runnable) ------------------------


def test_find_xplane_locates_pb_in_subdir(tmp_path) -> None:
    """jax.profiler.trace writes under plugins/profile/<run>/<host>.xplane.pb;
    finder must walk the tree."""
    inner = tmp_path / "plugins" / "profile" / "20260101_120000"
    inner.mkdir(parents=True)
    pb = inner / "host.xplane.pb"
    pb.write_bytes(b"")
    found = _find_xplane(str(tmp_path))
    assert found == str(pb)


def test_find_xplane_raises_when_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=r"(?i)xplane"):
        _find_xplane(str(tmp_path))


# ---- _xla_module_events: per-program-execution durations -----------------


class _FakeEvent:
    def __init__(self, start_ns: int, end_ns: int, duration_ns: int) -> None:
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.duration_ns = duration_ns


class _FakeLine:
    def __init__(self, name: str, events: list[_FakeEvent]) -> None:
        self.name = name
        self.events = events


class _FakePlane:
    def __init__(self, name: str, lines: list[_FakeLine]) -> None:
        self.name = name
        self.lines = lines


class _FakePD:
    def __init__(self, planes: list[_FakePlane]) -> None:
        self.planes = planes


def test_xla_module_events_returns_one_event_per_program_execution() -> None:
    """Each XLA Modules event represents one program execution; its
    ``duration_ns`` is the wall time from launch to completion. Reading
    durations off this line gives one event per ``iters``, no nesting,
    no clustering needed.

    Regression for the cumsum_xla_scan investigation: programs that
    emit a control-flow op like ``%while = ...`` (e.g. ``lax.scan``)
    produce a parent XLA Op event covering the loop AND inner per-iter
    XLA Op events covering each step, with the parent's duration
    encompassing the children. Summing XLA Ops durations double-counts
    the parent + children; XLA Modules has no nesting so the sum is
    the truth.

    The fake plane below mirrors that case: XLA Modules has 60ms per
    execution; XLA Ops would sum to 120ms because the outer ``while``
    and its inner ops are both recorded with full durations. The
    function under test ignores XLA Ops entirely.
    """
    pd = _FakePD(
        [
            _FakePlane(
                "/device:TPU:0",
                [
                    _FakeLine(
                        "XLA Modules",
                        [
                            _FakeEvent(0, 60_000_000, 60_000_000),
                            _FakeEvent(70_000_000, 130_000_000, 60_000_000),
                        ],
                    ),
                    _FakeLine(
                        "XLA Ops",
                        [
                            _FakeEvent(0, 60_000_000, 60_000_000),  # outer while #0
                            _FakeEvent(1_000_000, 31_000_000, 30_000_000),  # inside #0
                            _FakeEvent(31_000_000, 60_000_000, 29_000_000),  # inside #0
                            _FakeEvent(70_000_000, 130_000_000, 60_000_000),  # outer #1
                            _FakeEvent(71_000_000, 101_000_000, 30_000_000),  # inside #1
                            _FakeEvent(101_000_000, 130_000_000, 29_000_000),  # inside #1
                        ],
                    ),
                ],
            )
        ]
    )
    assert _xla_module_events(pd) == [
        (0, 60_000_000, 60_000_000),
        (70_000_000, 130_000_000, 60_000_000),
    ]


def test_xla_module_events_skips_planes_without_xla_modules_line() -> None:
    """A TPU plane present but with no 'XLA Modules' line yields []."""
    pd = _FakePD([_FakePlane("/device:TPU:0", [_FakeLine("XLA Ops", [_FakeEvent(0, 10, 10)])])])
    assert _xla_module_events(pd) == []


def test_xla_module_events_returns_empty_when_no_tpu_plane() -> None:
    pd = _FakePD([_FakePlane("/host:CPU", [_FakeLine("XLA Modules", [_FakeEvent(0, 10, 10)])])])
    assert _xla_module_events(pd) == []


def test_xla_module_events_raises_on_multiple_tpu_planes_with_events() -> None:
    """Sharded execution would record events on every chip's plane;
    silently picking one would understate BW% by 1/N."""
    pd = _FakePD(
        [
            _FakePlane("/device:TPU:0", [_FakeLine("XLA Modules", [_FakeEvent(0, 10, 10)])]),
            _FakePlane("/device:TPU:1", [_FakeLine("XLA Modules", [_FakeEvent(0, 10, 10)])]),
        ]
    )
    with pytest.raises(RuntimeError, match=r"(?i)multiple.*tpu|sharded"):
        _xla_module_events(pd)


# Keep jax import live (avoids "imported but unused" if all tests above are
# removed during refactors; the runner module needs jax for type-annotation
# evaluation at import time and we want that path exercised here).
_ = jax.devices
