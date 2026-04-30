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


# Keep jax import live (avoids "imported but unused" if all tests above are
# removed during refactors; the runner module needs jax for type-annotation
# evaluation at import time and we want that path exercised here).
_ = jax.devices
