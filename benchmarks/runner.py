"""Bench runner: warmup, timed iterations, summary stats.

Numbers come from `time.perf_counter` around `jax.block_until_ready`. JIT
compilation happens before the first timed iteration so it doesn't pollute
the median. Use `profile_dir` to drop an xprof trace alongside the run;
view it with `uv run xprof <profile_dir>`.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax


@dataclass(frozen=True)
class BenchResult:
    name: str
    times_s: list[float]  # per-iter wall times after warmup
    warmup_iters: int  # total untimed iterations (compile + warmup loop)
    timed_iters: int

    @property
    def median_s(self) -> float:
        return statistics.median(self.times_s)

    @property
    def p99_s(self) -> float:
        # quantiles requires n >= 2; fall back to max otherwise
        if len(self.times_s) < 2:
            return max(self.times_s)
        return statistics.quantiles(self.times_s, n=100)[98]

    @property
    def min_s(self) -> float:
        return min(self.times_s)

    @property
    def stdev_s(self) -> float:
        return statistics.stdev(self.times_s) if len(self.times_s) >= 2 else 0.0


def bench(
    name: str,
    fn: Callable[..., Any],
    args: Sequence[Any] = (),
    kwargs: dict[str, Any] | None = None,
    warmup: int = 5,
    iters: int = 20,
    profile_dir: str | None = None,
) -> BenchResult:
    """Run `fn(*args, **kwargs)` repeatedly and collect per-iter wall times.

    `fn` should already be `jax.jit`-wrapped. The first call compiles; we then
    do `warmup` untimed iterations, then `iters` timed iterations.
    """
    kwargs = kwargs or {}

    # Compile + warmup.
    out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    for _ in range(warmup):
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)

    # Timed loop, optionally inside a profiler trace.
    if profile_dir is not None:
        with jax.profiler.trace(profile_dir):
            times = _timed_loop(fn, args, kwargs, iters)
    else:
        times = _timed_loop(fn, args, kwargs, iters)

    # warmup + 1: the explicit compile call ahead of the warmup loop is also untimed.
    return BenchResult(name=name, times_s=times, warmup_iters=warmup + 1, timed_iters=iters)


def _timed_loop(
    fn: Callable[..., Any],
    args: Sequence[Any],
    kwargs: dict[str, Any],
    iters: int,
) -> list[float]:
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - start)
    return times
