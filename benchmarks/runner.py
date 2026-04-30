"""Bench runner: warmup, timed iterations, summary stats.

Two timing modes:

* ``timing="unroll"`` (default) — builds a single jit that calls ``fn`` k
  times, threading the previous output back as input via ``chain``. The
  k is auto-sized so one program execution takes at least ~10ms, which
  amortizes host-side dispatch overhead (~150 µs/call) below noise. Per-
  call times in the result are wallclock divided by k.
* ``timing="device"`` — measures kernel time directly off the TPU
  hardware clock by parsing the XPlane recorded inside
  ``jax.profiler.trace``. Single call per timed iter (no chaining).
  TPU-only; raises on CPU rather than fall back silently.

Use ``profile_dir`` to drop an xprof trace alongside the run; view it
with ``uv run xprof <profile_dir>``.
"""

from __future__ import annotations

import itertools
import math
import os
import shutil
import statistics
import tempfile
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax

# Sentinel: distinguishes "user did not pass chain" from "user passed chain=0".
# Exposed only to bench(); _build_unrolled etc. take a real ChainSpec.
_CHAIN_DEFAULT: Any = object()

ChainSpec = int | str | Callable[..., tuple[tuple[Any, ...], dict[str, Any]]]


@dataclass(frozen=True)
class BenchResult:
    name: str
    times_s: list[float]  # PER-CALL times (already divided by unroll)
    warmup_iters: int  # total untimed iterations (compile + warmup loop)
    timed_iters: int
    unroll: int = 1  # k actually used; 1 under timing="device"
    timing: Literal["unroll", "device"] = "unroll"
    # device-mode only: True iff the XPlane parse produced a different
    # number of clusters than ``timed_iters``. Surfaces a transient
    # warning into the persisted bench history record so a downstream
    # trend tool can flag the run rather than have it slide by silently.
    cluster_mismatch: bool = False

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
    *,
    timing: Literal["unroll", "device"] = "unroll",
    chain: Any = _CHAIN_DEFAULT,
    gap_ns: int | None = None,
) -> BenchResult:
    """Run ``fn(*args, **kwargs)`` repeatedly and collect per-iter times.

    With ``timing="unroll"`` (default), k is auto-sized so a single
    program execution takes at least ~10ms; per-call times in the result
    are wallclock divided by k. ``chain`` says how to thread the prior
    output back into ``fn``: ``int`` for positional arg, ``str`` for
    kwarg, callable ``(prev_out, *args, **kwargs) -> (new_args,
    new_kwargs)`` for full control. Default ``chain=0``.

    With ``timing="device"``, k is fixed at 1 and per-call times come
    from the TPU hardware clock (XPlane parse). ``chain`` has no effect
    and passing it raises. TPU-only. ``gap_ns`` overrides the inter-event
    gap (ns) above which two XLA Ops events are treated as separate
    program executions; default tracks ``_EXECUTION_GAP_NS`` (100µs).
    Bump it for kernels with intra-program idle (unfused multi-stage
    HLO, async collective waits) that splits a single iter into multiple
    clusters.
    """
    kwargs = kwargs or {}

    if timing == "device":
        if chain is not _CHAIN_DEFAULT:
            raise ValueError(
                "timing='device' measures kernel time directly off the device "
                "clock; chain= has no effect there. Pass chain= only with "
                "timing='unroll'."
            )
        if not _has_tpu():
            raise RuntimeError(
                "timing='device' requires a TPU backend (XPlane parsing). "
                "On CPU, use timing='unroll' (the default)."
            )
        return _bench_device(
            name,
            fn,
            args,
            kwargs,
            warmup=warmup,
            iters=iters,
            profile_dir=profile_dir,
            gap_ns=gap_ns if gap_ns is not None else _EXECUTION_GAP_NS,
        )

    if gap_ns is not None:
        raise ValueError(
            "gap_ns controls the device-mode XPlane cluster boundary; it "
            "has no meaning under timing='unroll'. Pass gap_ns only with "
            "timing='device'."
        )
    chain_spec: ChainSpec = 0 if chain is _CHAIN_DEFAULT else chain
    return _bench_unroll(
        name,
        fn,
        args,
        kwargs,
        warmup=warmup,
        iters=iters,
        profile_dir=profile_dir,
        chain=chain_spec,
    )


# ---- unroll path ---------------------------------------------------------


def _bench_unroll(
    name: str,
    fn: Callable[..., Any],
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    warmup: int,
    iters: int,
    profile_dir: str | None,
    chain: ChainSpec,
) -> BenchResult:
    _validate_chain(fn, args, kwargs, chain)

    # Sizing run at k=1 to estimate per-call cost.
    sized = _build_unrolled(fn, k=1, chain=chain)
    jax.block_until_ready(sized(*args, **kwargs))  # compile
    sizing_iters = 3
    sizing_times = []
    for _ in range(sizing_iters):
        start = time.perf_counter()
        out = sized(*args, **kwargs)
        jax.block_until_ready(out)
        sizing_times.append(time.perf_counter() - start)
    t1 = statistics.median(sizing_times)
    k = _choose_k(t1_s=t1)

    # If the sizing run already amortizes enough (k=1), reuse the compiled fn.
    unrolled = sized if k == 1 else _build_unrolled(fn, k=k, chain=chain)
    if k != 1:
        jax.block_until_ready(unrolled(*args, **kwargs))  # compile
    for _ in range(warmup):
        jax.block_until_ready(unrolled(*args, **kwargs))

    if profile_dir is not None:
        with jax.profiler.trace(profile_dir):
            unroll_times = _timed_loop(unrolled, args, kwargs, iters)
    else:
        unroll_times = _timed_loop(unrolled, args, kwargs, iters)

    per_call_times = [t / k for t in unroll_times]
    # Untimed iters: 1 (k=1 compile) + sizing_iters + (1 (k>1 compile) if k != 1 else 0) + warmup.
    extra_compile = 0 if k == 1 else 1
    total_warmup = 1 + sizing_iters + extra_compile + warmup
    return BenchResult(
        name=name,
        times_s=per_call_times,
        warmup_iters=total_warmup,
        timed_iters=iters,
        unroll=k,
        timing="unroll",
    )


def _build_unrolled(
    fn: Callable[..., Any],
    k: int,
    chain: ChainSpec,
) -> Callable[..., Any]:
    """Single jit that calls ``fn`` k times, threading prior output via chain."""

    @jax.jit
    def looped(*args: Any, **kwargs: Any) -> Any:
        out = fn(*args, **kwargs)
        for _ in range(k - 1):
            args, kwargs = _apply_chain(out, args, kwargs, chain)
            out = fn(*args, **kwargs)
        return out

    return looped


def _apply_chain(
    prev: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    chain: ChainSpec,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if callable(chain):
        return chain(prev, *args, **kwargs)
    if isinstance(chain, int):
        new_args = list(args)
        new_args[chain] = prev
        return tuple(new_args), kwargs
    if isinstance(chain, str):
        return args, {**kwargs, chain: prev}
    raise TypeError(f"chain must be int | str | Callable; got {type(chain).__name__}")


def _validate_chain(
    fn: Callable[..., Any],
    args: Sequence[Any],
    kwargs: dict[str, Any],
    chain: ChainSpec,
) -> None:
    """Trace-time check that ``chain`` will work for ``fn``'s actual output."""
    if callable(chain):
        return  # caller takes responsibility; we can't statically check.

    out_struct = jax.eval_shape(fn, *args, **kwargs)
    if not isinstance(out_struct, jax.ShapeDtypeStruct):
        raise ValueError(
            f"fn returns a tuple/dict (got {type(out_struct).__name__}); "
            f"int/str chain={chain!r} can't pick which output to feed back. "
            "Pass a callable chain to specify."
        )

    if isinstance(chain, int):
        if chain < 0 or chain >= len(args):
            raise ValueError(f"chain={chain} but fn was called with {len(args)} positional args")
        target = args[chain]
        target_label = f"args[{chain}]"
    else:
        if chain not in kwargs:
            raise ValueError(f"chain={chain!r} but {chain!r} not in kwargs")
        target = kwargs[chain]
        target_label = f"kwargs[{chain!r}]"

    target_shape = getattr(target, "shape", None)
    target_dtype = getattr(target, "dtype", None)
    if target_shape != out_struct.shape or target_dtype != out_struct.dtype:
        raise ValueError(
            f"chain={chain!r}: fn output has shape={out_struct.shape} "
            f"dtype={out_struct.dtype}, but chained slot {target_label} has "
            f"shape={target_shape} dtype={target_dtype}. Pass a callable chain "
            "if the shape genuinely differs."
        )


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _choose_k(t1_s: float, target_s: float = 0.010, k_max: int = 1024) -> int:
    """Smallest power of 2 such that k * t1_s >= target_s, capped at ``k_max``.

    Hitting ``k_max`` is a signal the kernel is too cheap to bench
    reliably via unroll; the cap prevents pathological compile times for
    fast kernels. Inspect ``BenchResult.unroll`` to detect the cap.
    """
    if t1_s <= 0:
        return 1
    needed = math.ceil(target_s / t1_s)
    return min(_next_pow2(needed), k_max)


# ---- device path (TPU-only) ---------------------------------------------

# Inter-event gap (ns) above which we treat events as belonging to different
# program executions. Between bench iterations there's >>100µs of host idle
# (Python loop overhead alone is ~1ms); within a single fused HLO program
# events are back-to-back (sub-µs gaps). 100µs cleanly separates the two.
_EXECUTION_GAP_NS = 100_000

# (start_ns, end_ns, duration_ns).
_EventTuple = tuple[int, int, int]


def _bench_device(
    name: str,
    fn: Callable[..., Any],
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    warmup: int,
    iters: int,
    profile_dir: str | None,
    gap_ns: int,
) -> BenchResult:
    """Run iters timed calls inside jax.profiler.trace; read durations
    off the device clock by parsing the resulting XPlane.

    Single call per timed iter (no chaining): each iter is one program
    execution, recorded as one cluster of XLA Ops events on the TPU
    plane. Per-call duration is the sum of those events' durations.
    """
    # Compile + warmup outside the trace so first-call overhead doesn't
    # pollute the recorded events.
    jax.block_until_ready(fn(*args, **kwargs))
    for _ in range(warmup):
        jax.block_until_ready(fn(*args, **kwargs))

    user_supplied_dir = profile_dir is not None
    capture_dir = profile_dir or tempfile.mkdtemp(prefix="bench_device_")
    try:
        with jax.profiler.trace(capture_dir):
            for _ in range(iters):
                jax.block_until_ready(fn(*args, **kwargs))
        durations, mismatch = _parse_xplane_durations(capture_dir, expected=iters, gap_ns=gap_ns)
    finally:
        if not user_supplied_dir:
            # Cheap cleanup: only remove the temp dir we created. If the
            # parse raised, the caller still gets the exception; we just
            # don't litter /tmp.
            shutil.rmtree(capture_dir, ignore_errors=True)

    return BenchResult(
        name=name,
        times_s=durations,
        warmup_iters=warmup + 1,
        timed_iters=iters,
        unroll=1,
        timing="device",
        cluster_mismatch=mismatch,
    )


def _parse_xplane_durations(
    profile_dir: str,
    expected: int,
    *,
    gap_ns: int,
) -> tuple[list[float], bool]:
    """Find xplane.pb in profile_dir, return durations (s) and mismatch flag.

    ``mismatch`` is True iff the parsed cluster count != ``expected``;
    callers stash it on BenchResult so trend tooling can spot a run
    where the gap heuristic miscounted iters.
    """
    from jax.profiler import ProfileData

    xplane_path = _find_xplane(profile_dir)
    pd = ProfileData.from_file(xplane_path)
    events = _xla_ops_events(pd)
    if not events:
        raise RuntimeError(
            f"no XLA Ops events in {xplane_path}; XPlane was empty or the "
            "TPU plane has a different name on this hardware."
        )
    clusters = _cluster_events_by_gap(events, gap_ns=gap_ns)
    durations = _durations_from_clusters(clusters)
    mismatch = len(durations) != expected
    if mismatch:
        warnings.warn(
            f"timing='device': expected {expected} program executions, "
            f"got {len(durations)} clusters from XPlane. Bump gap_ns= or "
            "use timing='unroll'. cluster_mismatch=True on the result.",
            stacklevel=3,
        )
    return durations, mismatch


def _find_xplane(profile_dir: str) -> str:
    """Recursive scan for *.xplane.pb. jax.profiler.trace writes under
    plugins/profile/<run>/<host>.xplane.pb; we don't depend on the layout."""
    for root, _dirs, files in os.walk(profile_dir):
        for f in files:
            if f.endswith(".xplane.pb"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"no *.xplane.pb under {profile_dir}")


def _xla_ops_events(pd: Any) -> list[_EventTuple]:
    """Pull (start_ns, end_ns, duration_ns) for events from the *single* TPU
    plane's XLA Ops line. Returns [] if no TPU plane has events.

    Single-chip-only today: if events are present on more than one TPU
    plane we raise rather than silently pick one, since BW% derived from
    one chip's events would understate a sharded execution. Sharded
    suites need to coalesce across planes — defer until the first
    multi-chip suite lands.
    """
    found: list[tuple[str, list[_EventTuple]]] = []
    for plane in pd.planes:
        # JAX names device planes "/device:TPU:N"; tolerate any plane that
        # advertises TPU. Multi-host adds host planes we explicitly skip.
        if "TPU" not in plane.name:
            continue
        for line in plane.lines:
            if line.name != "XLA Ops":
                continue
            events = [(int(e.start_ns), int(e.end_ns), int(e.duration_ns)) for e in line.events]
            if events:
                found.append((plane.name, events))
    if not found:
        return []
    if len(found) > 1:
        names = [name for name, _ in found]
        raise RuntimeError(
            f"timing='device' found XLA Ops events on multiple TPU planes "
            f"({names}); sharded execution isn't supported yet. Run on a "
            "single chip or extend the parser to coalesce planes."
        )
    return found[0][1]


def _cluster_events_by_gap(
    events: Sequence[_EventTuple],
    gap_ns: int,
) -> list[list[_EventTuple]]:
    """Group events into per-execution clusters by inter-event gap.

    Sorts by start_ns first; a gap > ``gap_ns`` between prev_end and next_start
    closes the current cluster.
    """
    sorted_events = sorted(events, key=lambda e: e[0])
    if not sorted_events:
        return []
    clusters: list[list[_EventTuple]] = [[sorted_events[0]]]
    for prev, cur in itertools.pairwise(sorted_events):
        if cur[0] - prev[1] > gap_ns:
            clusters.append([cur])
        else:
            clusters[-1].append(cur)
    return clusters


def _durations_from_clusters(
    clusters: Sequence[Sequence[_EventTuple]],
) -> list[float]:
    """Per-cluster duration (seconds): sum of event durations.

    We sum event ``duration_ns`` (device-active time) rather than
    ``last_end - first_start``, which would also count idle gaps within
    a cluster — for fused HLO that difference is negligible but for
    multi-op programs it matters.
    """
    return [sum(e[2] for e in cluster) * 1e-9 for cluster in clusters]


def _has_tpu() -> bool:
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except RuntimeError:
        return False


# ---- shared timed loop ---------------------------------------------------


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
