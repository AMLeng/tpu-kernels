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
    # device-mode only: True iff the XLA Modules event count did not
    # divide evenly into ``timed_iters`` per-call buckets — either the
    # XPlane parse went sideways or modules-per-call varied across iters.
    # Field name kept for backward compat with persisted bench history
    # JSON. Surfaces a transient warning into the record so a downstream
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
) -> BenchResult:
    """Run ``fn(*args, **kwargs)`` repeatedly and collect per-iter times.

    With ``timing="unroll"`` (default), k is auto-sized so a single
    program execution takes at least ~10ms; per-call times in the result
    are wallclock divided by k. ``chain`` says how to thread the prior
    output back into ``fn``: ``int`` for positional arg, ``str`` for
    kwarg, callable ``(prev_out, *args, **kwargs) -> (new_args,
    new_kwargs)`` for full control. Default ``chain=0``.

    With ``timing="device"``, k is fixed at 1 and per-call times come
    from the TPU hardware clock by parsing the XPlane's ``XLA Modules``
    line and clustering events into per-call buckets — a single jit'd
    call may dispatch more than one XLA program (e.g. cumsum's
    triangular reduce_window runs as 3 sequential scan levels per
    call). ``chain`` has no effect there and passing it raises.
    TPU-only.
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
    """Single jit that calls ``fn`` k times, threading prior output via chain.

    Each chained value passes through ``jax.lax.optimization_barrier`` before
    being fed back. Without the barrier XLA folds chains like
    ``(((x+1)+1)+1)+1`` into ``x+k``, collapsing k calls' worth of work
    into one op — per-call wallclock then reads as 1/k of the true cost.

    The barrier blocks *math* fusion only; it does not force materialization
    through HBM. If the working set fits in VMEM, XLA can still keep
    intermediates on chip across chained calls and the same per-call
    HBM-accounting bug surfaces. Suites must size inputs to several times
    VMEM — see CLAUDE.md / Bench inputs.
    """

    @jax.jit
    def looped(*args: Any, **kwargs: Any) -> Any:
        out = fn(*args, **kwargs)
        for _ in range(k - 1):
            out = jax.lax.optimization_barrier(out)
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
) -> BenchResult:
    """Run iters timed calls inside jax.profiler.trace; read durations
    off the device clock by parsing the resulting XPlane.

    Single Python call per timed iter (no chaining), but a single call
    may dispatch more than one XLA program (e.g. cumsum's triangular
    reduce_window lowers to a 3-level sequential scan). Per-call
    duration is the sum of consecutive ``XLA Modules`` events grouped
    into ``iters`` buckets — modules don't nest, so the sum is exact.
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
        durations, mismatch = _parse_xplane_durations(capture_dir, expected=iters)
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
) -> tuple[list[float], bool]:
    """Find xplane.pb in profile_dir, return per-call durations (s) and a
    cluster-mismatch flag.

    Reads every TPU plane's ``XLA Modules`` line, clusters each plane's
    events into ``expected`` per-call buckets via ``_per_call_durations``
    (a jit'd Python call may dispatch more than one XLA program — cumsum's
    triangular reduce_window lowers to 3 sequential scan levels), then folds
    the planes into one per-call duration via ``_coalesce_planes`` (the
    per-call max across chips). On a sharded run each chip contributes a
    plane; on single-chip there is just one. ``mismatch`` is True when the
    parse can't be trusted — a plane's events don't divide ``expected``, or
    the chips disagree on event count; callers stash it on BenchResult so
    trend tooling can spot a run where the parse went sideways.
    """
    from jax.profiler import ProfileData

    xplane_path = _find_xplane(profile_dir)
    pd = ProfileData.from_file(xplane_path)
    per_plane_events = _xla_module_events(pd)
    if not per_plane_events:
        raise RuntimeError(
            f"no XLA Modules events in {xplane_path}; XPlane was empty or "
            "the TPU plane has a different name on this hardware."
        )
    durations, mismatch = _coalesce_planes(per_plane_events, expected)
    if mismatch:
        counts = [len(ev) for ev in per_plane_events]
        warnings.warn(
            f"timing='device': a TPU plane's event count (counts={counts} "
            f"across {len(per_plane_events)} plane(s)) doesn't divide evenly "
            f"by {expected} iters, so per-call clustering can't be trusted. "
            "cluster_mismatch=True on the result.",
            stacklevel=3,
        )
    return durations, mismatch


def _per_call_durations(
    events: list[_EventTuple],
    iters: int,
) -> tuple[list[float], bool]:
    """Cluster sequential XLA Modules events into per-Python-call durations.

    A single jit'd call may dispatch more than one XLA program. The
    common case is 1:1 with iters (every previously-benched op). The
    multi-program case appears when XLA's TPU rewriter splits a single
    HLO into several sequential programs — cumsum's default lowering
    (a triangular ``reduce_window``) becomes a 3-level sequential scan,
    so each Python call records 3 module events.

    Modules don't nest, so when ``len(events)`` divides evenly by
    ``iters`` we group every ``len(events) // iters`` consecutive
    events as one call and sum their ``duration_ns``. When the count
    doesn't divide, the parse can't be trusted (truly variable
    modules-per-call across iters, or a pathological xplane); in that
    case return the raw per-event durations and flag ``mismatch=True``
    so the caller can warn and downstream tooling can drop the run.
    """
    n = len(events)
    if iters > 0 and n > 0 and n % iters == 0:
        per_call = n // iters
        durations = [
            sum(events[i * per_call + j][2] for j in range(per_call)) * 1e-9 for i in range(iters)
        ]
        return durations, False
    return [e[2] * 1e-9 for e in events], True


def _coalesce_planes(
    per_plane_events: list[list[_EventTuple]],
    iters: int,
) -> tuple[list[float], bool]:
    """Collapse N chip planes' event streams into one per-call duration list.

    Each chip runs the same SPMD program, but chips do *not* record the same
    number of module events: a collective makes the coordinator chip emit
    extra setup/reshard modules around the all-reduce (observed
    ``[60,20,20,20]`` events at iters=20 for sharded_matmul — 3 modules/call
    on chip 0 vs 1 on the others). So we cluster each plane independently
    with ``_per_call_durations`` against ``iters`` — different per-call module
    counts are fine, every plane still yields ``iters`` per-call durations —
    then report the per-call max across planes. The slowest chip bounds the
    cluster and gates any collective. Only per-call durations are compared
    across planes, never absolute timestamps, so plane clocks need not share
    an origin.

    ``mismatch=True`` (with an untrustworthy degraded payload) when a plane's
    event count doesn't divide evenly by ``iters`` — that plane can't be
    bucketed, so modules-per-call genuinely varied across iters or the
    xplane is pathological. Mirrors the single-plane fallback: return that
    plane's raw per-event durations so downstream trend tooling can drop the
    run rather than consume wrong numbers. Empty input yields ``([], False)``;
    the caller decides whether an empty profile is an error.
    """
    if not per_plane_events:
        return [], False

    per_plane_durations: list[list[float]] = []
    for events in per_plane_events:
        durations, mismatch = _per_call_durations(events, iters)
        if mismatch:
            return durations, True
        per_plane_durations.append(durations)

    # Every plane clustered to ``iters`` durations → elementwise max per call.
    coalesced = [max(per_call) for per_call in zip(*per_plane_durations, strict=True)]
    return coalesced, False


def _find_xplane(profile_dir: str) -> str:
    """Recursive scan for *.xplane.pb; on multiple matches, return the newest.

    ``jax.profiler.trace`` writes under ``plugins/profile/<run>/<host>.xplane.pb``
    so we walk rather than hardcode the layout. When ``profile_dir`` is a
    re-used path that accumulates xplanes across bench runs, returning
    whatever ``os.walk`` surfaces first means a stale xplane can shadow
    the one this bench just wrote (filesystem order, not write-time order).
    Picking by mtime pins the parse to the freshest trace, which is what
    every caller actually wants.
    """
    candidates: list[str] = []
    for root, _dirs, files in os.walk(profile_dir):
        for f in files:
            if f.endswith(".xplane.pb"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise FileNotFoundError(f"no *.xplane.pb under {profile_dir}")
    return max(candidates, key=os.path.getmtime)


def _events_from_line(pd: Any, line_name: str) -> list[list[_EventTuple]]:
    """Collect events per TPU plane from a named line.

    Returns one event list per TPU plane that has events on ``line_name``,
    ordered by plane name so downstream coalescing is deterministic. Empty
    outer list when no TPU plane has events. Each inner list holds that
    plane's events in trace order.

    A sharded run records the same SPMD program on every chip, so a
    multi-chip profile yields one stream per chip; ``_coalesce_planes``
    collapses them into one per-call duration. Host planes (multi-host) are
    skipped.
    """
    found: list[tuple[str, list[_EventTuple]]] = []
    for plane in pd.planes:
        # JAX names device planes "/device:TPU:N"; tolerate any plane that
        # advertises TPU. Multi-host adds host planes we explicitly skip.
        if "TPU" not in plane.name:
            continue
        for line in plane.lines:
            if line.name != line_name:
                continue
            events = [(int(e.start_ns), int(e.end_ns), int(e.duration_ns)) for e in line.events]
            if events:
                found.append((plane.name, events))
    # Sort by plane name ("/device:TPU:0", ":1", ...) for a stable order;
    # the coalesce step only compares per-call durations, but a deterministic
    # ordering keeps the mismatch fallback ("longest plane") reproducible.
    found.sort(key=lambda name_events: name_events[0])
    return [events for _name, events in found]


def _xla_module_events(pd: Any) -> list[list[_EventTuple]]:
    """Per-program-execution events from every TPU plane's XLA Modules line.

    One inner list per chip plane (see ``_events_from_line``). Each event is
    one full XLA program execution: ``start_ns`` is launch, ``end_ns`` is
    completion, ``duration_ns`` is the wall time of that execution. The line
    has no nesting, but a single jit'd Python call may dispatch more than one
    program — cumsum's triangular ``reduce_window`` is rewritten on TPU into
    a 3-level sequential scan that surfaces here as 3 events per call.
    ``_per_call_durations`` clusters consecutive events back into one
    duration per Python call; ``_coalesce_planes`` then folds the planes
    together. The XLA Ops line is sibling-and-children HLO ops; reading
    durations from it double-counts when the program contains an HLO
    ``while`` (e.g. ``lax.scan``), since the parent op's duration covers its
    children's.
    """
    return _events_from_line(pd, "XLA Modules")


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
