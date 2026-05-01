"""Compare N variants of an op against the same hardware roofline.

Usage from a suite file:

    from benchmarks.compare import compare
    from benchmarks.roofline import v5e

    compare(
        op="rmsnorm",
        variants={"xla": xla_fn, "pallas": pallas_fn},
        args=(x,),
        flops=count_flops(x),
        nbytes=count_bytes(x),
        hw=v5e(),
    )

Each variant is jit-compiled, benched with the runner, and analyzed against
the same roofline. Output is a table to stdout plus a JSON record under
bench_history/<op>/. With `dump_hlo=True`, prints the lowered HLO per variant.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Literal

import jax

from benchmarks import _history
from benchmarks.roofline import (
    FlopDtype,
    HardwarePeak,
    Roofline,
    analyze,
    arithmetic_intensity,
    peak_flops_for,
    regime,
    v5e,
)
from benchmarks.runner import BenchResult, bench

# Type alias for the per-variant tuple carried through the table/history helpers.
BenchRow = tuple[str, BenchResult, Roofline]


def _check_ici_consistency(hw: HardwarePeak, ici_bytes: int) -> None:
    """Warn if `ici_bytes > 0` was passed alongside a single-chip `hw`.

    ICI is the cross-chip interconnect; on `num_chips == 1` there's no traffic
    to measure, so a nonzero value is almost always a copy-paste error from a
    multi-chip suite. We warn instead of raising because someone explicitly
    modeling a what-if (e.g. `v5e(num_chips=1)` with hand-set ICI bytes) is a
    legitimate, if rare, exploration.
    """
    if ici_bytes > 0 and hw.num_chips == 1:
        # stacklevel=3 walks past `_check_ici_consistency` (frame 1) and the
        # `compare()` body (frame 2) so the warning points at the user's
        # `compare(...)` callsite — the only place they can fix the typo.
        warnings.warn(
            f"ici_bytes={ici_bytes} but num_chips == 1; "
            "ICI traffic only makes sense on multi-chip runs.",
            UserWarning,
            stacklevel=3,
        )


def compare(
    op: str,
    variants: dict[str, Callable[..., Any]],
    args: Sequence[Any] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    flops: int,
    nbytes: int,
    hw: HardwarePeak | None = None,
    flop_dtype: FlopDtype = "bf16",
    ici_bytes: int = 0,
    warmup: int = 5,
    iters: int = 20,
    dump_hlo: bool = False,
    profile_dir: str | None = None,
    timing: Literal["unroll", "device"] = "unroll",
    write_history: bool = True,
    config: dict[str, Any] | None = None,
) -> dict[str, Roofline]:
    """Bench each variant and print a roofline comparison table.

    Returns a dict mapping variant name -> Roofline result, also writes a
    timestamped JSON record to bench_history/<op>/ if write_history is True.

    `config` is stamped into each JSON record alongside the variant rows. Use
    it for per-call tuning knobs (e.g. `block_shape`) so a sweep can keep
    stable variant names and still be queryable across runs.
    """
    hw = hw if hw is not None else v5e()
    kwargs = kwargs or {}
    _check_ici_consistency(hw, ici_bytes)

    # Wrap each variant in jax.jit once; re-using the same wrapper across the
    # dump-HLO and bench passes also reuses JAX's compilation cache.
    jitted_variants: dict[str, Any] = {
        name: fn if _is_jitted(fn) else jax.jit(fn) for name, fn in variants.items()
    }

    if dump_hlo:
        for name, jitted in jitted_variants.items():
            print(f"\n----- HLO: {op}::{name} -----")
            try:
                hlo = jitted.lower(*args, **kwargs).compile().as_text()
                print(hlo)
            except Exception as e:
                print(f"(failed to lower {name}: {e})")

    results: dict[str, Roofline] = {}
    bench_results: list[BenchRow] = []
    for name, jitted in jitted_variants.items():
        br = bench(
            name=f"{op}::{name}",
            fn=jitted,
            args=args,
            kwargs=kwargs,
            warmup=warmup,
            iters=iters,
            profile_dir=profile_dir,
            timing=timing,
        )
        roof = analyze(
            flops=flops,
            nbytes=nbytes,
            seconds=br.median_s,
            hw=hw,
            flop_dtype=flop_dtype,
            ici_bytes=ici_bytes,
        )
        results[name] = roof
        bench_results.append((name, br, roof))

    _print_table(op, hw, flops, nbytes, ici_bytes, bench_results, flop_dtype)

    if write_history:
        _write_history(op, hw, flops, nbytes, flop_dtype, ici_bytes, bench_results, config)

    return results


def _is_jitted(fn: Callable[..., Any]) -> bool:
    """True if `fn` is the result of `jax.jit`.

    `jax.stages.Wrapped` is JAX's public runtime-checkable Protocol for
    jit-compiled callables — checking against it binds us to a documented
    attribute set instead of duck-typing on a chosen field like `lower`.
    """
    return isinstance(fn, jax.stages.Wrapped)


def _print_table(
    op: str,
    hw: HardwarePeak,
    flops: int,
    nbytes: int,
    ici_bytes: int,
    bench_results: list[BenchRow],
    flop_dtype: FlopDtype,
) -> None:
    arith_intensity = arithmetic_intensity(flops, nbytes)
    peak = peak_flops_for(hw, flop_dtype)
    ridge = (peak / hw.total_hbm_bw) if hw.total_hbm_bw else float("inf")
    regime_label = regime(arith_intensity, ridge)
    show_ici = ici_bytes > 0

    print(f"\n=== {op} ({hw.num_chips} chip{'s' if hw.num_chips > 1 else ''}) ===")
    print(f"  flops={flops:.3e}  bytes={nbytes:.3e}  intensity={arith_intensity:.1f} F/B")
    if show_ici:
        print(f"  ici_bytes={ici_bytes:.3e}  ici_peak={hw.total_ici_bw:.3e} B/s")
    print(f"  ridge point (peak_flops/peak_bw) = {ridge:.1f} F/B  →  {regime_label}")
    print()
    header = f"  {'variant':<20} {'med(ms)':>9} {'p99(ms)':>9} {'MFU%':>7} {'BW%':>7}"
    if show_ici:
        header += f" {'ICI%':>7}"
    header += f" {'binds':>8} {'SoL%':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, br, roof in bench_results:
        row = (
            f"  {name:<20} "
            f"{br.median_s * 1e3:>9.3f} "
            f"{br.p99_s * 1e3:>9.3f} "
            f"{roof.mfu * 100:>6.1f}% "
            f"{roof.bw_util * 100:>6.1f}%"
        )
        if show_ici:
            row += f" {roof.ici_util * 100:>6.1f}%"
        row += f" {roof.binds:>8} {roof.sol_pct * 100:>6.1f}%"
        print(row)
    print()


def _write_history(
    op: str,
    hw: HardwarePeak,
    flops: int,
    nbytes: int,
    flop_dtype: FlopDtype,
    ici_bytes: int,
    bench_results: list[BenchRow],
    config: dict[str, Any] | None = None,
) -> None:
    op_dir = _history.HISTORY_DIR / op
    op_dir.mkdir(parents=True, exist_ok=True)
    # Microsecond resolution: a sweep that fires two compare() calls inside one
    # second must not silently overwrite the earlier record's JSON.
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    record: dict[str, Any] = {
        "op": op,
        # ``kind`` lets a single aggregator read both compare and sweep records;
        # the sibling sweep writer stamps ``"sweep"`` for the same purpose.
        "kind": "compare",
        "timestamp": ts,
        "git_sha": _history.git_sha(),
        "hw": asdict(hw),
        "flops": flops,
        "bytes": nbytes,
        "ici_bytes": ici_bytes,
        "flop_dtype": flop_dtype,
        "config": config or {},
        "variants": {
            name: {
                "median_s": br.median_s,
                "p99_s": br.p99_s,
                "min_s": br.min_s,
                "stdev_s": br.stdev_s,
                # ``timing`` distinguishes wallclock-amortized (unroll) from
                # device-clock (XPlane) numbers; ``unroll`` exposes the k that
                # was used and surfaces a hit on the k_max cap.
                # ``cluster_mismatch`` is device-mode-only; True iff the
                # XPlane parse split iters into a different number of
                # clusters than expected (gap heuristic miscounted).
                "timing": br.timing,
                "unroll": br.unroll,
                "cluster_mismatch": br.cluster_mismatch,
                "mfu": roof.mfu,
                "bw_util": roof.bw_util,
                "ici_util": roof.ici_util,
                "sol_pct": roof.sol_pct,
                "binds": roof.binds,
            }
            for name, br, roof in bench_results
        },
    }
    out = op_dir / f"{ts}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"  wrote {out.relative_to(_history.REPO_ROOT)}")
