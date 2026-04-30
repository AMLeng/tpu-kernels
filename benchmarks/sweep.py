"""Cartesian sweep across kernel configs — sibling to ``compare()``.

Where ``compare()`` is "N variants x 1 config" (used to pick between
implementations), ``sweep()`` is "1 variant x N configs" (used to tune a
single implementation). Both share the same JSON contract under
``bench_history/<op>/`` (each record stamped with ``"kind"``: ``"compare"``
or ``"sweep"``) so an aggregator can read either kind without sniffing.

Output is a perf-sorted leaderboard to stdout plus a single timestamped
JSON. Configs that fail an optional ``is_valid`` predicate are skipped
without bench (and without crashing the loop on a divisibility mismatch);
configs that raise during build or run are caught and reported as errored.
Both sets land in the JSON record so the recorded shape of the sweep
matches what was *attempted*, not just what ran.

Single-chip-only: this module does not thread ``ici_bytes`` through to
``analyze()``. Pallas block-shape tuning, the intended use case, runs on
one chip. A multi-chip sharding sweep would need to mirror ``compare``'s
``ici_bytes`` parameter; defer until the first such suite lands.
"""

from __future__ import annotations

import itertools
import json
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

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

# (config, name, BenchResult, Roofline) for each successfully-benched config.
SweepRow = tuple[dict[str, Any], str, BenchResult, Roofline]


def _check_single_chip(hw: HardwarePeak) -> None:
    """Warn if ``hw.num_chips != 1``.

    sweep() is single-chip-only today (Pallas block-shape tuning) and does not
    thread ``ici_bytes`` through to ``analyze()``. A multi-chip ``hw`` would
    print "N chips" in the table header while the rooflines remained
    single-chip — the silent-lie-in-the-harness pattern CLAUDE.md flags. We
    warn instead of raising in case someone is exploring what the table looks
    like under a different chip count, but the default path makes the
    mismatch visible.
    """
    if hw.num_chips != 1:
        # stacklevel=3 walks past `_check_single_chip` (frame 1) and the
        # `sweep()` body (frame 2) so the warning points at the user's
        # `sweep(...)` callsite — the only place they can fix the mistake.
        warnings.warn(
            f"sweep() is single-chip-only today; got num_chips={hw.num_chips}. "
            "Multi-chip sharding sweep is not yet implemented.",
            UserWarning,
            stacklevel=3,
        )


def sweep(
    op: str,
    variant_factory: Callable[..., jax.stages.Wrapped],
    axes: dict[str, Sequence[Any]],
    args: Sequence[Any] = (),
    *,
    flops: int,
    nbytes: int,
    hw: HardwarePeak | None = None,
    flop_dtype: FlopDtype = "bf16",
    is_valid: Callable[..., bool] | None = None,
    warmup: int = 5,
    iters: int = 20,
    write_history: bool = True,
) -> dict[str, Roofline]:
    """Bench ``variant_factory(**config)`` for every config in ``product(axes)``.

    ``axes`` maps axis name → list of values. Insertion order is preserved
    in variant names, table columns, and JSON keys, so the user's chosen
    axis order is the one they'll see everywhere.

    ``variant_factory(**config)`` must return a ``jax.jit``-wrapped callable.
    The factory pattern (rather than passing pre-built variants) exists so
    the suite controls the jit closure — typically capturing per-config
    constants like ``block_shape`` so they don't become traced inputs. The
    type annotation pins this contract; pyright catches a factory that
    returns a plain function.

    ``is_valid(**config) -> bool`` filters the product before bench. Configs
    that fail validation are recorded but not built; configs that raise
    during build or run are caught and reported. The sweep returns
    ``{name: Roofline}`` for the successfully-benched configs only.
    """
    hw = hw if hw is not None else v5e()
    _check_single_chip(hw)

    axis_names = list(axes.keys())
    axis_values = [list(axes[k]) for k in axis_names]
    all_configs = [
        dict(zip(axis_names, vals, strict=True)) for vals in itertools.product(*axis_values)
    ]

    valid_configs: list[dict[str, Any]] = []
    skipped_configs: list[dict[str, Any]] = []
    if is_valid is None:
        valid_configs = all_configs
    else:
        for cfg in all_configs:
            (valid_configs if is_valid(**cfg) else skipped_configs).append(cfg)

    rows: list[SweepRow] = []
    errored: list[tuple[dict[str, Any], str]] = []
    for cfg in valid_configs:
        name = _config_name(cfg)
        try:
            fn = variant_factory(**cfg)
            br = bench(name=f"{op}::{name}", fn=fn, args=args, warmup=warmup, iters=iters)
            roof = analyze(
                flops=flops,
                nbytes=nbytes,
                seconds=br.median_s,
                hw=hw,
                flop_dtype=flop_dtype,
            )
            rows.append((cfg, name, br, roof))
        except Exception as e:
            # Catch anything: a sweep that aborts on the first divisibility crash
            # or compile error wastes the whole tuning loop. Errored configs
            # land in the JSON so they're visible after the fact.
            errored.append((cfg, repr(e)))

    # Sort by SoL% descending — winner on top, marked in the table.
    rows.sort(key=lambda r: r[3].sol_pct, reverse=True)

    _print_sweep_table(
        op, hw, flops, nbytes, axis_names, rows, skipped_configs, errored, flop_dtype
    )

    if write_history:
        _write_sweep_history(
            op, hw, flops, nbytes, flop_dtype, axes, rows, skipped_configs, errored
        )

    return {name: roof for _, name, _, roof in rows}


def _config_name(cfg: dict[str, Any]) -> str:
    """``{"bm": 8, "bn": 128}`` → ``"bm=8,bn=128"``. Preserves insertion order."""
    return ",".join(f"{k}={v}" for k, v in cfg.items())


def _print_sweep_table(
    op: str,
    hw: HardwarePeak,
    flops: int,
    nbytes: int,
    axis_names: list[str],
    rows: list[SweepRow],
    skipped: list[dict[str, Any]],
    errored: list[tuple[dict[str, Any], str]],
    flop_dtype: FlopDtype,
) -> None:
    arith_intensity = arithmetic_intensity(flops, nbytes)
    peak = peak_flops_for(hw, flop_dtype)
    ridge = (peak / hw.total_hbm_bw) if hw.total_hbm_bw else float("inf")
    regime_label = regime(arith_intensity, ridge)

    chip_word = "chip" if hw.num_chips == 1 else "chips"
    total = len(rows) + len(skipped) + len(errored)
    print(f"\n=== {op} sweep ({hw.num_chips} {chip_word}) ===")
    print(f"  flops={flops:.3e}  bytes={nbytes:.3e}  intensity={arith_intensity:.1f} F/B")
    print(f"  ridge point = {ridge:.1f} F/B  →  {regime_label}")
    print(
        f"  {len(rows)} of {total} configs benched ({len(skipped)} skipped, {len(errored)} errored)"
    )
    print()

    if not rows:
        print("  (no valid configs)")
        _print_skipped_and_errored(skipped, errored)
        print()
        return

    axis_widths = {
        name: max(len(name), max(len(str(cfg[name])) for cfg, _, _, _ in rows))
        for name in axis_names
    }
    axis_header = " ".join(f"{name:>{axis_widths[name]}}" for name in axis_names)
    header = (
        f"   {axis_header}  {'med(ms)':>9} {'p99(ms)':>9} "
        f"{'MFU%':>7} {'BW%':>7} {'binds':>8} {'SoL%':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, (cfg, _, br, roof) in enumerate(rows):
        marker = "*" if i == 0 else " "
        axis_cells = " ".join(f"{cfg[name]!s:>{axis_widths[name]}}" for name in axis_names)
        print(
            f"  {marker}{axis_cells}  "
            f"{br.median_s * 1e3:>9.3f} {br.p99_s * 1e3:>9.3f} "
            f"{roof.mfu * 100:>6.1f}% {roof.bw_util * 100:>6.1f}% "
            f"{roof.binds:>8} {roof.sol_pct * 100:>6.1f}%"
        )

    best_cfg, _, _, best_roof = rows[0]
    print()
    print(f"  best: {_config_name(best_cfg)}  (SoL {best_roof.sol_pct * 100:.1f}%)  [* = best]")
    _print_skipped_and_errored(skipped, errored)
    print()


def _print_skipped_and_errored(
    skipped: list[dict[str, Any]],
    errored: list[tuple[dict[str, Any], str]],
) -> None:
    """Skipped/errored summaries — truncated past 5 to keep the table tight."""
    if skipped:
        names = ", ".join(_config_name(c) for c in skipped[:5])
        if len(skipped) > 5:
            names += f", … (+{len(skipped) - 5} more)"
        print(f"  skipped: {names}")
    for cfg, msg in errored[:5]:
        print(f"  errored: {_config_name(cfg)} — {msg}")
    if len(errored) > 5:
        print(f"  … (+{len(errored) - 5} more errored)")


def _write_sweep_history(
    op: str,
    hw: HardwarePeak,
    flops: int,
    nbytes: int,
    flop_dtype: FlopDtype,
    axes: dict[str, Sequence[Any]],
    rows: list[SweepRow],
    skipped: list[dict[str, Any]],
    errored: list[tuple[dict[str, Any], str]],
) -> None:
    op_dir = _history.HISTORY_DIR / op
    op_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    record: dict[str, Any] = {
        "op": op,
        # ``kind`` is the cross-record discriminator; compare records use
        # ``"compare"``. Keeps an aggregator from having to sniff fields.
        "kind": "sweep",
        "timestamp": ts,
        "git_sha": _history.git_sha(),
        "hw": asdict(hw),
        "flops": flops,
        "bytes": nbytes,
        "flop_dtype": flop_dtype,
        # ``config`` matches compare's slot — knobs constant across the
        # whole run. Empty here because each variant carries its own
        # config under ``variants[name]["config"]``.
        "config": {},
        "sweep_axes": {k: list(v) for k, v in axes.items()},
        "skipped": skipped,
        "errored": [{"config": cfg, "error": msg} for cfg, msg in errored],
        "variants": {
            name: {
                "config": cfg,
                "median_s": br.median_s,
                "p99_s": br.p99_s,
                "min_s": br.min_s,
                "stdev_s": br.stdev_s,
                "mfu": roof.mfu,
                "bw_util": roof.bw_util,
                "ici_util": roof.ici_util,
                "sol_pct": roof.sol_pct,
                "binds": roof.binds,
            }
            for cfg, name, br, roof in rows
        },
    }
    out = op_dir / f"{ts}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"  wrote {out.relative_to(_history.REPO_ROOT)}")
