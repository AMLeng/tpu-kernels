"""Regression tests for benchmarks/sweep.py — Cartesian sweeps over kernel configs.

Sweep is a sibling to compare(): "1 variant x N configs" instead of "N variants
x 1 config". These tests pin its contract — Cartesian product, validity
filtering, error containment, JSON record shape, leaderboard sort — so the
machinery a Pallas tuning loop relies on can't silently break.

Per CLAUDE.md, every harness fix needs a regression test; this file is where
sweep-side ones land.
"""

from __future__ import annotations

import json
import linecache
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from benchmarks.roofline import HardwarePeak, v5e
from benchmarks.runner import BenchResult
from benchmarks.sweep import _check_single_chip, _config_name, sweep
from benchmarks.workload import Workload


def _identity_kernel(x: jax.Array, *, block_shape: Any = None) -> jax.Array:
    """Stand-in pallas_fn: declares ``block_shape`` (the convention every
    Pallas kernel in this repo follows), ignores it, returns x unchanged.

    sweep() jit-wraps the kernel as ``jax.jit(functools.partial(pallas_fn,
    block_shape=value))`` and validates that ``block_shape`` is a declared
    parameter, so the test stand-in must declare it too.
    """
    return x


def _setup_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("benchmarks._history.HISTORY_DIR", tmp_path)
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)


def _x() -> jax.Array:
    # ``random.normal`` not zeros: keeps the suite consistent with CLAUDE.md's
    # "no zeros/ones for bench inputs" rule (constant-folding risk). These
    # tests don't measure speed, so the rule's spirit is fine either way, but
    # one convention across the file is easier to reason about.
    return jax.random.normal(jax.random.key(0), (1,), dtype=jnp.float32)


def _w(op: str = "op", *, flops: int = 1, nbytes: int = 1) -> Workload:
    """Default Workload for tests: scalar input, unit flops/nbytes. Tests
    that care about specific values build their own ``Workload`` directly."""
    return Workload(op=op, flops=flops, nbytes=nbytes, args=(_x(),))


def test_cartesian_product_includes_all_combinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`{a:[1,2,3], b:[10,20]}` should produce 6 configs, named in axis order."""
    _setup_history(tmp_path, monkeypatch)
    results = sweep(
        _w(),
        pallas_fn=_identity_kernel,
        axes={"a": [1, 2, 3], "b": [10, 20]},
        warmup=0,
        iters=1,
    )
    assert set(results.keys()) == {
        "a=1,b=10",
        "a=1,b=20",
        "a=2,b=10",
        "a=2,b=20",
        "a=3,b=10",
        "a=3,b=20",
    }


def test_is_valid_filters_configs_before_bench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """``is_valid`` runs first; falsy configs are skipped — bench sees only
    valid ones, and skipped ones don't appear in results."""
    _setup_history(tmp_path, monkeypatch)
    benched_names: list[str] = []

    def fake_bench(name: str, **_kw: Any) -> BenchResult:
        benched_names.append(name)
        return BenchResult(name=name, times_s=[1e-3], warmup_iters=1, timed_iters=1)

    monkeypatch.setattr("benchmarks.sweep.bench", fake_bench)
    results = sweep(
        _w(),
        pallas_fn=_identity_kernel,
        axes={"a": [1, 5, 6]},
        is_valid=lambda *, a: 12 % a == 0,
        warmup=0,
        iters=1,
    )
    # Only divisors of 12 (1 and 6) should have reached bench.
    assert sorted(benched_names) == ["op::a=1", "op::a=6"]
    assert set(results.keys()) == {"a=1", "a=6"}
    out = capsys.readouterr().out
    assert "skipped" in out.lower()


def test_errored_configs_dont_abort_the_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A kernel that raises during trace for a specific block_shape is
    reported, the sweep continues, and the surviving configs land in
    results."""
    _setup_history(tmp_path, monkeypatch)

    def kernel(x: jax.Array, *, block_shape: tuple[int, ...]) -> jax.Array:
        if block_shape == (2,):
            raise RuntimeError("synthetic build failure")
        return x

    results = sweep(
        _w(),
        pallas_fn=kernel,
        axes={"a": [1, 2, 3]},
        warmup=0,
        iters=1,
    )
    assert set(results.keys()) == {"a=1", "a=3"}
    out = capsys.readouterr().out
    assert "errored" in out.lower()
    assert "a=2" in out


def test_assembles_block_shape_tuple_from_cfg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sweep always passes ``block_shape`` as a tuple, regardless of axis
    count. 1-axis sweeps yield a 1-tuple (``(8,)``, not bare ``8``);
    multi-axis sweeps yield an N-tuple in dict insertion order so
    ``axes={"bm": ..., "bn": ...}`` produces ``block_shape=(bm, bn)`` —
    matching the convention every Pallas kernel uses."""
    _setup_history(tmp_path, monkeypatch)
    seen: list[tuple[int, ...]] = []

    def kernel(x: jax.Array, *, block_shape: tuple[int, ...]) -> jax.Array:
        seen.append(block_shape)
        return x

    sweep(
        _w(),
        pallas_fn=kernel,
        axes={"bm": [8, 16]},
        warmup=0,
        iters=1,
    )
    assert sorted(seen) == [(8,), (16,)]
    seen.clear()

    sweep(
        _w(),
        pallas_fn=kernel,
        axes={"bm": [8, 16], "bn": [128]},
        warmup=0,
        iters=1,
    )
    assert sorted(seen) == [(8, 128), (16, 128)]


def test_rejects_pallas_fn_without_block_shape_kwarg() -> None:
    """Every Pallas kernel benched by sweep() must declare a ``block_shape``
    parameter — that's the harness-wide convention compare()/sweep() rely
    on to bake the value in via partial. A kernel missing the kwarg would
    otherwise silently fall through to whatever default the kernel uses,
    not the value the suite is sweeping. Catch it at call time."""

    def kernel_without_block_shape(x: jax.Array, *, tile: int = 0) -> jax.Array:
        return x

    with pytest.raises(ValueError, match="block_shape"):
        sweep(
            _w(),
            pallas_fn=kernel_without_block_shape,
            axes={"a": [1]},
            warmup=0,
            iters=1,
        )


def test_sweep_writes_one_history_record_with_per_variant_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One JSON per sweep, with each variant's config preserved structurally
    so an aggregator doesn't have to parse the variant name string. Top-level
    ``sweep_axes`` mirrors compare's per-record-config slot — ``config`` is
    reserved for run-wide knobs (empty here) so the field means the same
    thing across both kinds.
    """
    _setup_history(tmp_path, monkeypatch)
    sweep(
        _w("op_x"),
        pallas_fn=_identity_kernel,
        axes={"bm": [8, 16], "bn": [128]},
        warmup=0,
        iters=1,
    )
    files = list((tmp_path / "op_x").glob("*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record["kind"] == "sweep"
    assert record["sweep_axes"] == {"bm": [8, 16], "bn": [128]}
    assert record["config"] == {}
    for entry in record["variants"].values():
        assert "config" in entry
        assert set(entry["config"].keys()) == {"bm", "bn"}


def test_sweep_records_unroll_and_timing_per_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each sweep variant entry carries ``unroll`` and ``timing`` so a
    trend tool can tell which timing mode produced a number and whether
    a kernel hit the unroll cap. Sibling test for compare lives in
    tests/test_compare.py.
    """
    _setup_history(tmp_path, monkeypatch)
    sweep(
        _w("op_t"),
        pallas_fn=_identity_kernel,
        axes={"bm": [8]},
        warmup=0,
        iters=1,
    )
    files = list((tmp_path / "op_t").glob("*.json"))
    record = json.loads(files[0].read_text())
    for entry in record["variants"].values():
        assert isinstance(entry["unroll"], int)
        assert entry["unroll"] >= 1
        assert entry["timing"] in {"unroll", "device"}
        # On CPU/unroll this is always False; the assertion proves the
        # field round-trips as a bool, not that the parse mismatched.
        assert entry["cluster_mismatch"] is False


def test_sweep_forwards_timing_kwarg_to_bench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``sweep(timing="device")`` reaches ``bench(...)`` for every config."""
    _setup_history(tmp_path, monkeypatch)
    captured: list[str] = []

    def fake_bench(*, timing: str = "unroll", **_kwargs: Any) -> BenchResult:
        captured.append(timing)
        return BenchResult(name="x", times_s=[1e-3], warmup_iters=1, timed_iters=1)

    monkeypatch.setattr("benchmarks.sweep.bench", fake_bench)
    sweep(
        _w("op_z"),
        pallas_fn=_identity_kernel,
        axes={"a": [1, 2]},
        timing="device",
        warmup=0,
        iters=1,
        write_history=False,
    )
    assert captured == ["device", "device"]


def test_skipped_and_errored_configs_persist_in_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipped and errored configs land in the JSON so the recorded shape of
    the sweep matches what was *attempted*, not just what ran. Useful for
    later auditing — "did we try (8, 128)?" is answerable without rerunning.
    """
    _setup_history(tmp_path, monkeypatch)

    def kernel(x: jax.Array, *, block_shape: tuple[int, ...]) -> jax.Array:
        if block_shape == (3,):
            raise RuntimeError("boom")
        return x

    sweep(
        _w("op_y"),
        pallas_fn=kernel,
        axes={"a": [1, 2, 3, 4]},
        is_valid=lambda *, a: a != 2,
        warmup=0,
        iters=1,
    )
    files = list((tmp_path / "op_y").glob("*.json"))
    record = json.loads(files[0].read_text())
    assert record["skipped"] == [{"a": 2}]
    errored = record["errored"]
    assert len(errored) == 1
    assert errored[0]["config"] == {"a": 3}
    assert "boom" in errored[0]["error"]


def test_table_marks_winner_and_sorts_by_sol_desc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Leaderboard rows must come out in SoL%-descending order, with `*` on
    the highest one. We pin the order by mocking ``bench`` to return scripted
    timings (``a=2`` fastest → highest SoL%), then assert the row sequence
    in stdout. A noisy real bench would let a wrong-direction sort slip
    through; this test wouldn't.
    """
    _setup_history(tmp_path, monkeypatch)

    scripted_times = {"op::a=1": 0.010, "op::a=2": 0.001, "op::a=3": 0.005}

    def fake_bench(
        name: str,
        fn: Any,
        args: Sequence[Any] = (),
        kwargs: Any = None,
        warmup: int = 5,
        iters: int = 20,
        profile_dir: str | None = None,
        **_kw: Any,  # tolerate timing= and any future bench kwargs
    ) -> BenchResult:
        t = scripted_times[name]
        return BenchResult(name=name, times_s=[t, t], warmup_iters=1, timed_iters=2)

    monkeypatch.setattr("benchmarks.sweep.bench", fake_bench)

    sweep(
        _w(),
        pallas_fn=_identity_kernel,
        axes={"a": [1, 2, 3]},
        warmup=0,
        iters=2,
    )
    out = capsys.readouterr().out
    # Data rows are the only ones ending in a numeric percent (the SoL%
    # column); the header ends in the literal "SoL%". Order of appearance
    # in stdout == leaderboard order, so this also tests the sort.
    data_rows = [ln for ln in out.splitlines() if re.search(r"\d+\.\d+%\s*$", ln)]
    # First token is either "*<value>" (winner) or "<value>" (rest); strip
    # the marker so we compare bare axis values.
    leaderboard_values = [ln.split()[0].lstrip("*") for ln in data_rows]
    # fastest → slowest: a=2 (0.001), a=3 (0.005), a=1 (0.010).
    assert leaderboard_values == ["2", "3", "1"], (
        f"expected SoL-desc order [2, 3, 1] but got {leaderboard_values}\n{out}"
    )
    # Exactly one row gets the marker, and it's the winner.
    starred = [ln for ln in data_rows if ln.lstrip().startswith("*")]
    assert len(starred) == 1
    assert starred[0].split()[0] == "*2"
    assert "best: a=2" in out


def test_sweep_with_no_valid_configs_doesnt_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Pathological case: every config skipped by is_valid. Sweep returns {}
    and prints a "no valid configs" message instead of bailing.
    """
    _setup_history(tmp_path, monkeypatch)
    results = sweep(
        _w(),
        pallas_fn=_identity_kernel,
        axes={"a": [1, 2, 3]},
        is_valid=lambda *, a: False,
        warmup=0,
        iters=1,
    )
    assert results == {}
    assert "no valid configs" in capsys.readouterr().out.lower()


def test_check_single_chip_warns_on_multi_chip() -> None:
    """sweep() is single-chip-only today; multi-chip ``hw`` is a callsite mistake.

    Without this guard, a sweep called with ``hw=v5e(num_chips=8)`` would print
    "8 chips" in the table header while the rooflines were single-chip — the
    silent-lie-in-the-harness pattern CLAUDE.md flags as never-acceptable.
    """
    with pytest.warns(UserWarning, match="single-chip-only"):
        _check_single_chip(v5e(num_chips=8))


def test_check_single_chip_silent_on_single_chip() -> None:
    """The default (num_chips=1) path must not warn — the supported case."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_single_chip(v5e(num_chips=1))


def test_check_single_chip_warning_skips_sweep_frame() -> None:
    """stacklevel=3 points the warning at the user's sweep(...) callsite.

    Mirrors compare's ICI-consistency stacklevel test: the warning needs to
    land on the line the user can fix, not inside ``benchmarks/sweep.py``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        def stand_in_for_sweep() -> None:
            _check_single_chip(v5e(num_chips=8))

        stand_in_for_sweep()  # this is the line that should be reported

    assert len(caught) == 1
    line = linecache.getline(caught[0].filename, caught[0].lineno).strip()
    assert "stand_in_for_sweep()" in line


def test_sweep_warns_on_multi_chip_hw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: ``sweep(hw=v5e(num_chips=8))`` emits the guard warning.

    The unit tests above cover the helper; this one pins the fact that
    ``sweep()`` actually calls it. Without this assertion, the helper could
    sit unused and the silent lie would still ship.
    """
    _setup_history(tmp_path, monkeypatch)
    with pytest.warns(UserWarning, match="single-chip-only"):
        sweep(
            _w(),
            pallas_fn=_identity_kernel,
            axes={"a": [1]},
            hw=v5e(num_chips=8),
            warmup=0,
            iters=1,
        )


def test_sweep_raises_when_any_config_reports_unphysical_sol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SoL > 100% on any swept config must surface as an error.

    Mirror of the compare-side guard. Catches the unroll-mode tile-pipelining
    artifact that motivated the fix (rmsnorm at small shapes), and any
    future case where the roofline constants or flops/nbytes are wrong.
    Sweep still prints the leaderboard (already sorted by SoL desc) before
    raising, so the operator sees which configs went unphysical.
    """
    _setup_history(tmp_path, monkeypatch)

    def fake_bench(**_kwargs: Any) -> BenchResult:
        return BenchResult(name="x", times_s=[1e-9], warmup_iters=1, timed_iters=1)

    monkeypatch.setattr("benchmarks.sweep.bench", fake_bench)
    with pytest.raises(RuntimeError, match=r"(?i)sol.*100%|100%.*unphysical"):
        sweep(
            _w("op_unphys", nbytes=10**12),
            pallas_fn=_identity_kernel,
            axes={"a": [1, 2]},
            warmup=0,
            iters=1,
        )
    # History must NOT have been written when the run is unphysical.
    assert not list((tmp_path / "op_unphys").glob("*.json"))


def test_sweep_rejects_unsupported_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirror of compare's hw-validity guard. A swept tuning run on the
    wrong hardware would emit a leaderboard of fictional SoL%/BW% — the
    operator might pick the "best" config based on numbers that came
    from a different chip entirely. Refuse up front."""
    _setup_history(tmp_path, monkeypatch)
    fake = HardwarePeak(
        kind="v6e",
        bf16_flops=918e12,
        f32_flops=459e12,
        int8_ops=1836e12,
        hbm_bw=1640e9,
        ici_bw_per_link=400e9,
        ici_links_per_chip=4,
    )
    with pytest.raises(NotImplementedError, match="v5e"):
        sweep(
            _w(),
            pallas_fn=_identity_kernel,
            axes={"a": [1]},
            hw=fake,
            warmup=0,
            iters=1,
        )


def test_config_name_encoding_preserves_axis_insertion_order() -> None:
    """The variant name is a stable string key. Insertion order matters
    because suites pass axes as a dict and the user's chosen order is the
    one that should show up in tables and JSON keys."""
    assert _config_name({"bm": 8, "bn": 128}) == "bm=8,bn=128"
    assert _config_name({"bn": 128, "bm": 8}) == "bn=128,bm=8"
    assert _config_name({"a": 1}) == "a=1"
