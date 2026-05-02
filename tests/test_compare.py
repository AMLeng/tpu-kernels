"""Regression tests for benchmarks/compare.py output and history helpers.

These pin behavior of the table-printing path so silent breakage of the
ridge / regime classification can't slip through, and pin the dirty-tree
marker on `_git_sha` so bench-history records can't lie about the tree
state they were produced from. CLAUDE.md mandates a test for every
harness fix; this file is where compare-side ones land.
"""

from __future__ import annotations

import json
import linecache
import subprocess
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from benchmarks._history import git_sha
from benchmarks.compare import (
    BenchRow,
    _check_ici_consistency,
    _print_table,
    _write_history,
)
from benchmarks.roofline import FlopDtype, analyze, v5e
from benchmarks.runner import BenchResult


def _row(flop_dtype: FlopDtype, flops: int, nbytes: int):
    """Build one (name, BenchResult, Roofline) row for _print_table."""
    br = BenchResult(name="x", times_s=[1.0], warmup_iters=1, timed_iters=1)
    roof = analyze(flops=flops, nbytes=nbytes, seconds=1.0, hw=v5e(), flop_dtype=flop_dtype)
    return ("variant", br, roof)


def test_print_table_ridge_uses_int8_peak_when_dtype_int8(capsys: pytest.CaptureFixture) -> None:
    """Regression: _print_table previously hardcoded total_bf16_flops for the ridge.

    On v5e: bf16 ridge ≈ 240 F/B, int8 ridge ≈ 480 F/B. At intensity 300 F/B
    with flop_dtype="int8" the kernel is below the int8 ridge → memory-bound.
    The buggy version would compare against the bf16 ridge and (mis)label it
    compute-bound.
    """
    flops, nbytes = 300_000, 1_000  # intensity = 300 F/B
    _print_table("dummy", v5e(), flops, nbytes, 0, [_row("int8", flops, nbytes)], "int8")
    out = capsys.readouterr().out
    assert "memory-bound" in out
    assert "compute-bound" not in out


def test_print_table_ridge_uses_bf16_peak_when_dtype_bf16(capsys: pytest.CaptureFixture) -> None:
    """Same intensity, bf16 dtype → above bf16 ridge → compute-bound."""
    flops, nbytes = 300_000, 1_000
    _print_table("dummy", v5e(), flops, nbytes, 0, [_row("bf16", flops, nbytes)], "bf16")
    out = capsys.readouterr().out
    assert "compute-bound" in out
    assert "memory-bound" not in out


def _init_repo(path: Path) -> None:
    """Create a tiny one-commit git repo at `path`. Used by _git_sha tests."""
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=path, check=True)
    (path / "f").write_text("hello")
    subprocess.run(["git", "add", "f"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def test_git_sha_clean_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _init_repo(tmp_path)
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)
    sha = git_sha()
    assert sha is not None
    assert len(sha) == 40  # full SHA, no suffix
    assert not sha.endswith("-dirty")


def test_git_sha_dirty_tree_with_modified_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    (tmp_path / "f").write_text("modified")  # tracked file changed, not committed
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)
    sha = git_sha()
    assert sha is not None
    assert sha.endswith("-dirty")


def test_git_sha_dirty_tree_with_untracked_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Untracked files matter too — a forgotten benchmarks/foo.py would change
    # the run even though `git diff HEAD` wouldn't see it.
    _init_repo(tmp_path)
    (tmp_path / "untracked.py").write_text("print('x')")
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)
    sha = git_sha()
    assert sha is not None
    assert sha.endswith("-dirty")


def test_git_sha_returns_none_outside_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)
    assert git_sha() is None


def test_git_sha_clean_when_only_bench_history_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``bench_history/`` is *output*, never input — it can't have affected
    the run that produced it. Without filtering it from the porcelain check,
    a second compare() in the same shell sees the first run's untracked JSON
    and stamps the second SHA dirty even though the kernel/harness code is
    unchanged.
    """
    _init_repo(tmp_path)
    history = tmp_path / "bench_history" / "op_a"
    history.mkdir(parents=True)
    (history / "1234.json").write_text("{}")
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)
    sha = git_sha()
    assert sha is not None
    assert not sha.endswith("-dirty")


def test_git_sha_dirty_when_bench_history_and_code_both_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filtering bench_history/ must not mask real code changes alongside it."""
    _init_repo(tmp_path)
    (tmp_path / "f").write_text("modified")
    history = tmp_path / "bench_history" / "op_a"
    history.mkdir(parents=True)
    (history / "1234.json").write_text("{}")
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)
    sha = git_sha()
    assert sha is not None
    assert sha.endswith("-dirty")


def _history_row() -> BenchRow:
    """Single fake (name, BenchResult, Roofline) triple for _write_history tests."""
    br = BenchResult(name="v", times_s=[1.0], warmup_iters=1, timed_iters=1)
    roof = analyze(flops=1, nbytes=1, seconds=1.0, hw=v5e())
    return ("v", br, roof)


def test_write_history_records_in_same_second_dont_collide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two records produced inside one wall-clock second must not overwrite.

    A seconds-resolution timestamp silently dropped the earlier record on
    sweep loops; microsecond resolution prevents that.
    """
    monkeypatch.setattr("benchmarks._history.HISTORY_DIR", tmp_path)
    # _write_history prints `out.relative_to(REPO_ROOT)`; keep that legal.
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)

    fake_times = iter(
        [
            datetime(2026, 4, 29, 12, 0, 0, 100, tzinfo=UTC),
            datetime(2026, 4, 29, 12, 0, 0, 200, tzinfo=UTC),
        ]
    )

    class StubDateTime:
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return next(fake_times)

    monkeypatch.setattr("benchmarks.compare.datetime", StubDateTime)

    rows = [_history_row()]
    _write_history("op_a", v5e(), 1, 1, "bf16", 0, rows)
    _write_history("op_a", v5e(), 1, 1, "bf16", 0, rows)

    files = sorted((tmp_path / "op_a").glob("*.json"))
    assert len(files) == 2


def test_check_ici_consistency_warns_on_single_chip() -> None:
    """ici_bytes > 0 with num_chips == 1 is a callsite mistake — warn."""
    with pytest.warns(UserWarning, match="num_chips == 1"):
        _check_ici_consistency(v5e(num_chips=1), ici_bytes=100)


def test_check_ici_consistency_silent_when_ici_bytes_zero() -> None:
    """The default single-chip path (ici_bytes=0) must not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_ici_consistency(v5e(num_chips=1), ici_bytes=0)


def test_check_ici_consistency_silent_when_multi_chip() -> None:
    """ICI on multi-chip runs is exactly what ici_bytes is for — no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_ici_consistency(v5e(num_chips=8), ici_bytes=100)


def test_check_ici_consistency_warning_skips_compare_frame() -> None:
    """stacklevel=3 points the warning at the user's compare(...) callsite.

    Regression: stacklevel=2 attributed the warning to compare.py itself
    (one frame too few), where the user can't fix anything. We simulate
    the real chain — user → compare() → _check_ici_consistency → warn —
    and check the warning's filename/lineno land on the user-equivalent line.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        def stand_in_for_compare() -> None:
            _check_ici_consistency(v5e(num_chips=1), ici_bytes=100)

        stand_in_for_compare()  # this is the line that should be reported

    assert len(caught) == 1
    line = linecache.getline(caught[0].filename, caught[0].lineno).strip()
    # With stacklevel=3 we jump past `_check_ici_consistency` and
    # `stand_in_for_compare`'s body to land on the call site here.
    assert "stand_in_for_compare()" in line


def test_write_history_records_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-call config (e.g. block_shape) lands in the JSON record.

    Lets a sweep keep stable variant names ("pallas", not "pallas_256x256")
    while still being queryable across runs.
    """
    monkeypatch.setattr("benchmarks._history.HISTORY_DIR", tmp_path)
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)
    rows = [_history_row()]
    cfg = {"block_shape": [256, 256]}
    _write_history("op_x", v5e(), 1, 1, "bf16", 0, rows, config=cfg)
    files = list((tmp_path / "op_x").glob("*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record["config"] == cfg


def test_write_history_config_defaults_to_empty_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No config kwarg → record still has a `config` key (empty dict).

    Always-present shape keeps downstream parsers from needing a `.get`
    fallback.
    """
    monkeypatch.setattr("benchmarks._history.HISTORY_DIR", tmp_path)
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)
    rows = [_history_row()]
    _write_history("op_y", v5e(), 1, 1, "bf16", 0, rows)
    files = list((tmp_path / "op_y").glob("*.json"))
    record = json.loads(files[0].read_text())
    assert record["config"] == {}


def test_compare_forwards_timing_kwarg_to_bench(monkeypatch: pytest.MonkeyPatch) -> None:
    """``compare(timing="device")`` must reach ``bench(...)`` unchanged.

    Caller-side regression: forgetting to forward ``timing`` would silently
    keep wallclock-amortized numbers, undoing phase 2.
    """
    captured: list[str] = []

    def fake_bench(*, timing: str = "unroll", **_kwargs: Any) -> BenchResult:
        captured.append(timing)
        return BenchResult(name="x", times_s=[1e-3], warmup_iters=1, timed_iters=1)

    monkeypatch.setattr("benchmarks.compare.bench", fake_bench)
    import jax
    import jax.numpy as jnp
    from benchmarks.compare import compare

    @jax.jit
    def f(x: jax.Array) -> jax.Array:
        return x

    compare(
        op="op",
        variants={"v": f},
        args=(jnp.zeros(1),),
        flops=1,
        nbytes=1,
        timing="device",
        write_history=False,
    )
    assert captured == ["device"]


def test_compare_defaults_timing_to_unroll(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``timing`` arg → bench gets ``timing="unroll"`` (the bench default)."""
    captured: list[str] = []

    def fake_bench(*, timing: str = "unroll", **_kwargs: Any) -> BenchResult:
        captured.append(timing)
        return BenchResult(name="x", times_s=[1e-3], warmup_iters=1, timed_iters=1)

    monkeypatch.setattr("benchmarks.compare.bench", fake_bench)
    import jax
    import jax.numpy as jnp
    from benchmarks.compare import compare

    @jax.jit
    def f(x: jax.Array) -> jax.Array:
        return x

    compare(
        op="op",
        variants={"v": f},
        args=(jnp.zeros(1),),
        flops=1,
        nbytes=1,
        write_history=False,
    )
    assert captured == ["unroll"]


def test_write_history_records_unroll_and_timing_per_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-variant ``unroll`` and ``timing`` round-trip into the JSON record.

    Trend tooling needs to know whether a number came from unroll-mode
    (wallclock / k) or device-mode (XPlane), since they aren't directly
    comparable. ``unroll`` also surfaces when a kernel hit ``k_max``.
    """
    monkeypatch.setattr("benchmarks._history.HISTORY_DIR", tmp_path)
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)
    br = BenchResult(
        name="v",
        times_s=[1.0],
        warmup_iters=1,
        timed_iters=1,
        unroll=64,
        timing="unroll",
        cluster_mismatch=True,
    )
    roof = analyze(flops=1, nbytes=1, seconds=1.0, hw=v5e())
    rows: list[BenchRow] = [("v", br, roof)]
    _write_history("op_q", v5e(), 1, 1, "bf16", 0, rows)
    files = list((tmp_path / "op_q").glob("*.json"))
    record = json.loads(files[0].read_text())
    entry = record["variants"]["v"]
    assert entry["unroll"] == 64
    assert entry["timing"] == "unroll"
    assert entry["cluster_mismatch"] is True


def test_write_history_stamps_kind_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every compare record carries ``kind: "compare"`` so an aggregator can
    multiplex it with sweep records (``kind: "sweep"``) without sniffing fields.
    Sibling test for sweep lives in tests/test_sweep.py.
    """
    monkeypatch.setattr("benchmarks._history.HISTORY_DIR", tmp_path)
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)
    rows = [_history_row()]
    _write_history("op_z", v5e(), 1, 1, "bf16", 0, rows)
    files = list((tmp_path / "op_z").glob("*.json"))
    record = json.loads(files[0].read_text())
    assert record["kind"] == "compare"


def test_compare_raises_when_variant_reports_unphysical_sol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SoL > 100% is unphysical — compare() must surface it as an error.

    A kernel can't run faster than its hardware roofline. The most common
    cause is unroll-mode chained calls letting XLA pipeline tiles across
    them when the input fits in VMEM (rmsnorm at small shapes was the
    motivating bug). We raise rather than emit a misleading bench-history
    record; the table still prints first so the operator sees the numbers.
    """
    _setup_history(tmp_path, monkeypatch)

    def fake_bench(**_kwargs: Any) -> BenchResult:
        # nbytes=1e12, seconds=1e-9 → modeled BW = 1e21 B/s, well above peak.
        return BenchResult(name="x", times_s=[1e-9], warmup_iters=1, timed_iters=1)

    monkeypatch.setattr("benchmarks.compare.bench", fake_bench)
    import jax
    import jax.numpy as jnp
    from benchmarks.compare import compare

    @jax.jit
    def f(x: jax.Array) -> jax.Array:
        return x

    with pytest.raises(RuntimeError, match=r"(?i)sol.*100%|100%.*unphysical"):
        compare(
            op="op",
            variants={"v": f},
            args=(jnp.zeros(1),),
            flops=1,
            nbytes=10**12,
        )
    # And: history must NOT have been written when the run is unphysical.
    assert not list(tmp_path.glob("op/*.json"))


def test_compare_silent_for_sol_at_or_below_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boundary: SoL == 1.0 is on the roofline, not above it. No raise."""

    def fake_bench(**_kwargs: Any) -> BenchResult:
        # seconds chosen so modeled BW exactly matches v5e total HBM peak.
        return BenchResult(
            name="x",
            times_s=[1.0 / v5e().total_hbm_bw],
            warmup_iters=1,
            timed_iters=1,
        )

    monkeypatch.setattr("benchmarks.compare.bench", fake_bench)
    import jax
    import jax.numpy as jnp
    from benchmarks.compare import compare

    @jax.jit
    def f(x: jax.Array) -> jax.Array:
        return x

    compare(
        op="op",
        variants={"v": f},
        args=(jnp.zeros(1),),
        flops=1,
        nbytes=1,
        write_history=False,
    )


def _setup_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("benchmarks._history.HISTORY_DIR", tmp_path)
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)
