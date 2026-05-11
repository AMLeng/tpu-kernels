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

import jax
import jax.numpy as jnp
import pytest
from benchmarks._history import git_sha
from benchmarks.compare import (
    BenchRow,
    _check_ici_consistency,
    _print_table,
    _write_history,
    compare,
)
from benchmarks.roofline import FlopDtype, HardwarePeak, analyze, v5e
from benchmarks.runner import BenchResult
from benchmarks.workload import Workload


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


def test_git_sha_clean_when_only_tests_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests are not bench inputs — they don't affect kernel behavior or
    bench output, so a modified test file shouldn't taint the SHA. This
    lets us iterate on tests (e.g. tightening thresholds, adding xfails)
    without invalidating bench JSONs taken under the same kernel code."""
    _init_repo(tmp_path)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text("def test_x(): assert True")
    subprocess.run(["git", "add", "tests/test_x.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add test"], cwd=tmp_path, check=True)
    (tests / "test_x.py").write_text("def test_x(): assert False")  # tracked test edit
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)
    sha = git_sha()
    assert sha is not None
    assert not sha.endswith("-dirty")


def test_git_sha_dirty_when_tests_and_code_both_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Excluding tests/ must not mask real code changes alongside it
    (sibling to ``..._when_bench_history_and_code_both_changed``)."""
    _init_repo(tmp_path)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text("def test_x(): assert True")
    subprocess.run(["git", "add", "tests/test_x.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add test"], cwd=tmp_path, check=True)
    (tests / "test_x.py").write_text("def test_x(): assert False")
    (tmp_path / "f").write_text("modified")  # real code change alongside
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

    @jax.jit
    def f(x: jax.Array) -> jax.Array:
        return x

    compare(
        Workload(op="op", flops=1, nbytes=1, args=(jnp.zeros(1),)),
        variants={"v": f},
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

    @jax.jit
    def f(x: jax.Array) -> jax.Array:
        return x

    compare(
        Workload(op="op", flops=1, nbytes=1, args=(jnp.zeros(1),)),
        variants={"v": f},
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

    @jax.jit
    def f(x: jax.Array) -> jax.Array:
        return x

    with pytest.raises(RuntimeError, match=r"(?i)sol.*100%|100%.*unphysical"):
        compare(
            Workload(op="op", flops=1, nbytes=10**12, args=(jnp.zeros(1),)),
            variants={"v": f},
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

    @jax.jit
    def f(x: jax.Array) -> jax.Array:
        return x

    compare(
        Workload(op="op", flops=1, nbytes=1, args=(jnp.zeros(1),)),
        variants={"v": f},
        write_history=False,
    )


def _setup_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("benchmarks._history.HISTORY_DIR", tmp_path)
    monkeypatch.setattr("benchmarks._history.REPO_ROOT", tmp_path)


def test_compare_rejects_unsupported_hardware() -> None:
    """compare() must refuse a HardwarePeak that doesn't match the kernels
    and constants the suite is tuned for. Without this guard, pointing
    the bench harness at a v6e host (or hand-rolling a v5p HardwarePeak)
    would silently emit fictional BW%/MFU%."""

    @jax.jit
    def f(x: jax.Array) -> jax.Array:
        return x

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
        compare(
            Workload(op="op", flops=1, nbytes=1, args=(jnp.zeros(1),)),
            variants={"v": f},
            hw=fake,
            write_history=False,
        )


def test_pallas_variant_rejects_kernel_without_block_shape_kwarg() -> None:
    """Every Pallas kernel benched by compare()/sweep() must declare a
    ``block_shape`` parameter — that's the repo-wide convention. A typo
    (``block_size``, ``tile``) would otherwise produce a silent mis-bench:
    a TypeError deep inside jax, or — worse — the kwarg landing in a
    kernel that quietly uses its own default. ``pallas_variant()`` catches
    the mismatch at construction so the error surfaces at the suite line
    that names the kernel."""
    from benchmarks.compare import pallas_variant

    def kernel_without_block_shape(x: jax.Array, *, tile: int = 0) -> jax.Array:
        return x

    with pytest.raises(ValueError, match="block_shape"):
        pallas_variant(kernel_without_block_shape, block_shape=(8,))


def test_pallas_variant_returns_jit_wrapped_callable_with_block_shape_baked_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pallas_variant(fn, block_shape=v)`` returns a jit-wrapped callable
    that calls ``fn(*args, block_shape=v)`` — the block is a closed-over
    Python static, not a traced input."""
    from benchmarks.compare import pallas_variant

    seen: list[tuple[int, ...]] = []

    def kernel(x: jax.Array, *, block_shape: tuple[int, ...]) -> jax.Array:
        seen.append(block_shape)
        return x

    variant = pallas_variant(kernel, block_shape=(64, 128))
    variant(jnp.zeros(1))
    assert seen == [(64, 128)]


def test_compare_kwargs_param_is_keyword_only() -> None:
    """``kwargs`` is reserved for the rare suite that wants to pass kernel
    keyword arguments through to ``bench``/``lower``; nothing in this repo
    uses it today. Keeping it keyword-only prevents a future suite from
    silently binding a third positional dict to it — a 3-arg call that
    *meant* something else (an ``is_valid``-style closure dict, perhaps)
    would otherwise pass type-check and run the bench against the wrong
    contract.
    """

    @jax.jit
    def f(x: jax.Array) -> jax.Array:
        return x

    workload = Workload(op="op", flops=1, nbytes=1, args=(jnp.zeros(1),))
    with pytest.raises(TypeError, match="positional"):
        compare(workload, {"v": f}, {"some": "dict"})  # type: ignore[misc]


def test_compare_passes_per_variant_subdir_to_bench_under_profile_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``profile_dir`` must be sharded per variant before reaching ``bench``.

    Regression: ``compare()`` forwarded the same ``profile_dir`` to every
    ``bench()`` call. With two variants, both wrapped their timed loop in
    ``jax.profiler.trace(profile_dir)`` against the same path — the second
    trace clobbered the first's xplane.pb, and on TPU the active-trace
    overhead made both variants' wallclock converge to the same number.
    Each variant must get its own subdir so traces don't collide and the
    per-variant artifact is recoverable from xprof.
    """
    captured: list[str | None] = []

    def fake_bench(*, profile_dir: str | None = None, **_kw: Any) -> BenchResult:
        captured.append(profile_dir)
        return BenchResult(name="x", times_s=[1e-3], warmup_iters=1, timed_iters=1)

    monkeypatch.setattr("benchmarks.compare.bench", fake_bench)

    @jax.jit
    def f(x: jax.Array) -> jax.Array:
        return x

    compare(
        Workload(op="op", flops=1, nbytes=1, args=(jnp.zeros(1),)),
        variants={"xla": f, "pallas": f},
        profile_dir=str(tmp_path),
        write_history=False,
    )

    # Each variant gets its own profile dir; no two variants share a path.
    assert len(captured) == 2
    assert all(p is not None for p in captured)
    assert len(set(captured)) == 2
    # Each subdir lives under the user-supplied profile_dir.
    for p in captured:
        assert p is not None
        assert Path(p).parent == tmp_path
    # Subdir names match the variant names so the user can find traces.
    assert {Path(p).name for p in captured if p is not None} == {"xla", "pallas"}


def test_compare_invokes_force_pallas_debug_per_variant_when_dump_mosaic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``compare(dump_mosaic=True)`` must wrap each variant's
    ``.lower().compile()`` pass in ``force_pallas_debug()`` — that's what
    forces ``pl.pallas_call(debug=True)`` for the duration of the
    lowering, which is what makes JAX print the Mosaic IR. Without this
    plumbing, the flag would be a silent no-op.
    """
    import contextlib

    enters: list[str] = []
    compiles: list[str] = []
    inside: list[bool] = []
    active: list[bool] = [False]

    @contextlib.contextmanager
    def fake_force() -> Any:
        active[0] = True
        enters.append("enter")
        try:
            yield
        finally:
            active[0] = False

    monkeypatch.setattr("benchmarks.compare.force_pallas_debug", fake_force)

    def fake_bench(name: str, **_kw: Any) -> BenchResult:
        return BenchResult(name=name, times_s=[1e-3], warmup_iters=1, timed_iters=1)

    monkeypatch.setattr("benchmarks.compare.bench", fake_bench)

    class FakeCompiled:
        def as_text(self) -> str:
            return ""

    class FakeLowered:
        def compile(self) -> FakeCompiled:
            compiles.append("compile")
            inside.append(active[0])
            return FakeCompiled()

    class FakeJitted:
        def lower(self, *_args: Any, **_kwargs: Any) -> FakeLowered:
            return FakeLowered()

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return args[0] if args else None

    monkeypatch.setattr("benchmarks.compare._is_jitted", lambda _fn: True)

    compare(
        Workload(op="op", flops=1, nbytes=1, args=(jnp.zeros(1),)),
        variants={"a": FakeJitted(), "b": FakeJitted()},
        dump_mosaic=True,
        write_history=False,
    )

    assert enters == ["enter", "enter"]
    assert compiles == ["compile", "compile"]
    # Each .compile() must have run while the patch was active — otherwise
    # the Mosaic dump wouldn't fire even though the wrapper was entered.
    assert inside == [True, True]


def test_compare_does_not_invoke_force_pallas_debug_when_dump_mosaic_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default ``dump_mosaic=False`` must not patch ``pl.pallas_call`` —
    the monkey-patch is reserved for the dump pass. Leaking it into every
    bench run would force every Pallas variant to print its kernel jaxpr
    and Mosaic IR on every compile."""
    enters: list[str] = []

    import contextlib

    @contextlib.contextmanager
    def fake_force() -> Any:
        enters.append("enter")
        yield

    monkeypatch.setattr("benchmarks.compare.force_pallas_debug", fake_force)

    def fake_bench(**_kw: Any) -> BenchResult:
        return BenchResult(name="x", times_s=[1e-3], warmup_iters=1, timed_iters=1)

    monkeypatch.setattr("benchmarks.compare.bench", fake_bench)

    @jax.jit
    def f(x: jax.Array) -> jax.Array:
        return x

    compare(
        Workload(op="op", flops=1, nbytes=1, args=(jnp.zeros(1),)),
        variants={"v": f},
        write_history=False,
    )
    assert enters == []


def test_compare_passes_none_profile_dir_to_bench_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``profile_dir=None`` (default) must pass through unchanged.

    No subdir synthesis when the user didn't ask for a trace — the
    profiler-active dispatch overhead would otherwise leak into every
    benched run.
    """
    captured: list[str | None] = []

    def fake_bench(*, profile_dir: str | None = None, **_kw: Any) -> BenchResult:
        captured.append(profile_dir)
        return BenchResult(name="x", times_s=[1e-3], warmup_iters=1, timed_iters=1)

    monkeypatch.setattr("benchmarks.compare.bench", fake_bench)

    @jax.jit
    def f(x: jax.Array) -> jax.Array:
        return x

    compare(
        Workload(op="op", flops=1, nbytes=1, args=(jnp.zeros(1),)),
        variants={"v": f},
        write_history=False,
    )
    assert captured == [None]


def test_compare_can_bench_multiple_pallas_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The variants dict treats every entry uniformly — so a compare() can
    pit two Pallas kernels (same op, different impls) against an XLA
    baseline. This is the matmul curriculum step's actual flow:
    ``pallas`` vs ``pipelined`` vs ``xla``. Without this, a Pallas pair
    would need either two compare() calls or per-variant closures, both
    of which fragment the bench history."""
    from benchmarks.compare import pallas_variant

    benched: list[str] = []

    def fake_bench(name: str, **_kw: Any) -> BenchResult:
        benched.append(name)
        return BenchResult(name=name, times_s=[1e-3], warmup_iters=1, timed_iters=1)

    monkeypatch.setattr("benchmarks.compare.bench", fake_bench)

    def kernel_a(x: jax.Array, *, block_shape: tuple[int, ...]) -> jax.Array:
        return x

    def kernel_b(x: jax.Array, *, block_shape: tuple[int, ...]) -> jax.Array:
        return x

    compare(
        Workload(op="myop", flops=1, nbytes=1, args=(jnp.zeros(1),)),
        variants={
            "xla": jax.jit(lambda x: x),
            "pallas": pallas_variant(kernel_a, block_shape=(64, 128)),
            "pipelined": pallas_variant(kernel_b, block_shape=(64, 128)),
        },
        write_history=False,
    )
    assert sorted(benched) == ["myop::pallas", "myop::pipelined", "myop::xla"]
