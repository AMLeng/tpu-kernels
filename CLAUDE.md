# CLAUDE.md

Agent-facing notes for the `tpu-kernels` repo. Human-facing docs and the
copy-pasteable cookbook live in `README.md`.

## What this project is

JAX / Pallas kernels targeting TPU v5e, with a benchmarking harness built
around roofline analysis (MFU% and HBM bandwidth %). Single-host v5e-8
today; multi-host planned.

## Per-op workflow

Every op lives at `src/tpu_kernels/ops/<op>/` and ships in up to three variants:

1. `naive.py` — readable JAX, the correctness oracle. **Never tuned.**
2. `xla.py` — `@jax.jit`-wrapped, tuned plain JAX. First speed-of-light attempt.
3. `pallas.py` — only if `xla.py` fell short of the roofline target.

`PERF.md` in the op dir is forward-looking, not a log. Four lines:
`Target` / `Current` / `Bottleneck` / `Next`. Use `docs/PERF_TEMPLATE.md`.
Per-attempt history goes in `git log` and `bench_history/<op>/*.json`,
not in `PERF.md`.

If `xla.py` already hits the target, **stop**. Don't write Pallas just
because the slot exists; it must beat `xla` in `compare` to justify itself.

## After modifying a kernel

1. `uv run pytest tests/correctness/test_<op>.py` — must pass.
2. `uv run python -m benchmarks.suites.<op>` — re-bench (TPU only; CPU
   numbers from `interpret=True` are not meaningful).
3. Update `PERF.md`: new `Current %`, new `Bottleneck` hypothesis, new `Next`.
4. The new `bench_history/<op>/<timestamp>.json` is part of the commit.

## After any change

```bash
uv run ruff check . && uv run ruff format . && uv run pyright
uv run pytest
```

All four must be clean before committing. CI (`.github/workflows/ci.yml`)
runs the same checks, so a green local run should mean a green PR.

## Adding a new op

The kernel to build next is the first `[planned]` entry in
[`docs/curriculum.md`](docs/curriculum.md); read its rationale before
starting. Copy `src/tpu_kernels/ops/scale/` as the template. Files needed:

```
src/tpu_kernels/ops/<name>/{__init__.py, naive.py, xla.py, PERF.md}
src/tpu_kernels/ops/<name>/pallas.py        # optional
tests/correctness/test_<name>.py            # parametrize variants vs naive
benchmarks/suites/<name>.py                 # argparse CLI calling compare(...)
```

`__init__.py` re-exports as `<op>_naive`, `<op>_xla`, `<op>_pallas`.

Pallas variants are plain functions taking `interpret: bool = False` so CPU
correctness tests can pass it through to `pl.pallas_call(..., interpret=...)`.
They also declare `block_shape: tuple[int, ...]` (no other name) — the
harness validates the kwarg via `inspect.signature` and bakes the value
in via `pallas_variant(fn, block_shape=...)` for compare or directly
inside `sweep()` for a tuning loop. Don't decorate the variant itself
with `@jax.jit`; the harness does that.

When the new kernel hits its regime target, flip its curriculum entry
from `[planned]` to `[done]` and link the new `PERF.md`.

## Bench harness

- `benchmarks/runner.py` — warmup + N timed iters around `block_until_ready`.
- `benchmarks/roofline.py` — v5e per-chip peaks, MFU/BW math. **All
  hardware constants live here.** Update in one place if Google revises figures.
  `check_supported_hardware()` refuses non-v5e hw or non-v5e host TPUs.
- `benchmarks/workload.py` — `Workload` dataclass bundling the
  (op, args, flops, nbytes, flop_dtype) shared by compare and sweep.
- `benchmarks/compare.py` — N-variant comparison table, optional HLO dump,
  writes JSON to `bench_history/<op>/` with the current git SHA. Build
  Pallas entries with `pallas_variant(fn, block_shape=...)`.
- `benchmarks/sweep.py` — Cartesian sweep over `block_shape` configs for a
  single Pallas kernel; sibling JSON record under `bench_history/<op>/`.
- `benchmarks/suites/_common.py` — shared CLI scaffolding (`base_parser`
  for `--dtype`/`--timing`/`--block`/`--sweep-block`,
  `validate_block_shapes` for the per-op axis-count check).
- `benchmarks/suites/<op>.py` — one suite file per op.

Run a suite: `uv run python -m benchmarks.suites.<op>`. The README cookbook
has the full set of flags (`--dump-hlo`, `--profile-dir`, `--block`, etc.).

## Fixing a bug in the harness

Harness changes (`benchmarks/`, `tests/conftest.py`, anything that judges
kernels) are TDD-only: write the regression test first, watch it fail,
then make it pass. The failing test ships in the same commit as the
fix. The harness is what we trust to call a kernel correct or fast —
silent regressions there are silent regressions everywhere. Kernel
edits don't carry this rule; harness edits always do.

## Bench inputs

Use `jax.random.normal(jax.random.key(0), shape, dtype)` for reproducible
inputs. **Avoid `jnp.zeros` / `jnp.ones`** — XLA can constant-fold them and
make a kernel look faster than it is.

**Size inputs to ≥4× v5e VMEM (32 MiB → ≥128 MiB).** This is what
actually defends per-call BW% accounting under unroll mode. Smaller
inputs fit on chip; XLA can keep intermediates in VMEM across chained
calls, so the program crosses HBM once for k calls and per-call BW%
inflates by ~k. The unroll harness wraps chained values in
`jax.lax.optimization_barrier` to block math fusion (e.g. `(((x+1)+1)+1)`
collapsing to `x+3`), but the barrier doesn't force materialization
through HBM — if the working set fits in VMEM the same accounting bug
still happens. Sane input sizing is the only mechanism that prevents
it. Suite-default tests (`tests/test_*_suite.py`) pin each suite's
no-flag shape against this floor.

## Tests

- `tests/correctness/` — `allclose` vs `naive`. Pallas paths use
  `interpret=True` so they run on CPU.
- `tests/perf/` — threshold-gated regression tests, TPU-only
  (`@pytest.mark.tpu`, `@pytest.mark.perf`).

Run: `uv run pytest`. CPU-only runs skip TPU-marked tests.

## Roofline targets (starting points)

| Regime              | Target                  |
| ------------------- | ----------------------- |
| Memory-bound        | ≥ 80% HBM BW            |
| Compute-bound       | ≥ 80% MFU               |
| Mixed/attention-ish | ≥ 70% of speed-of-light |

Tighten once a kernel has been measured and you know what's reachable on
v5e for that shape.

## Conventions

- Python ≥ 3.12. `from __future__ import annotations` at the top of every
  module that has type annotations.
- Ruff: line length 100, rules `E F I B UP N RUF SIM`. Pyright in `basic` mode.
- Type hints expected on public functions.
- Hardware constants are sourced from the v5e announcement and `jax-ml/maxtext`;
  cite the source in a comment if you change one.
- bf16 is the default working dtype; f32 only when a kernel needs it.

## Commit style

Conventional commits, scoped by op when relevant:
`feat(scale): add Pallas variant`, `perf(rmsnorm): tile by (8, 128)`,
`docs(v5e): correct HBM bandwidth`. A perf-relevant commit should bundle
the kernel edit, the new `bench_history/<op>/<timestamp>.json`, and the
`PERF.md` update.

Co-author trailer (when applicable): use the bare RFC form
`Co-Authored-By: <Name> <email>`. **No parentheticals or annotations
inside the trailer** (e.g. `Claude (1M context)` is wrong; some downstream
parsers reject anything that isn't `Name <email>`).

## Things to avoid

- Don't add a new top-level directory without a clear reason — the layout
  in `README.md` is the contract.
- Don't skip `naive.py` for a new op. Correctness oracle first, always.
- Don't tune `naive.py` — that defeats its purpose.
- Don't put per-attempt logs in `PERF.md` — that's `git log`'s job.
- Don't hardcode hardware numbers outside `benchmarks/roofline.py`.
- Don't gitignore `bench_history/` — trend tracking depends on it.
- Don't commit `bench_history/` runs from non-canonical hardware (only
  real v5e results belong in trend tracking).
- Don't use `jnp.zeros` / `jnp.ones` as bench inputs (constant-folding).

## Roadmap

Kernel ordering and per-kernel rationale live in
[`docs/curriculum.md`](docs/curriculum.md) — the source of truth for
*what's planned next* and *why*. When a kernel lands, flip its entry there
from `[planned]` to `[done]` and link the new `PERF.md`. Don't restate the
list of upcoming kernels in this file; it would drift.

Built so far: `scale` (memory-bound primer + harness validation).
