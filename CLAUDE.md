# CLAUDE.md

Agent-facing notes for the `tpu-kernels` repo. Human-facing docs and the
copy-pasteable cookbook live in `README.md`.

## What this project is

JAX / Pallas kernels targeting TPU v5e, with a benchmarking harness built
around roofline analysis (MFU% and HBM bandwidth %). Single-host v5e-8
today; multi-host planned.

This commit is project scaffolding only — directory layout, build config,
hardware reference, and the per-op contract. The bench harness and the
first op (`scale`) land in the next commit.

## Per-op workflow

Every op lives at `src/tpu_kernels/ops/<op>/` and ships in up to three variants:

1. `naive.py` — readable JAX, the correctness oracle. **Never tuned.**
2. `xla.py` — `@jax.jit`-wrapped, tuned plain JAX. First speed-of-light attempt.
3. `pallas.py` — only if `xla.py` fell short of the roofline target.

`PERF.md` in the op dir is forward-looking, not a log. Four lines:
`Target` / `Current` / `Bottleneck` / `Next`. Use `docs/PERF_TEMPLATE.md`.
Per-attempt history goes in `git log`, not in `PERF.md`.

If `xla.py` already hits the target, **stop**. Don't write Pallas just
because the slot exists.

## After any change

```bash
uv run ruff check . && uv run ruff format . && uv run pyright
uv run pytest
```

All four must be clean before committing. CI runs the same checks
(`.github/workflows/ci.yml`) so a green local run should mean a green PR.

## Tests

- `tests/correctness/` — `allclose` vs `naive`. Pallas paths use
  `interpret=True` so they run on CPU.
- `tests/perf/` — threshold-gated regression tests, TPU-only
  (`@pytest.mark.tpu`, `@pytest.mark.perf`).

Run: `uv run pytest`. CPU-only runs skip TPU-marked tests.

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
`docs(v5e): correct HBM bandwidth`.

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

## Roadmap

No ops built yet. Planned next, in rough order of complexity:
scale → RMSNorm → softmax → tiled matmul → distributed primitives →
flash attention → paged attention → ragged paged → MoE. Each follows
the op-per-directory shape above.
