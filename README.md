# tpu-kernels

A repo for learning to write performant TPU v5e kernels — in plain JAX
where it's enough, in Pallas where it isn't. Each kernel is judged against
the v5e roofline: MFU% on compute-bound work, HBM bandwidth % on
memory-bound work, and the fraction of speed-of-light hit overall.

The repo is built around three ideas:

1. **Kernels follow a curriculum** — see
   [`docs/curriculum.md`](docs/curriculum.md) for what's been built, what's
   planned next, and why each kernel is in the list. The order is hybrid:
   practice JAX where XLA can already approach speed-of-light, drop to
   Pallas where it can't.
2. **Each operation has multiple implementations** — a readable `naive` for
   correctness, a tuned `xla` (plain JAX), and a `pallas` kernel only when
   XLA can't reach the target. They live side by side and bench against the
   same roofline so you can see whether dropping to Pallas was worth it.
3. **Benchmarking is part of the inner loop**, not a separate audit. One
   command prints a roofline table, dumps lowered HLO, or attaches an xprof
   trace. Results land in `bench_history/` so trends are tracked.

---

## Quick start

On a TPU v5e VM:

```bash
git clone <this repo> && cd tpu-kernels
uv sync --extra tpu
uv run pytest                                # correctness on CPU via interpret=True
uv run python -m benchmarks.suites.scale     # bench scale-by-2; prints roofline table
```

On a CPU machine for development (Pallas runs in `interpret=True` mode for
tests; perf numbers will be meaningless):

```bash
uv sync
uv run pytest
```

CI (`.github/workflows/ci.yml`) runs `ruff` + `pyright` + `pytest` on every
push and PR.

---

## Repo layout

```
src/tpu_kernels/
  ops/<op>/           one directory per operation
    naive.py          readable JAX — correctness oracle, never tuned
    xla.py            tuned plain JAX — first speed-of-light attempt
    pallas.py         Pallas kernel — only if XLA fell short
    PERF.md           target, current %, bottleneck, next idea

benchmarks/
  runner.py           warmup + N timed iters + stats
  roofline.py         v5e peak constants, MFU & BW % math
  compare.py          N-variant comparison, HLO dump, JSON history
  suites/             one bench script per op

tests/
  correctness/        allclose vs naive; uses interpret=True on CPU
  perf/               threshold-gated regression tests, TPU-only

bench_history/<op>/   JSON results timestamped + git-stamped
docs/                 curriculum, hardware reference, kernel patterns
```

New top-level directories under `src/tpu_kernels/` are added when a real
consumer lands (e.g. multi-host helpers when the first distributed op
needs them) — not in advance.

---

## Iterating on an existing kernel

The loop, made concrete with the `scale` op:

**1. Read the current state.**

```bash
cat src/tpu_kernels/ops/scale/PERF.md
```

That four-line file tells you the target, where the kernel is now, what's
believed to be the bottleneck, and what to try next.

**2. Make a change** in `xla.py` or `pallas.py`.

**3. Verify correctness.** Fast — runs on CPU via `interpret=True`:

```bash
uv run pytest tests/correctness/test_scale.py
```

**4. Benchmark.** Run on a v5e:

```bash
uv run python -m benchmarks.suites.scale
```

Prints a table like:

```
=== scale (1 chip) ===
  flops=6.711e+07  bytes=2.684e+08  intensity=0.2 F/B
  ridge point (peak_flops/peak_bw) = 240.5 F/B  →  memory-bound

  variant                med(ms)   p99(ms)    MFU%     BW%    binds    SoL%
  -------------------------------------------------------------------------
  xla                      0.412     0.414    0.0%   79.4%   memory   79.4%
  pallas                   0.405     0.406    0.0%   80.9%   memory   80.9%

  wrote bench_history/scale/20260501T120000Z.json
```

The numbers and JSON record are tagged with the current git SHA, so trend
tracking is just `ls bench_history/scale/`.

**5. Inspect the HLO** when you want to know what XLA actually emitted:

```bash
uv run python -m benchmarks.suites.scale --dump-hlo
```

**6. Profile on device** when the bench number doesn't match your model
of the kernel:

```bash
uv run python -m benchmarks.suites.scale --profile-dir /tmp/scale_trace

# Inspect the trace. Pick whichever you prefer:
uv run xprof /tmp/scale_trace                     # full XProf UI on :8791
# or drag /tmp/scale_trace/plugins/profile/*/*.trace.json.gz into
# ui.perfetto.dev for a quick timeline view (no install, no server).
```

`xprof` is in the `dev` group, so a plain `uv sync` brings it in. It
gives you the Op Profile, Memory Profile, Trace Viewer, and HLO graph —
the views that actually report MFU / HBM% / step time. Perfetto only
shows the timeline; use it for quick "what ran when" sanity checks.

**7. Sweep a config knob.** For Pallas kernels, this is usually block
shape. Use `--sweep-block`, which Cartesian-products the two lists,
filters out divisibility-invalid pairs, and prints a perf-sorted
leaderboard with the winner marked:

```bash
uv run python -m benchmarks.suites.scale \
  --sweep-block 8,16,32,64,128,256 128,256,512,1024
```

One JSON lands in `bench_history/scale/` per sweep — `kind: "sweep"`,
with each variant's structured config preserved alongside its timing.

**8. Update `PERF.md`** with the new current %, new bottleneck hypothesis,
and new next-thing-to-try. The history of what was tried lives in `git log`;
`PERF.md` is the forward-looking state.

**9. Commit.** A single change should bundle: the kernel edit, any new
`bench_history/` JSON it produced, and the `PERF.md` update.

---

## Adding a new op

The shape is fixed, which keeps the bench/test/history layer plug-and-play.
For an op called `<name>`:

```bash
mkdir -p src/tpu_kernels/ops/<name>
```

Then create:

```
src/tpu_kernels/ops/<name>/__init__.py     # re-exports the variants
src/tpu_kernels/ops/<name>/naive.py        # def <name>(x, ...) — obviously correct
src/tpu_kernels/ops/<name>/xla.py          # tuned plain JAX, jit'd
src/tpu_kernels/ops/<name>/pallas.py       # only if XLA fell short
src/tpu_kernels/ops/<name>/PERF.md         # use docs/PERF_TEMPLATE.md
tests/correctness/test_<name>.py           # parametrize over variants vs naive
benchmarks/suites/<name>.py                # argparse CLI calling compare(...)
```

The simplest reference is `ops/scale/` — copy it, rename, replace the body.
You don't have to write `pallas.py`. If `xla` already hits the target,
note "xla is good enough" in `PERF.md` and move on.

Run the new tests + bench:

```bash
uv run pytest tests/correctness/test_<name>.py
uv run python -m benchmarks.suites.<name>
```

---

## Cookbook

```bash
# ── Setup ─────────────────────────────────────────────────────────────
uv sync --extra tpu                              # TPU VM
uv sync                                          # CPU dev box

# ── Tests ─────────────────────────────────────────────────────────────
uv run pytest                                    # all correctness tests
uv run pytest tests/correctness/test_scale.py    # one op
uv run pytest -k pallas                          # all pallas variants
uv run pytest -m "not perf"                      # skip TPU-only perf tests

# ── Benchmarks ────────────────────────────────────────────────────────
uv run python -m benchmarks.suites.scale         # default shape
uv run python -m benchmarks.suites.scale --m 16384 --n 16384
uv run python -m benchmarks.suites.scale --dtype bf16
uv run python -m benchmarks.suites.scale --block 512 512
uv run python -m benchmarks.suites.scale --dump-hlo
uv run python -m benchmarks.suites.scale --profile-dir /tmp/trace
uv run xprof /tmp/trace                          # open the trace at :8791

# Cartesian sweep over (bm, bn) — one JSON record, sorted leaderboard,
# divisibility-invalid shapes skipped instead of crashing the loop.
uv run python -m benchmarks.suites.scale --sweep-block 8,16,32,64,128,256 128,256,512,1024

# Diff the two most recent bench records for an op
ls -t bench_history/scale/*.json | head -2 | xargs diff

# ── Code quality ──────────────────────────────────────────────────────
uv run ruff check .                              # lint
uv run ruff format .                             # auto-format
uv run pyright                                   # type-check
```

---

## What "good" looks like

For each op, set a target percentage of speed-of-light in `PERF.md` and
iterate until you hit it. Reasonable starting targets:

| Regime               | Target                             |
| -------------------- | ---------------------------------- |
| Memory-bound op      | ≥ 85% HBM bandwidth                |
| Compute-bound op     | ≥ 80% MFU                          |
| Mixed/attention-ish  | ≥ 70% of speed-of-light            |

These are starting points. Tighten them when you've actually measured a
kernel and know what's reachable on v5e for that shape.

---

## Hardware

TPU v5e. See `docs/v5e_hw.md` for the per-chip peaks fed into the roofline
math (197 TFLOP/s bf16, 819 GB/s HBM, 32 MiB VMEM, 128×128 MXU). Single-host
v5e-8 works today; multi-host (`jax.distributed.initialize`) is planned.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
