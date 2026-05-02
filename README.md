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

## Human-facing surface

What a human reads or edits when iterating in this repo:

- This `README.md` and everything under [`docs/`](docs/)
- Everything under `src/tpu_kernels/ops/<op>/` — kernel modules and `PERF.md`
- `benchmarks/suites/<op>.py` — the per-op CLI
- `bench_history/<op>/` — JSON records of past runs (read-only)

Everything else is harness; agents maintain it.

---

## Performance targets

For each op, set a target percentage of speed-of-light in `PERF.md` and
iterate until you hit it. Reasonable starting targets:

| Regime               | Target                             |
| -------------------- | ---------------------------------- |
| Memory-bound op      | ≥ 80% HBM bandwidth                |
| Compute-bound op     | ≥ 80% MFU                          |
| Mixed/attention-ish  | ≥ 70% of speed-of-light            |

Targets sit just below the v5e plateau — clear the bar and the kernel
is done.

---

## Iterating on an existing kernel

The loop, made concrete with the `scale` op:

**1. Read the current state.**

```bash
cat src/tpu_kernels/ops/scale/PERF.md     # target, current %, bottleneck, next
```

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

**6. Profile on device** when the bench number surprises you:

```bash
uv run python -m benchmarks.suites.scale --profile-dir /tmp/scale_trace
uv run xprof /tmp/scale_trace                     # full UI on :8791
# or drag /tmp/scale_trace/plugins/profile/*/*.trace.json.gz into
# ui.perfetto.dev for a quick timeline.
```

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

Usually agent work. The scaffolding contract — directory shape,
required files, what `__init__.py` re-exports — lives in
[`CLAUDE.md`](CLAUDE.md)'s "Adding a new op" section, which is what
the agent reads. The next op to build is the first `[planned]` entry
in [`docs/curriculum.md`](docs/curriculum.md).

---

## Cookbook

Copy-pasteable command reference: [`docs/cookbook.md`](docs/cookbook.md).

---

## Hardware

TPU v5e. See `docs/v5e_hw.md` for the per-chip peaks fed into the roofline
math (197 TFLOP/s bf16, 819 GB/s HBM, 32 MiB VMEM, 128×128 MXU). Any
single-host v5e-N works today; multi-host (`jax.distributed.initialize`)
is planned.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
