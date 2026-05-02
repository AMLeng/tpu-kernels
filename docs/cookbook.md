# Cookbook

Copy-pasteable command reference for common workflows. The narrative
README's "Iterating on an existing kernel" links here from its commands.

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
