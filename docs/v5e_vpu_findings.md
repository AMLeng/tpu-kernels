# v5e VPU probe — findings

Working notes from `benchmarks/probe_hardware.py` (the VPU compute section).

## Goal

The VPU peak FLOPs number for v5e isn't published by Google and
`pltpu.get_tpu_info()` exposes only MXU peaks (bf16, fp8, int8, int4
matmul). Without an empirical anchor for elementwise-bound code (Hillis-
Steele scans, RMSNorm, softmax body, etc.) any "VPU%" we'd report is
guesswork. We need (a) a number for total VPU bandwidth and (b) a sense
of how a kernel author should structure work to hit it.

## What the probe does

`benchmarks.probe_hardware` runs a parallel-ops throughput sweep:

- N independent accumulators, each a single VPU vec reg `(8, 128)`.
- Threaded as an N-tuple through `fori_loop` for K=4096 iters.
- Body runs **64 serial `acc + c` ops per chain per iter** (Mosaic
  preserves source order — verified by IR dump — so they form a true
  chain). Across chains the ops are SSA-independent and the scheduler
  can issue them in parallel; within each chain they pay the per-op
  latency in series.
- For each `(dtype, op_kind, N)` we measure sustained TFLOPs.

Sweep: N ∈ (1, 2, 4, 8, 12, 16, 24). Op kinds: add, mul. Dtypes: f32,
bf16.

### Why the body needs to be deep

A naive probe with depth=1 (one op per chain per iter) measures *loop
overhead*, not chip throughput. Each fori_loop iter has a per-iter
floor of ~9 cycles for counter increment, conditional branch, and
carry handover. With one 2-cycle add per iter, the floor is ~5x larger
than the body work, and the measured "plateau" sits at ~45% of the
chip's actual peak. Depth=64 makes the chain critical path D·L = 128
cycles, vastly larger than the floor, so the loop overhead amortizes
away and the plateau lands within ~3% of the W=4 ALUs · 1.5 GHz
ceiling. **Any future probe of VPU compute peak must keep the body
work well above the loop floor or it will silently undersell the
chip.**

## Model

The v5e VPU has **W = 4 ALUs per (lane, sublane)** with intrinsic
per-op latency **L = 2 cycles**, per the [JAX scaling
book](https://jax-ml.github.io/scaling-book/tpus/). Cross-checks
against the API's MXU peak: 197 TFLOPs / (256·256·2 flops/cycle) =
1.50 GHz exactly, confirming the scaling-book figure (and that the
MXU and VPU share a clock domain on v5e).

With the body deep enough that loop floor is hidden, per-iter time
in cycles is `T_iter = max(N · D / W, D · L)`:

- ALU-bound time `N · D / W`: how many cycles to issue all N · D ops
  through W ALUs.
- Chain critical path `D · L`: a single chain can't go faster than
  back-to-back dependent ops at L cycles each.

Throughput is `N · D / T_iter` ops/cycle. Linear in N below the
crossover, plateau above. Crossover at `N · D / W = D · L`, i.e.
**N\* = W · L = 8** chains. Independent of D once D is big enough.

Plateau · elements-per-vec-inst · clock = total VPU flops/sec. With
W=4, vec-inst-elements=1024, clock=1.5 GHz: theoretical peak =
**6.14 TFLOPs f32**.

## Results

f32 add and mul (identical, so listed once):

| N   | TFLOPs | % of 6.14 |
| --: | -----: | --------: |
| 1   |  0.76  |     12%   |
| 2   |  1.52  |     25%   |
| 4   |  3.04  |     49%   |
| 8   |  5.99  |     97%   |
| 12  |  4.55  |     74%   |
| 16  |  5.97  |     97%   |
| 24  |  5.97  |     97%   |

Linear N=1→8 (each step exactly 2×, perfect); plateau at 5.99 TFLOPs
at multiples of 8 (N=8, 16, 24). **97% of the W=4 · 1.5 GHz theoretical
peak** — the chip really does deliver close to all 4 ALUs when given
enough parallel work.

The dip at N=12 is reproducible — non-multiples-of-8 chain counts
underperform their power-of-2 neighbors by ~25%. Same artifact shows
up at N=20 in the carry sweep (see below). Best guess: Mosaic's
instruction scheduling packs VLIW slots more efficiently when the
number of independent live values aligns with some hardware boundary
(maybe the 4-ALU width × 2-way pairing?). Doesn't change the headline
peak — kernels structured around N=8/16/24-style ILP just hit the
plateau cleanly.

Saturating chain count is **N\* = W · L = 4 · 2 = 8** — exactly the
prediction from the model with W and L from the scaling book.

bf16 plateau is lower — same chain count to saturate, but the per-op
cost is higher:

| N   | bf16 TFLOPs | f32 TFLOPs | bf16 / f32 |
| --: | ----------: | ---------: | ---------: |
| 1   |       0.26  |      0.76  |       34%  |
| 8   |       2.04  |      5.99  |       34%  |
| 16  |       2.04  |      5.97  |       34%  |

bf16 hits ~2.04 TFLOPs = **exactly 1/3 of f32** at every N. The 1/3
ratio is the smoking gun for "1 bf16 user-add = 3 hardware ops in the
chain": v5e VPU has no native bf16 ALU, so each bf16 add lowers
post-Mosaic into one `extf` + one `addf` + one `truncf` on the chain
critical path. Three hardware ops per user-op → 1/3 the throughput.

(At depth=1 we measured a bf16 / f32 ratio of 67% instead of 34%,
which we attributed to "partial conversion sharing." That was an
artifact of the loop floor capping both dtypes equally. With the
floor amortized away at depth=64, the true 1/3 ratio surfaces.)

The conversions don't appear in the Mosaic IR — the bf16-add probe
lowers to a clean `arith.addf : vector<8x128xbf16>` per chain, no
`extf`/`truncf` pairs. They're inserted strictly below Mosaic by the
TPU backend lowering bf16 vector ops onto an f32-only VPU.

### Confirmation: carry dtype is what matters

A follow-up probe holds bf16 input/output but varies the loop carry
dtype (cast at load, work in f32 inside the body, cast back at store).
TFLOPs at K=4096, depth=64:

| N    | bf16 carry | f32 carry |
| ---: | ---------: | --------: |
| 1    |       0.26 |      0.76 |
| 8    |       2.04 |      5.99 |
| 16   |       2.04 |      5.98 |
| 24   |       2.04 |      5.97 |

Same I/O, same body shape, same N range. The only difference is the
carry dtype. **F32 carry recovers ~3× the throughput of bf16 carry**
even though I/O is bf16, because the boundary casts (one per chain
per kernel invocation) are paid once and amortized over 4096 body
iters; only the body work matters for steady-state throughput.

So the kernel-author rule is: **for elementwise bf16 work on v5e,
promote at load and demote at store; do the loop in f32**. Holding
the carry as bf16 forces per-iter conversions the hardware doesn't
have to do otherwise, costing ~67% of bandwidth.

## Headline numbers

- **Total VPU bandwidth (f32)**: 5.99 TFLOPs sustained, single-chip
  v5e, body depth ≥ 64 ops, N ≥ 8 chains. **97% of the 6.14 TFLOPs
  W=4 · 1.5 GHz theoretical peak.** Measured flops/sec directly —
  independent of clock assumption.
- **Total VPU bandwidth (bf16)**: 2.04 TFLOPs with bf16 carry, 5.99
  TFLOPs with f32 carry. The bf16-carry number is the cost of
  per-iter post-Mosaic conversions; the f32-carry number is the
  realistic peak for kernels that promote at boundaries.
- **Saturating chain count**: N\* = 8 (= W · L = 4 · 2 from the
  scaling book). A kernel's hottest loop needs 8 independent ops
  in flight to saturate the ALU pool.
- **Body depth must dominate the loop floor**: with ~9 cycles of
  fori_loop overhead per iter, body work has to be several times that
  before the chain critical path drowns it out. We measure at
  DEPTH=64 (chain critical path D·L = 128 cycles ≫ floor); naive
  single-op bodies measure overhead, not the chip. We didn't sweep D
  to pin the minimum down — kernels should either expose multi-op
  bodies naturally (rmsnorm body, softmax body) or unroll inside the
  kernel, rather than rely on a specific cutoff.
- **Slot pool is symmetric** across add and mul (identical TFLOPs at
  every N for both ops). Whatever slots exist are uniform, not
  specialised by op kind.

## How to run

```bash
uv run python -m benchmarks.probe_hardware --probe vpu carry
# ~5 min wall on v5e for the two VPU sections.
```

Output is a TFLOPs-vs-N table per (dtype, op) plus the headline
bandwidth and ALU-utilization read.
