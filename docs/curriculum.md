# Curriculum

The planned progression of kernels for learning to write performant TPU
v5e work in both plain JAX and Pallas. Organized in stages: single-chip
primitives, collectives, MLP layers, attention, and MoE. JAX-first
where XLA can already approach speed-of-light, Pallas where XLA hits a
wall or compute-comm overlap is the lever each stage cares about.

This file is the source of truth for *what's planned next* and *why*.
Per-attempt history lives in `git log` and `bench_history/<op>/*.json`;
this file only tracks kernel-level status and the original rationale.

Entries are numbered per stage (`A.1–A.6` for Stage A, `B.1–B.3` for
Stage B, etc.). The prefix doubles as the disambiguator when one
entry's body refers to another, and stage letters leave room to insert
new stages later without renumbering.

## How to use this

Pick the first kernel not marked **[done]** and follow CLAUDE.md's
per-op workflow:

1. Read its annotation here for the *why*.
2. Create the op directory under `src/tpu_kernels/ops/<name>/`.
3. Build the variants listed in the kernel's entry. Hit the regime target
   from CLAUDE.md (or a tighter one if `PERF.md` calls for it).
4. Update the entry below to **[done]** with a pointer to its `PERF.md`.

The stage-to-stage dependency is real: Stage C's layer-level ops
compose Stage B's collectives, and Stage D's attention reuses both.
Within a stage, entries can ship in any order the contract supports.

## Stage A — Single-chip primitives

The on-chip toolkit: VMEM, scratch refs, MXU, online reductions,
data-dependent indexing, parallel prefix. Each kernel below introduces
one Pallas concept the rest of the spine reuses.

### A.1. scale — **[done]**

Memory-bound elementwise (×2). Pure hello-world: validates the harness
end-to-end and proves `pallas_call`, `BlockSpec`, the runner, and the
roofline math agree on a trivial kernel. No JAX or Pallas lesson beyond
"the toolchain works."

See [`ops/scale/PERF.md`](../src/tpu_kernels/ops/scale/PERF.md).

### A.2. rmsnorm — **[done]**

Memory-bound row-wise reduction. **Why Pallas was needed even though
the math obviously fuses**: XLA reads the input twice no matter how
you write it — once for the mean-square reduction, once again for the
normalize step — capping the XLA variant at 55% HBM BW. Pallas pulls
each row-tile into VMEM once and reuses that single read for both the
reduction and the output, hitting 80.2% under device timing. The 25pp
delta is the whole reason Pallas exists for this kernel class. Also
the first real exercise of `pallas_call` + grid + `BlockSpec` for a
row-tiled reduction (one grid axis over rows, full hidden dim per
tile so the reduction lives inside one block, `scale` loaded once via
a constant index_map).

See [`ops/rmsnorm/PERF.md`](../src/tpu_kernels/ops/rmsnorm/PERF.md).

### A.3. softmax — **[done]**

Memory-bound stable-form chain (max → sub → exp → sum → div). **Two
levers, stacked**:

- *JAX-side*: rewrite as **online softmax** — one streaming pass
  carrying `(running max, running sum)` and renormalizing on each new
  max. Folds two reductions into one and lets XLA fuse the divide on
  the reduction's output. Lifted naive 42.8% → xla 55.6% HBM BW.
- *Pallas*: stalls at the same 55% wall as rmsnorm-xla, for the same
  reason — XLA reads x twice (once for the paired reduction, once for
  the divide). Tile by full rows so the `(bm, hidden)` slab lives in
  VMEM end-to-end; max/sub/exp/sum/div all run on the resident copy.
  Hits 80.2% BW. Once the row is resident the online recurrence is
  unnecessary — single-pass naive form inside the kernel suffices, and
  the online algorithm becomes the lesson cashed in at flash attention.

Pallas mechanics mirror rmsnorm exactly: one grid axis over rows, full
hidden dim per tile so the reduction lives inside one block.

See [`ops/softmax/PERF.md`](../src/tpu_kernels/ops/softmax/PERF.md).

### A.4. matmul (tiled bf16) — **[done]**

The documented exception to "skip Pallas when XLA hits target." At
(8192, 8192, 8192) bf16 the v5e MXU is so close to its plateau that
naive, xla, and pallas all land at ~95% MFU within noise; pallas does
not beat xla and isn't expected to. It ships anyway as the teaching
artifact for the tiled-MXU pattern flash attention will reuse.

**Why** (`xla.py`): a one-line `jnp.matmul`. `Precision.DEFAULT` on bf16
inputs already drives the MXU at native rate (bf16 multiplies, f32
accumulator), so the JAX baseline clears the 80% MFU bar with no
rewrite. Higher precisions (`HIGH` / `HIGHEST`) emulate wider products
on the MXU — they're the `--dump-hlo` exercise the curriculum cares
about, not a default the variant uses.

**Why** (`pallas.py`): first MXU exposure inside a Pallas kernel, via
`jnp.dot(..., preferred_element_type=jnp.float32)`. Teaches the 3-axis
grid (M/N/K), a VMEM-resident f32 scratch accumulator, and the
`pl.when(program_id(2) == 0)` / `pl.when(program_id(2) == num_programs(2)-1)`
guards that zero-init the accumulator on the first K-step and flush it
to the output on the last. The kernel-visible block at the default
shape lands at (1024, 1024, 512) — the (1024, 1024, 1024) cube that
would in theory go further OOMs the 16 MiB scoped VMEM limit, so K=512
is the sweet spot. The 128×128 MXU sublane/lane tile sits below this
block and is the compiler's concern.

See [`ops/matmul/PERF.md`](../src/tpu_kernels/ops/matmul/PERF.md).

### A.5. embedding lookup — **[done]**

Memory-bound gather: read selected rows of a parameter table at
integer indices. **What turned out to matter**: not the kernel choice
but the *layout* the table lives in. Ships as two ops, one per layout:

- `embedding_lookup` ([`PERF.md`](../src/tpu_kernels/ops/embedding_lookup/PERF.md))
  takes the natural 2-D `T(8, 128)(2, 1)` form. HBM stores the array
  as 2 KiB tiles holding 8 rows interleaved per tile, and Mosaic only
  emits tile-aligned DMAs against tiled refs — so a single-row read
  isn't expressible from Pallas (the lowering pass rejects sub-tile
  slice ops; sub-tile reads would need scatter-gather across the
  tile's striped bytes, which the hardware can do but Mosaic doesn't
  generate). The Pallas variant is structurally capped near 12.5% BW
  (slab-and-permute) and lands at ~11%. XLA's `gather_custom_fusion`
  (kCustom, C++) reaches ~33% by mechanisms unreachable from JAX
  surface — Pallas loses to XLA on this layout.
- `embedding_lookup_packed` ([`PERF.md`](../src/tpu_kernels/ops/embedding_lookup_packed/PERF.md))
  takes a 4-D `(vocab, hidden//1024, 8, 128)` form whose natural
  Mosaic layout is byte-equivalent to 1-D `T(1024)(128)(2, 1)`. Each
  row's bytes are contiguous in HBM and span an integer number of
  tiles, so per-row DMAs are tile-aligned. Pallas does explicit
  read-many / write-one DMAs (per-row reads HBM→VMEM, one bulk
  VMEM→HBM per grid step), reaching ~69% and edging out XLA's own
  staged gather (~65%).

### A.6. segment cumsum — **[planned]**

Memory-bound scan: cumulative sum within each segment of a segmented
input (boundaries given by `segment_ids`). **Why Pallas was needed**:
teaches the parallel-prefix / `lax.associative_scan` pattern with a
carry across tiles, plus the segment-boundary reset that distinguishes
segmented scan from plain cumsum. The carry-across-tiles mechanic is
what scan-heavy kernels reuse, and segment-aware offsets are the
prerequisite for every ragged kernel in Stage D — knowing where each
segment starts in a packed buffer is exactly this primitive. As with
matmul and embedding lookup, XLA's `lax.associative_scan` is
competitive on bandwidth; Pallas earns its place on the technique.

## Stage B — Collectives (single-host v5e-N over ICI)

The cross-chip toolkit: ICI bandwidth, semaphores,
`pltpu.make_async_remote_copy`, `pltpu.CompilerParams(collective_id=...)`.
Each kernel below has trivial math — the lesson is the communication
pattern. `all_reduce` is **not** a separate kernel: it composes as
`reduce_scatter ∘ all_gather`, which is the decomposition modern stacks
exploit for compute-comm overlap (Megatron-SP, sequence-parallel TP).

### B.1. all_gather — **[planned]**

Data-only collective: each chip ends up with a concatenation of every
chip's slice. **Why first**: pure ICI transfer pattern, no math to
reason about, so the lesson lands on `make_async_remote_copy`, ring
topology, and double-buffered pipelined reception.

### B.2. reduce_scatter — **[planned]**

Reduction-then-shard: each chip ends up with one slice of the
elementwise sum across all chips. **Why second**: builds on
`all_gather`'s transfer pattern, adds an on-chip reduction at each
step. Together with `all_gather` it composes `all_reduce`, and the
two-kernel decomposition is the lever for compute-comm overlap in TP
layers.

### B.3. all_to_all — **[planned]**

General routing: each chip sends a different slice to each other chip
and receives a different slice from each. **Why third**: hardest
routing pattern; the prerequisite for sequence-parallel attention
(Ulysses-style axis switch between sequence-sharded and head-sharded
states) and MoE expert dispatch / combine.

## Stage C — MLP layer (TP and FSDP, fwd and bwd)

> **Multi-host scale-out gates Stage C.** Stage B ships on single-host
> v5e-N (8 chips over ICI). Stage C's FSDP pattern only earns its lesson
> at scale, so the harness, roofline, and benches must extend across
> hosts before the MLP stage starts. This flips the repo's hardware
> scope from "single-host today, multi-host planned" to "multi-host
> live."

The first *layer-level* ops in the curriculum. Each variant composes
already-built kernels (matmul + activation + collective) into an
end-to-end MLP, with compute-comm overlap as the Pallas lever. Pallas
earns its place not by beating JAX's individual matmul or collective —
it doesn't — but by overlapping them. JAX as a sequential composition
leaves comm on the critical path; Pallas with async DMA hides it.

`PERF.md` targets here are end-to-end *layer MFU*. The bench captures
total FLOPs and total bytes (incl. cross-chip traffic), not
kernel-local figures.

### C.1. mlp_tp_fwd — **[planned]**

Tensor-parallel MLP forward. Column-parallel up-proj (matmul), GeLU /
SwiGLU, row-parallel down-proj (matmul), `all_reduce`
(`reduce_scatter ∘ all_gather`) after. **Why**: the canonical Megatron
pattern. Teaches column- and row-parallel matmul layouts and the
all-reduce-after-down-proj structure every TP transformer ships.

### C.2. mlp_tp_bwd — **[planned]**

Tensor-parallel MLP backward. Backprop through row-parallel down-proj,
activation grad, backprop through column-parallel up-proj, `all_reduce`
of the input gradient. **Why**: introduces bwd structure (recompute or
stash, gradient accumulation) and the second `all_reduce` per layer
that bwd injects. Symmetric to fwd in shape but doubles comm traffic
— non-trivial roofline.

### C.3. mlp_fsdp_fwd — **[planned]**

Fully-sharded data-parallel MLP forward. Parameters live sharded; an
`all_gather` reconstitutes them right before the matmul, then they're
released. **Why**: introduces parameter-sharding. The overlap lesson
is hiding the `all_gather` behind the previous layer's compute, which
the bench captures end-to-end.

### C.4. mlp_fsdp_bwd — **[planned]**

Fully-sharded data-parallel MLP backward. Re-`all_gather` parameters
for the bwd pass, compute grads, `reduce_scatter` grads back to the
sharded layout. **Why**: introduces the `reduce_scatter` step that
distinguishes FSDP from naive DP, and shows the second `all_gather`
of the same parameters (or a stashed copy) that bwd requires.

## Stage D — Attention

Eight kernels covering the production lifecycle as a 2×2 of
training-vs-serving × single-vs-ragged, with the inner axis being
fwd/bwd for training and prefill/decode for serving. The first kernel
carries the "XLA can't fuse softmax into the matmul" wall narrative —
it ships naive + xla + pallas, where xla materializes O(N²) and BW%
collapses, and pallas (flash) is the answer. Subsequent attention
kernels ship naive + pallas only.

### D.1. attention, no-cache, single-seq, fwd — **[planned]**

Flash attention forward on a single sequence. **Why first**: the wall.
Online softmax + streaming K/V tiles + MXU inside Pallas + scratch-
resident running statistics (max, lse). Teaches the core flash pattern
every later attention kernel reuses.

### D.2. attention, no-cache, single-seq, bwd — **[planned]**

Flash attention backward on a single sequence. Two-pass split: a dQ
pass tiled over Q with K/V streamed; a dK/dV pass tiled over K with Q
streamed. Recomputes attention probabilities from the stashed LSE,
plus a small `D = rowsum(dO * O)` pre-kernel. **Why**: introduces
bwd's split-kernel structure and the recompute-from-stash pattern,
with no new Pallas concept beyond what step D.1 taught.

### D.3. attention, no-cache, ragged, fwd — **[planned]**

Flash attention forward over a packed batch of variable-length
sequences. **Why**: introduces segment-aware tiling — `segment_ids`
on the Q outer loop, K/V streaming bounded by segment, masking at
segment boundaries. Realistic packed training shape. Builds on the
segment cumsum pattern from step A.6 for offset construction.

### D.4. attention, no-cache, ragged, bwd — **[planned]**

Flash attention backward over a packed batch. **Why**: hardest
training kernel. Combines bwd's split structure with raggedness;
gradient accumulation must stay segment-local on both the dQ and
dK/dV passes.

### D.5. attention, paged, single-seq, prefill — **[planned]**

Paged attention prefill on a single prompt. **Compute-bound** (full
prompt at once against the cache). **Why**: introduces `block_table`
indirection on K/V loads — K and V live in fixed-size pages addressed
by a per-sequence block_table, so the kernel does a gather-then-attend
pattern rather than streaming a contiguous K/V tile. The gather
mechanics are the same as embedding lookup at step A.5, applied to K/V
pages instead of an embedding table.

### D.6. attention, paged, single-seq, decode — **[planned]**

Paged attention decode on a single sequence. **Memory-bound** (Q is
one token; K/V is the whole cache). **Why**: completely different
roofline regime from prefill. The lesson is exploiting head-parallelism
across the (typically tiny) decode batch and saturating HBM bandwidth
on the K/V cache reads.

### D.7. attention, paged, ragged, prefill — **[planned]**

Paged attention prefill across a multi-tenant batch. **Why**: combines
the segment-aware tiling of step D.3 with the `block_table` pattern of
step D.5. Production prefill hits this shape when serving multiple
prompts simultaneously.

### D.8. attention, paged, ragged, decode — **[planned]**

The vLLM-style production serving kernel. **Why**: the destination —
ragged batch of decoding sequences, each with its own block_table,
computed in parallel. Combines every lesson the spine has taught.

## Stage E — Mixture of Experts

### E.1. moe — **[planned]**

Top-k routing → `all_to_all` dispatch → grouped GEMM (one matmul per
expert, weights different) → `all_to_all` combine. **Why last**: the
final consumer of `all_to_all` from Stage B and the grouped variant of
matmul from Stage A. Production-relevant for sparse models like
Mixtral / DeepSeek-V3.

## Future considerations

Candidates for promotion onto the spine if the difficulty curve calls
for more bridging in practice. Each smooths a specific later step.

- **Top-k** — partial sort, score-aware reduction. Pre-Stage E
  prerequisite for MoE routing; also useful for beam search and
  attention pruning.
- **Histogram / scatter-add** — the dual to gather. Pre-Stage E for
  the MoE combine step; also exercises contention handling on TPU.
- **GQA / MQA** — attention variant slotted around step D.1. Production
  relevance: every modern serving stack uses GQA. Teaches K/V head
  grouping and broadcast against the same tiled-attention skeleton.
- **1D / 2D convolution** — outside-ML breadth. Stencil / halo pattern;
  tile-boundary reasoning that sliding-window attention and sparse
  kernels both reuse.
- **Sort (bitonic or radix)** — outside-ML breadth. Parallel sort
  patterns; gives top-k for free at some cost.

## Notes

- **Skip Pallas for memory-bound unary elementwise after `scale`.** XLA
  fuses `silu`, `dropout`, and friends perfectly; there's no learning in
  those Pallas kernels.
- **Bench the baseline before its follow-up.** A Pallas variant whose
  win comes from compute-comm overlap (Stage C's `pallas.py` vs the
  sequentially-composed `xla.py`) or from a pipelined / double-buffered
  refinement only earns its place as the *delta* against an
  un-overlapped or un-pipelined version that already has a
  `bench_history/` entry. Without the baseline, the comparison is
  unanchored.
- **Read the HLO at every JAX-side step.** Without `--dump-hlo` you
  can't tell whether your reformulation helped or hurt. Most of the
  spine is Pallas-only; the JAX-side decisions concentrate in Stage C's
  `xla.py` (verifying the layer compiles to matmul + collective with
  comm on the critical path) and at step D.1 (the "softmax can't fuse
  into the matmul" wall is read in the HLO).
- **`compare()` is for variants of one op, not across ops.** Stage C's
  `mlp_tp_fwd` and `mlp_fsdp_fwd` (and the eight Stage D attention
  variants) are different ops with different inputs, sharding, and
  byte counts — they live in separate `ops/<name>/` directories and
  aren't benched against each other.
- **Update this file when a kernel lands.** Status flips from `[planned]`
  to `[done]`; add a link to the new `PERF.md`. The rationale lines stay
  put — they're what motivated the choice, and that doesn't change.
