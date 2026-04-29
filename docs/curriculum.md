# Curriculum

The planned progression of kernels for learning to write performant TPU v5e
work in both plain JAX and Pallas. The order is **hybrid**: JAX-first for
ops where XLA can already approach speed-of-light, Pallas only when XLA
hits a wall. Each step is chosen for what it teaches — either a JAX-level
lever you can pull, or a Pallas concept you'll need later.

This file is the source of truth for *what's planned next* and *why*.
Per-attempt history lives in `git log` and `bench_history/<op>/*.json`;
this file only tracks kernel-level status and the original rationale.

## How to use this

Pick the first kernel not marked **[done]** and follow CLAUDE.md's
per-op workflow:

1. Read its annotation here for the *why*.
2. Create the op directory under `src/tpu_kernels/ops/<name>/`.
3. Build the variants listed in the kernel's entry. Hit the regime target
   from CLAUDE.md (or a tighter one if `PERF.md` calls for it).
4. Update the entry below to **[done]** with a pointer to its `PERF.md`.

## Status

The transition point — where pure-JAX stops being enough and Pallas earns
its complexity — is at step 4 → 5 (`mha_eager` → `flash_attention_fwd`).

### 0. scale — **[done]**

Memory-bound elementwise (×2). Pure hello-world: validates the harness
end-to-end and proves `pallas_call`, `BlockSpec`, the runner, and the
roofline math agree on a trivial kernel. No JAX or Pallas lesson beyond
"the toolchain works."

See [`ops/scale/PERF.md`](../src/tpu_kernels/ops/scale/PERF.md).

### 1. rmsnorm — **[planned]**

`xla.py` only. Add `pallas.py` only if XLA misses the regime target.

**Why**: practices accumulator dtype (f32 inside, bf16 out), `lax.rsqrt`,
and fusion behavior. Memory-bound; this is the lightest-weight kernel
where dtype/precision choices visibly move the BW% number. Read the
HLO afterward to confirm the chain fused into one elementwise loop.

### 2. softmax — **[planned]**

`xla.py` only. Add `pallas.py` only if XLA misses the regime target.

**Why**: practices fusion across the stable-form chain
(max → sub → exp → sum → div, ideally as one kernel). Memory-bound; XLA
usually fuses this, but `--dump-hlo` is the actual exercise — verify it
fused, identify the breaks if it didn't.

### 3. matmul (tiled bf16) — **[planned]**

**Both** `xla.py` and `pallas.py`. The only step where both variants are
mandatory.

**Why** (`xla.py`): exercises `dot_general(precision=...)`, bf16 vs
bf16-with-f32-accumulator tradeoff, einsum reorder. The cheapest place to
feel how precision flags affect both correctness and speed.

**Why** (`pallas.py`): first MXU exposure via `lax.dot_general` inside
a Pallas kernel. Teaches accumulator handling and 128×128 tile shapes.
Worth writing even when `xla` wins — the MXU practice is a prerequisite
for flash attention.

A follow-up `pallas_pipelined.py` adds `pltpu.emit_pipeline` plus
double-buffering once the un-pipelined Pallas version is benched. The
delta between them is the lesson on async DMA.

### 4. mha_eager (full attention, no flash) — **[planned]**

`xla.py` only. **Do not write a Pallas variant.**

**Why**: this is the wall. XLA materializes the O(N²) attention matrix and
can't fuse softmax into the matmul. The exercise is reading `--dump-hlo`,
watching BW% collapse on long sequences, and confirming there's no
JAX-level move that fixes it. Step 5 is the response — not a Pallas
variant of this kernel.

### 5. flash_attention_fwd — **[planned]**

`pallas.py` only.

**Why**: composes the patterns from steps 1–3 (in-tile reductions, MXU,
optional pipelining). The first kernel where Pallas's complexity earns
its keep. Mixed regime; expect a longer iteration loop.
`flash_attention_bwd` follows once forward hits target.

### 6. all_reduce (ring) — **[planned]**

`pallas.py` only.

**Why**: distributed primer with trivial math, so semaphores,
`pltpu.make_async_remote_copy`, and `pltpu.CompilerParams(collective_id=...)`
are the only moving pieces. Ordered after `flash_attention_fwd` so
distribution arrives as a separable concern, not bundled into a complex
kernel's debugging session.

## Beyond the curriculum

In rough order, with the same per-op contract:

- `flash_attention_bwd` — gradient correctness vs reference.
- `paged_attention` — fixed-size pages.
- `ragged_paged_attention` — variable seqlen with page-table indirection.
- `moe` — top-k routing, sorted dispatch, grouped GEMM, combine.

These are the destination. Once they're built, the repo has covered the
non-trivial production transformer kernels for TPU v5e and the curriculum
goal — *learn to write performant kernels in both JAX and Pallas* — is
fulfilled.

## Notes

- **Skip Pallas for memory-bound unary elementwise after `scale`.** XLA
  fuses `silu`, `dropout`, and friends perfectly; there's no learning in
  those Pallas kernels.
- **Don't add follow-up variants until the baseline is benched.** E.g.
  don't start `pallas_pipelined.py` for matmul until the un-pipelined
  Pallas version has a `bench_history/` entry. The delta between them
  *is* the lesson.
- **Read the HLO at every JAX-side step.** Without `--dump-hlo` you can't
  tell whether your reformulation helped or hurt — you'll be guessing.
- **Update this file when a kernel lands.** Status flips from `[planned]`
  to `[done]`; add a link to the new `PERF.md`. The rationale lines stay
  put — they're what motivated the choice, and that doesn't change.
