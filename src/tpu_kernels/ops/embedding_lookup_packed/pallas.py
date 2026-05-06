"""Pallas embedding lookup on the packed 4-D layout.

``params`` arrives as ``bf16[vocab, hidden//1024, 8, 128]`` with natural
Mosaic ``T(8, 128)(2, 1)`` on the inner two dims. That layout is byte-
equivalent to the 1-D ``T(1024)(128)(2, 1)`` form: the bf16 packing
pair lands on adjacent in-row positions instead of adjacent rows, so
each row's bytes are contiguous in memory and a single async DMA can
fetch one row in one shot.

The kernel uses a **read-many / write-one** structure per grid step:

1. Issue ``bm`` async per-row reads HBM → VMEM scratch (random ``ids``,
   so the reads are unavoidably one-per-id; 4-semaphore fan-out
   pipelines them).
2. After all reads are in VMEM, issue **one bulk async DMA** of
   ``bm * hidden`` bytes VMEM → HBM, writing the consecutive output
   rows for this grid step in one shot.

Why bulk-write: a naive HBM→HBM per-row pattern (``bm`` small writes)
spends most of its time on per-DMA setup overhead at HBM. Staging in
VMEM and emitting one bulk write per grid step amortizes that
overhead. Confirmed empirically — the per-row-write variant lands at
~56% BW; the bulk-write variant lands at ~67%.

Implementation notes:
- Pallas sees the params as 1-D (``params.reshape(-1)``, a free
  metadata bitcast since the 4-D and 1-D forms are byte-equivalent).
- Per-row read DMA src/dst use ``at[pl.ds(byte_offset, hidden)]`` on
  1-D refs; addressing in 1-D lowers more cleanly than ``at[id]`` on a
  4-D ref.
- The VMEM scratch is also 1-D (``(bm * hidden,)``) so per-row writes
  at offset ``k * hidden`` are lane-aligned (``hidden % 128 == 0``)
  rather than sublane-aligned (would need ``bm`` divisible by 8).
"""

from __future__ import annotations

from functools import partial

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_BLOCK = (128,)
NUM_SEMS = 4


def _emb_kernel(
    ids_ref,
    params_flat_ref,
    o_flat_ref,
    scratch,
    *sems,
    bm: int,
    hidden: int,
) -> None:
    i = pl.program_id(0)

    # Phase 1 — per-row reads HBM -> VMEM scratch, fan-out across
    # NUM_SEMS semaphores so the small DMAs pipeline.
    read_copies = []
    for k in range(bm):
        row_id = ids_ref[i * bm + k]
        copy = pltpu.make_async_copy(
            src_ref=params_flat_ref.at[pl.ds(row_id * hidden, hidden)],
            dst_ref=scratch.at[pl.ds(k * hidden, hidden)],
            sem=sems[k % NUM_SEMS],
        )
        copy.start()
        read_copies.append(copy)
    for copy in read_copies:
        copy.wait()

    # Phase 2 — single bulk write VMEM -> HBM. One DMA setup cost,
    # one large contiguous transfer; amortizes the per-DMA overhead
    # that dominates if we wrote each row separately.
    bulk_copy = pltpu.make_async_copy(
        src_ref=scratch,
        dst_ref=o_flat_ref.at[pl.ds(i * bm * hidden, bm * hidden)],
        sem=sems[0],
    )
    bulk_copy.start()
    bulk_copy.wait()


def embedding_lookup(
    params: jax.Array,
    ids: jax.Array,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
    interpret: bool = False,
) -> jax.Array:
    """Pallas embedding lookup against ``params`` in packed 4-D layout.

    Per-call shape contract: ``params: bf16[vocab, hidden//1024, 8, 128]``
    (caller-pre-formatted into the natural-Mosaic packed layout),
    ``ids: int32[M]``, returns ``bf16[M, hidden//1024, 8, 128]``.
    """
    if params.ndim != 4:
        raise ValueError(
            f"expected 4-D packed params (vocab, hidden//1024, 8, 128), "
            f"got shape {params.shape}; see module docstring."
        )
    if ids.ndim != 1:
        raise ValueError(f"expected 1-D ids, got shape {ids.shape}")
    if len(block_shape) != 1:
        raise ValueError(f"expected 1-axis block_shape, got {block_shape}")

    _vocab, h_chunks, sub, lane = params.shape
    if (sub, lane) != (8, 128):
        raise ValueError(f"packed params must have inner shape (8, 128); got ({sub}, {lane})")
    hidden = h_chunks * sub * lane
    (bm,) = block_shape
    m = ids.shape[0]
    if m % bm:
        raise ValueError(f"ids shape {ids.shape} not divisible by bm={bm}")

    # Flatten params to 1-D for the kernel — the 4-D packed layout is
    # byte-equivalent to 1-D `T(1024)(128)(2,1)`, so this is a free
    # bitcast. Inside the kernel `at[pl.ds(row_id*hidden, hidden)]` on a
    # 1-D ref lowers more cleanly than `at[row_id]` on a 4-D ref.
    params_flat = params.reshape(-1)
    out_flat = pl.pallas_call(
        partial(_emb_kernel, bm=bm, hidden=hidden),
        out_shape=jax.ShapeDtypeStruct((m * hidden,), params.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(m // bm,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((bm * hidden,), params.dtype),
                *([pltpu.SemaphoreType.DMA] * NUM_SEMS),
            ],
        ),
        interpret=interpret,
    )(ids, params_flat)
    # Reshape back to 4-D — same bitcast, no byte movement.
    return out_flat.reshape(m, h_chunks, sub, lane)
