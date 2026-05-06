"""Pallas embedding lookup on the natural 2-D bf16 layout.

XLA stores TensorCore-bound bf16 tensors in ``T(8, 128)(2, 1)`` layout:
HBM holds the array as 2 KiB tiles, each tile carrying 8 rows
interleaved within its bytes. Mosaic only emits tile-aligned DMAs
against tiled refs (sub-tile reads would need scatter-gather across
the tile's striped bytes — possible at the hardware level, but not
something the lowering pass generates), so a single-row read against
the natural layout isn't expressible from Pallas. That bites on both
ends of the gather: HBM reads must be 8-row slabs, and HBM writes
must be 8-row slabs.

So the kernel processes ids in 8-id chunks. For each chunk:

1. Read 8 slabs (each ``(8, hidden)``) from HBM into VMEM — one slab
   per id, async, semaphore fan-out.
2. Build the chunk's output ``(8, hidden)`` block in VMEM by, for each
   of the 8 ids, multiplying its slab by a ``(8, 8)`` one-hot
   permutation matrix that picks ``sub_id = id % 8`` and places it at
   row ``sub`` (the id's index within the chunk). ``sum`` across the 8
   per-sub contributions assembles the full block.
3. DMA the ``(8, hidden)`` block to HBM as one tile-aligned write.

Why not single-row mask-and-sum: ``dynamic_slice`` is not lowered by
Pallas TC, and the natural ``mask[:, None]`` form needs a ``(8,) -> (8,
1)`` shape cast that Pallas TC also doesn't lower. Using a ``(8, 8)``
permutation matrix sidesteps both — the matrix is built directly in 2-D
shape via two ``broadcasted_iota`` calls, and the matmul against a
``(8, hidden)`` slab lowers to plain mul + reduce.

Hard cap at ``1 / PAGE_ROWS = 12.5%`` BW because every read-DMA fetches
8 rows and only 1 is useful per id. The XLA baseline
(``jit(params[ids])``) hits ~32% via ``gather_custom_fusion``, so on
this layout Pallas loses to XLA. Beating XLA here would require a
storage layout that exposes single-row alignment, which is outside
the scope of this op.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_BLOCK = (128,)
NUM_SEMS = 4
PAGE_ROWS = 8  # natural bf16 outer sublane tile size; the slab unit


def _emb_kernel(
    ids_ref,
    params_ref,
    o_ref,
    scratch_slab,
    scratch_out,
    *sems,
    bm: int,
    hidden: int,
) -> None:
    i = pl.program_id(0)
    n_chunks = bm // PAGE_ROWS

    # Phase 1 — issue bm async slab reads HBM -> VMEM scratch_slab.
    read_copies = []
    for k in range(bm):
        row_id = ids_ref[i * bm + k]
        tile_idx = row_id // PAGE_ROWS
        copy = pltpu.make_async_copy(
            src_ref=params_ref.at[pl.ds(tile_idx * PAGE_ROWS, PAGE_ROWS)],
            dst_ref=scratch_slab.at[k],
            sem=sems[k % NUM_SEMS],
        )
        copy.start()
        read_copies.append(copy)
    for copy in read_copies:
        copy.wait()

    # Phase 2 — build each chunk's (8, hidden) output block in VMEM.
    # For sub in 0..7, build a (8, 8) permute matrix one-hot at (sub, sub_id),
    # matmul with slab[chunk*8+sub] to produce a (8, hidden) where row `sub`
    # is the wanted row from that slab and other rows are zero. Sum across
    # subs to assemble the block.
    row_iota = lax.broadcasted_iota(jnp.int32, (PAGE_ROWS, PAGE_ROWS), 0)
    col_iota = lax.broadcasted_iota(jnp.int32, (PAGE_ROWS, PAGE_ROWS), 1)
    for c in range(n_chunks):
        # Pallas TC requires the matmul accumulator to be f32; downcast to
        # the slab dtype (bf16 in the real configuration) at the end before
        # the VMEM store. For bf16 matmul inputs this matches MXU semantics
        # (bf16 * bf16 -> f32 accumulate -> bf16 result).
        chunk_out = jnp.zeros((PAGE_ROWS, hidden), jnp.float32)
        for sub in range(PAGE_ROWS):
            slab_idx = c * PAGE_ROWS + sub
            sub_id = ids_ref[i * bm + slab_idx] % PAGE_ROWS
            permute = ((row_iota == sub) & (col_iota == sub_id)).astype(scratch_slab.dtype)
            chunk_out = chunk_out + jnp.matmul(
                permute, scratch_slab[slab_idx], preferred_element_type=jnp.float32
            )
        scratch_out[c] = chunk_out.astype(scratch_slab.dtype)

    # Phase 3 — write 8-row chunks to HBM. Sems can be reused: phase 1
    # waits drained them and the compute phase has no async DMAs.
    write_copies = []
    for c in range(n_chunks):
        copy = pltpu.make_async_copy(
            src_ref=scratch_out.at[c],
            dst_ref=o_ref.at[pl.ds(i * bm + c * PAGE_ROWS, PAGE_ROWS)],
            sem=sems[c % NUM_SEMS],
        )
        copy.start()
        write_copies.append(copy)
    for copy in write_copies:
        copy.wait()


def embedding_lookup(
    params: jax.Array,
    ids: jax.Array,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
    interpret: bool = False,
) -> jax.Array:
    """Pallas embedding lookup against natural 2-D ``T(8,128)(2,1)`` ``params``.

    Per-call shape contract: ``params: bf16[vocab, hidden]``, ``ids:
    int32[M]``, returns ``bf16[M, hidden]``. ``block_shape`` is the per-grid
    DMA fan-out (number of slab reads issued per program step); must be a
    multiple of ``PAGE_ROWS`` so the chunk-write structure works out.
    """
    if params.ndim != 2:
        raise ValueError(f"expected 2-D params, got shape {params.shape}")
    if ids.ndim != 1:
        raise ValueError(f"expected 1-D ids, got shape {ids.shape}")
    if len(block_shape) != 1:
        raise ValueError(f"expected 1-axis block_shape, got {block_shape}")

    vocab, hidden = params.shape
    (bm,) = block_shape
    m = ids.shape[0]

    if hidden % 128:
        raise ValueError(f"need hidden % 128 == 0, got {hidden}")
    if m % bm:
        raise ValueError(f"ids shape {ids.shape} not divisible by bm={bm}")
    if vocab % PAGE_ROWS:
        raise ValueError(f"vocab {vocab} not divisible by PAGE_ROWS={PAGE_ROWS}")
    if bm % PAGE_ROWS:
        raise ValueError(f"bm={bm} must be divisible by PAGE_ROWS={PAGE_ROWS}")

    return pl.pallas_call(
        partial(_emb_kernel, bm=bm, hidden=hidden),
        out_shape=jax.ShapeDtypeStruct((m, hidden), params.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(m // bm,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((bm, PAGE_ROWS, hidden), params.dtype),
                pltpu.VMEM((bm // PAGE_ROWS, PAGE_ROWS, hidden), params.dtype),
                *([pltpu.SemaphoreType.DMA] * NUM_SEMS),
            ],
        ),
        interpret=interpret,
    )(ids, params)
