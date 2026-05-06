# Embedding lookup

Target: match XLA at ~32% HBM BW @ (vocab=128256, hidden=4096, M=8192), bf16
Current: not measured yet (kernel just landed; bench in a follow-up commit)
Bottleneck: HBM tiling + Mosaic DMA emission. XLA stores
TensorCore-bound bf16 tensors in `T(8, 128)(2, 1)` layout: 2 KiB HBM
tiles holding 8 rows interleaved per tile. Mosaic only emits
tile-aligned DMAs (sub-tile reads would need scatter-gather across
the tile's striped bytes — possible at the hardware level, but not
something the lowering pass generates), so a single-row read against
this layout isn't expressible from Pallas. Pallas fetches 8-row
slabs and uses 1 row → hard cap at 12.5% BW. XLA's
`gather_custom_fusion` (kind=kCustom, backend-internal C++) gets to
~32% by mechanisms we can't reach from JAX surface. So Pallas-on-
natural-layout *loses* to XLA, structurally.
Next: nothing actionable on this layout. The 12.5% cap is structural;
the path past it is a different storage layout for `params`, not a
better kernel against the natural 2-D form.
