# Segment cumsum

Target: ≥80% HBM BW @ (M=2^28, mean_length=1024), bf16
Current: 22.9% HBM BW @ bm=524288 (M=2^28, mean_length=1024)
Bottleneck: VPU/DMA serialization. Sweep monotonic in bm (13.9% → 22.9% over 32K → 512K) consistent with per-tile DMA setup amortizing, but the asymptote sits well below the v5e HBM ceiling. Realized ~187 GB/s on a binds=memory op means the VPU work per tile (7 lane shifts + cross-row seg-HS + broadcast-add) is running in series with the HBM read rather than overlapped. bm=1048576 OOMs scoped VMEM at this shape (16.04 MiB vs 16 MiB limit), capping reachable bm at 524288.
Next: try emit_pipeline (or double-buffered grid) to overlap HBM read of tile i+1 with VPU work on tile i; if that doesn't recover the 4x gap, profile to find whether the cross-row scan or the row-carry broadcast-add is the inner-tile limiter.
