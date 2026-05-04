# Matmul (tiled bf16)

Target: ≥80% MFU @ (M=16384, N=16384, K=16384), bf16
Current: naive 96.7% / xla 96.7% / pallas 95.5% MFU on v5e (compute-bound), 2026-05-04. Target cleared by all three.
Bottleneck: at the MXU plateau — the remaining ~3% is fixed-cost overhead (DMA setup, kernel launch). pallas matches xla within noise but does not beat it; the (1024, 1024, 1024) cube would in theory go further but OOMs the 16 MiB scoped VMEM limit (working set 18.4 MiB), so K=512 is the largest tile that fits. The scoped limit is a per-kernel compile-time scratch budget — independent of the total 128 MiB VMEM, so the recent VMEM-capacity correction doesn't relax it.
Next: kernel is done at this shape. Per CLAUDE.md, pallas only justifies itself by beating xla — keep it as a teaching artifact for the tiled-MXU pattern flash attention reuses, but don't tune further. Re-evaluate if a non-square or skinny shape pushes xla off the plateau.
