# Matmul (tiled bf16)

Target: ≥80% MFU @ (M=8192, N=8192, K=8192), bf16
Current: naive 94.9% / xla 94.9% / pallas 94.6% MFU on v5e (compute-bound). Target cleared by all three.
Bottleneck: at the MXU plateau — the remaining ~5% is fixed-cost overhead (DMA setup, kernel launch). pallas matches xla within noise but does not beat it; the (1024, 1024, 1024) cube would in theory go further but OOMs VMEM (18.4 MiB > 16 MiB scoped limit), so K=512 is the largest tile that fits.
Next: kernel is done at this shape. Per CLAUDE.md, pallas only justifies itself by beating xla — keep it as a teaching artifact for the tiled-MXU pattern flash attention reuses, but don't tune further. Re-evaluate if a non-square or skinny shape pushes xla off the plateau.
