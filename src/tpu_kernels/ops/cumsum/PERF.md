# Cumsum

Target: ≥80% HBM BW @ (M=2^28), bf16
Current: 72.5% HBM BW (Pallas, block=524288); xla 19.0% on the same shape.
Bottleneck: tile-to-tile serialization through the (1,) scratch carry — 512 tiles at bm=524288 chain serially through it. Residual (b, 128, 128) reshapes around the inner dot remain but are likely secondary now that the sublane↔lane transpose is gone.
Next: break the carry into a two-pass scan (per-tile totals → exclusive prefix → fused add) so tiles can run in parallel.
