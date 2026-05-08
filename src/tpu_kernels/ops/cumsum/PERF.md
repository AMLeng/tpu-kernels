# Cumsum

Target: ≥80% HBM BW @ (M=2^28), bf16
Current: 60.2% HBM BW (Pallas, block=524288); xla 19.0% on the same shape.
Bottleneck: relayouts (the sublane↔lane transpose and the (b, 128, 128) reshapes around the inner dot) and tile-to-tile serialization through the (1,) scratch carry.
Next: profile one tile to attribute the gap, then fold the relayout into the dot's contracting/output dims and/or break the carry into a two-pass scan.
