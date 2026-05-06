# Embedding lookup (packed)

Target: ≥65% HBM BW @ (vocab=128256, hidden=4096, M=8192), bf16 packed
Current: xla 65.4% / pallas 68.8% HBM BW on v5e (memory-bound), 2026-05-06. Both clear the ≥65% target; pallas edges out xla as predicted.
Bottleneck: speed-of-light HBM. All variants transfer the theoretical-
minimum 134 MiB; the remaining gap to 100% is HBM utilization, not
algorithm. The Pallas kernel uses an explicit read-many / write-one
DMA pattern: ``bm`` per-row reads HBM→VMEM (random ids, can't bulk),
then **one** bulk write VMEM→HBM per grid step, amortizing the
per-DMA setup cost that dominates if the writes are all per-row.
Next: nothing pressing.
