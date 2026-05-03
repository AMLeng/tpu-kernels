# Embedding lookup

Target: ≥80% HBM BW @ (vocab=32K, hidden=8192, M=8192), bf16
Current: not measured yet (xla baseline pending bench; Pallas not written)
Bottleneck: TBD — measure xla baseline first; expectation per the matmul precedent is that XLA's `jnp.take` saturates HBM since a pure gather has nothing to fuse, and the Pallas variant will match rather than beat it.
Next: bench xla, then write Pallas variant exercising data-dependent indexing (`pl.dynamic_slice` against runtime offsets, or a runtime-driven `BlockSpec` `index_map`) — the pattern that returns at the paged-attention steps as `block_table` lookups.
