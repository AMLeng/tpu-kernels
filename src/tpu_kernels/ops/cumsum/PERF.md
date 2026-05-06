# Cumsum

Target: ≥80% HBM BW @ (M=2^28), bf16
Current: not measured yet
Bottleneck: `jnp.cumsum` binds `cumsum_p`, whose default lowering is a triangular `reduce_window` ("O(N^2) reduce-window" per JAX source); XLA's rewriter doesn't recover an efficient parallel-prefix on v5e at this shape, so the kernel is sequential-dependency-bound rather than HBM-bound. The other JAX-level path — `jax.lax.associative_scan(jnp.add, x)` — is also unworkable at this scale: it's a Python-side recursive expansion into O(N log N) HLO, and compile time grows superlinearly (16 s at M=2^20, ~25 min projected at M=2^24). Neither pure-JAX form scales.
Next: a Pallas carry-across-tiles variant is the only path to HBM-bound prefix-sum at M=2^28. Off-scope for this stepping-stone op — the lesson is captured in segment_cumsum (A.6); cumsum's job here is to make the "neither pure-JAX path scales" finding explicit so it doesn't have to be re-derived.
