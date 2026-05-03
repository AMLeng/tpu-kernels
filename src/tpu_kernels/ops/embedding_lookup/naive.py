"""Naive embedding lookup: the obviously-correct reference. Never tuned for perf."""

from __future__ import annotations

import jax


def embedding_lookup(params: jax.Array, ids: jax.Array) -> jax.Array:
    return params[ids]
