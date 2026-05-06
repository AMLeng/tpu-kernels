"""Naive embedding lookup on the packed 4-D shape. Never tuned."""

from __future__ import annotations

import jax


def embedding_lookup(params: jax.Array, ids: jax.Array) -> jax.Array:
    return params[ids]
