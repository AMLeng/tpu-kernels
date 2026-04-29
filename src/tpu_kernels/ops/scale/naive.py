"""Naive scale: the obviously-correct reference. Never tuned for perf."""

from __future__ import annotations

import jax


def scale(x: jax.Array) -> jax.Array:
    return x * 2
