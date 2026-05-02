"""Pallas RMSNorm: tiled kernel with f32 reduction.

The reduction is accumulated in f32 regardless of input dtype to match the
naive oracle (cf. ``naive.py``) — the squared-sum / rsqrt / normalize chain
runs in f32 and only downcasts to ``x.dtype`` immediately before the
per-feature ``scale`` multiply, mirroring the canonical Llama pattern.

``eps`` is bound via ``functools.partial`` so it becomes a compile-time
literal in the jaxpr — no input slot, no Ref. Different ``eps`` values
therefore produce distinct traces.

``block_shape`` is a 1-tuple ``(bm,)`` rather than a bare ``int``: every
Pallas kernel in this repo declares its tile geometry under the same
parameter name, so the bench harness (``compare()`` / ``sweep()``) can
bake the value in via ``functools.partial`` without per-kernel policy.

This module exposes a plain function — bench/test sites jit it themselves
(the bench captures ``block_shape`` in a closure, the correctness tests
use ``interpret=True`` and don't need jit).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

DEFAULT_BLOCK = (128,)


def _rms_kernel(x_ref, scale_ref, o_ref, *, eps: float) -> None:
    x_32 = x_ref[...].astype(jnp.float32)
    ms = jnp.mean(x_32 * x_32, axis=-1, keepdims=True)
    o_ref[...] = (x_32 * jax.lax.rsqrt(ms + eps)).astype(o_ref.dtype) * scale_ref[...]


def rmsnorm(
    x: jax.Array,
    scale: jax.Array,
    eps: float = 1e-6,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
    interpret: bool = False,
) -> jax.Array:
    if x.ndim != 2:
        raise ValueError(f"rmsnorm expects 2-D input, got {x.shape}")
    if scale.ndim != 1:
        raise ValueError(f"rmsnorm expects 1-D scale, got {scale.shape}")
    if len(block_shape) != 1:
        raise ValueError(f"rmsnorm expects a 1-axis block_shape, got {block_shape}")
    (bm,) = block_shape
    m, n = x.shape
    tile_shape = (bm, n)
    if m % bm:
        raise ValueError(f"shape {x.shape} not divisible by block {tile_shape}")
    if scale.shape[0] != n:
        raise ValueError(f"scale shape {scale.shape} does not match feature dim {n}")

    return pl.pallas_call(
        partial(_rms_kernel, eps=eps),
        grid=(m // bm,),
        in_specs=[
            pl.BlockSpec(tile_shape, lambda i: (i, 0)),
            pl.BlockSpec(scale.shape, lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec(tile_shape, lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x, scale)
