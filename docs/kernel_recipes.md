# Kernel patterns

Reusable Pallas patterns picked up while building kernels in this repo.
Add an entry when a kernel teaches you something the next kernel can
copy verbatim — not a list of patterns we *want*, but a record of patterns
we've actually used. For what's planned next, see
[`curriculum.md`](curriculum.md).

Keep entries short: name, when to use it, code shape.

## Tiled elementwise (memory-bound)

When the op is one read + one write per element. Speed-of-light is HBM BW.

```python
def kernel(x_ref, o_ref):
    o_ref[...] = f(x_ref[...])

pl.pallas_call(
    kernel,
    grid=(M // bm, N // bn),
    in_specs=[pl.BlockSpec((bm, bn), lambda i, j: (i, j))],
    out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
)(x)
```

Block sizes worth sweeping: `(128, 128)`, `(256, 256)`, `(512, 512)`,
`(1024, 1024)`. Bigger blocks reduce DMA overhead but use more VMEM.
First built in `ops/scale/pallas.py`.
