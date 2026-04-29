# Kernel recipes

Living patterns doc. Add an entry when a kernel teaches you something
reusable. Keep entries short — a name, when to use it, the code shape.

## Tiled elementwise (memory-bound primer)

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

## Row reduction (softmax/layernorm primer)

_TODO: add when softmax lands._

## Tiled matmul (compute-bound primer)

_TODO: add when matmul lands._

## Async DMA + double-buffer

_TODO: add when first kernel needs hand-pipelining._

## Distributed ring (collective primer)

_TODO: add when Pallas all-reduce lands._
