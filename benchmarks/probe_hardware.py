"""Empirically verify v5e roofline constants against the actual chip.

Two kinds of check, both reported in one run:

1. *API cross-check*: compare ``benchmarks/roofline.py`` constants against
   ``pltpu.get_tpu_info()``. Catches drift with JAX's own numbers without
   touching the chip — same source that powers the TPU-marked unit test
   in ``tests/test_roofline.py``.
2. *Empirical probe*: allocate Pallas scratch buffers in VMEM and SMEM,
   binary-search the largest size the XLA:TPU compiler accepts.
   Independent of any source-of-truth — confirms what the chip itself
   will let you allocate. Run when you don't trust the API or when
   bringing up a new TPU generation.

Run::

    uv run python -m benchmarks.probe_hardware           # full report
    uv run python -m benchmarks.probe_hardware --no-probe  # API check only
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from benchmarks import roofline


def _compare_api_to_roofline() -> list[tuple[str, object, object, bool]]:
    """Return ``(name, roofline_value, api_value, ok)`` rows for every
    constant the runtime exposes that has a roofline.py counterpart.
    ``ok`` uses exact equality for byte counts and ``rel<=1e-3`` for
    advertised peaks (rounding diffs between Google's numbers and the
    runtime's are not real drift)."""
    info = pltpu.get_tpu_info()

    def _approx(a: float, b: float, rel: float = 1e-3) -> bool:
        return abs(a - b) / max(abs(a), abs(b), 1.0) <= rel

    return [
        # Exact equality — these are integer byte counts.
        (
            "V5E_VMEM_CAPACITY",
            roofline.V5E_VMEM_CAPACITY,
            info.vmem_capacity_bytes,
            info.vmem_capacity_bytes == roofline.V5E_VMEM_CAPACITY,
        ),
        (
            "V5E_SMEM_CAPACITY",
            roofline.V5E_SMEM_CAPACITY,
            info.smem_capacity_bytes,
            info.smem_capacity_bytes == roofline.V5E_SMEM_CAPACITY,
        ),
        (
            "V5E_HBM_CAPACITY",
            roofline.V5E_HBM_CAPACITY,
            info.hbm_capacity_bytes,
            info.hbm_capacity_bytes == roofline.V5E_HBM_CAPACITY,
        ),
        # Tolerant — peaks are advertised, runtime API rounds slightly.
        (
            "V5E_HBM_BANDWIDTH",
            roofline.V5E_HBM_BANDWIDTH,
            info.mem_bw_bytes_per_second,
            _approx(roofline.V5E_HBM_BANDWIDTH, info.mem_bw_bytes_per_second),
        ),
        (
            "V5E_BF16_PEAK_FLOPS",
            roofline.V5E_BF16_PEAK_FLOPS,
            info.bf16_ops_per_second,
            _approx(roofline.V5E_BF16_PEAK_FLOPS, info.bf16_ops_per_second),
        ),
        (
            "V5E_INT8_PEAK_OPS",
            roofline.V5E_INT8_PEAK_OPS,
            info.int8_ops_per_second,
            _approx(roofline.V5E_INT8_PEAK_OPS, info.int8_ops_per_second),
        ),
    ]


def _try_alloc_vmem(scratch_bytes: int) -> bool:
    """Compile+run a Pallas kernel with a VMEM scratch of ``scratch_bytes``.
    Return True on success, False if the compiler refuses."""
    # bf16 = 2 bytes; tile is (8, 128) so shape rows to a multiple of 8.
    cols = 128
    rows = max(8, (scratch_bytes // 2 // cols) // 8 * 8)
    if rows * cols * 2 < scratch_bytes:
        return False  # rounding lost too much; caller will retry larger

    def kernel(x_ref, y_ref, scratch_ref):
        scratch_ref[...] = jnp.zeros_like(scratch_ref)
        y_ref[...] = x_ref[...]

    @jax.jit
    def run(x):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            scratch_shapes=[pltpu.VMEM((rows, cols), jnp.bfloat16)],
        )(x)

    try:
        x = jnp.zeros((8, 128), dtype=jnp.bfloat16)
        run(x).block_until_ready()
        return True
    except Exception:
        return False


def _try_alloc_smem(scratch_bytes: int) -> bool:
    """SMEM analogue — int32 ref of ``scratch_bytes // 4`` elements."""
    elems = max(1, scratch_bytes // 4)

    def kernel(x_ref, y_ref, smem_ref):
        smem_ref[0] = jnp.int32(1)
        y_ref[...] = x_ref[...]

    @jax.jit
    def run(x):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            scratch_shapes=[pltpu.SMEM((elems,), jnp.int32)],
        )(x)

    try:
        x = jnp.zeros((8, 128), dtype=jnp.bfloat16)
        run(x).block_until_ready()
        return True
    except Exception:
        return False


def _bisect(try_alloc, hi_hint: int, granule: int) -> int:
    """Binary-search the largest scratch size ``try_alloc`` accepts. Start
    from ``hi_hint``, expand if it fits; bisect downward if it doesn't.
    Resolve to the nearest ``granule`` bytes — finer ``granule`` means more
    compilations (each ~1s), so caller picks the precision they need."""
    if try_alloc(hi_hint):
        lo, hi = hi_hint, hi_hint * 2
        while try_alloc(hi):
            lo, hi = hi, hi * 2
    else:
        hi = hi_hint
        lo = hi_hint // 2
        while not try_alloc(lo):
            lo //= 2
            if lo < granule:
                return 0
    while hi - lo > granule:
        mid = (lo + hi) // 2
        if try_alloc(mid):
            lo = mid
        else:
            hi = mid
    return lo


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the empirical Pallas-allocation probes; only do the API cross-check.",
    )
    args = parser.parse_args()

    info = pltpu.get_tpu_info()
    chip = info.chip_version.value  # pyright: ignore[reportAttributeAccessIssue]
    print(f"Detected: {chip} ({info.num_cores} core(s))")
    print()

    # 1. API cross-check.
    print("API vs roofline.py:")
    rows = _compare_api_to_roofline()
    width = max(len(r[0]) for r in rows)
    for name, rl_val, api_val, ok in rows:
        marker = "✓" if ok else "✗"
        print(f"  {marker} {name:<{width}}  roofline={rl_val:<16}  runtime={api_val}")
    drift = [r for r in rows if not r[3]]
    if drift:
        print(f"\n  {len(drift)} mismatch(es). Update benchmarks/roofline.py.")
    print()

    if args.no_probe:
        return

    # 2. Empirical probes — kernel allocations against the live compiler.
    # Granule scales with memory size: 1 MiB for VMEM, 4 KiB for SMEM.
    # The compiler's actual resolution is finer; granule trades probe time
    # (each compilation is ~1s) for displayed precision.
    print("Empirical probe (Pallas scratch allocation, binary search):")
    vmem = _bisect(_try_alloc_vmem, info.vmem_capacity_bytes, granule=1024 * 1024)
    print(
        f"  VMEM: {vmem / 2**20:>7.2f} MiB max scratch  "
        f"(API reports {info.vmem_capacity_bytes / 2**20:.0f} MiB)"
    )
    smem = _bisect(_try_alloc_smem, info.smem_capacity_bytes, granule=4 * 1024)
    print(
        f"  SMEM: {smem / 2**10:>7.2f} KiB max scratch  "
        f"(API reports {info.smem_capacity_bytes / 2**10:.0f} KiB)"
    )
    print()
    print("Note: probe ceiling is below API capacity by however much internal scratch")
    print("each kernel reserves (a few hundred bytes for these probes).")


if __name__ == "__main__":
    main()
