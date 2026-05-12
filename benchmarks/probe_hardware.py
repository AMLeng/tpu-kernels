"""Empirically verify v5e roofline constants against the actual chip.

Four kinds of check, all reported in one run:

1. *API cross-check*: compare ``benchmarks/roofline.py`` constants against
   ``pltpu.get_tpu_info()``. Catches drift with JAX's own numbers without
   touching the chip — same source that powers the TPU-marked unit test
   in ``tests/test_roofline.py``.
2. *Scratch probe*: allocate Pallas scratch buffers in VMEM and SMEM,
   binary-search the largest size the XLA:TPU compiler accepts.
   Independent of any source-of-truth — confirms what the chip itself
   will let you allocate. Run when you don't trust the API or when
   bringing up a new TPU generation.
3. *VPU compute probe*: parallel-ops throughput sweep. The body holds N
   independent (8, 128) accumulators — one VPU register each — and
   advances each by a single add (or mul) per iter. fori_loop carries
   force a true iter-to-iter data dep within each chain; across chains
   there is no data flow, so the scheduler can issue all N adds in the
   same cycle if the hardware has the ALU slots. Sweep N until throughput
   plateaus. The plateau in flops/sec is the **total sustained VPU
   bandwidth** — divide by the chip clock and per-op element count to
   compare against the W=4 ALU-slot ceiling from the scaling book
   (https://jax-ml.github.io/scaling-book/tpus/). The chain count at
   which the plateau kicks in is set by the per-iter overhead of
   fori_loop (a kernel can't get below the loop floor), not by ALU
   latency directly. There is no API counterpart for VPU peak (the
   runtime only exposes MXU peaks), so this is the only source for
   v5e VPU compute peaks.
4. *VPU bf16 carry cross-check*: same probe, same bf16 I/O, but the
   in-loop carry is varied between bf16 and f32. f32 carry hits the
   f32 plateau even with bf16 I/O, while bf16 carry stays at ~67% of
   it. Proves the bf16 deficit is per-iter bf16↔f32 conversion that
   the TPU backend inserts below Mosaic (the v5e VPU has no native
   bf16 ALU), not anything visible in the lowered IR.

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


# VPU compute probe: parallel-ops throughput sweep.
#
# Each chain holds a single (SUBLANE, LANE) = (8, 128) accumulator — exactly
# one VPU vector register, so each ``acc + c`` lowers to exactly one vec inst
# (no within-chain chunking, no implicit pipelining inside one "op"). Per body
# iter, each chain runs ``DEPTH`` serial ``acc = acc + c`` ops; Mosaic preserves
# source order so they form a true SSA dep chain and don't reassociate (verified
# by IR dump). The N accumulators are carried through fori_loop as a tuple, so
# iter k+1 of each chain has a true data dep on iter k of *that chain only*.
# Across chains there is no data flow.
#
# DEPTH must be large enough for the body work to dominate the per-iter
# fori_loop floor (counter inc, conditional branch, carry handover; ~9 cycles
# at 1.5 GHz). With DEPTH=1 the floor swallows ~80% of every iter (9 / (9+2))
# and the measured "plateau" is mostly loop overhead, not chip throughput.
# We fix DEPTH=64 — chain critical path D*L = 128 cycles vastly exceeds the
# floor, so amortization is essentially complete and the plateau lands
# within ~3% of the W=4 ALUs * 1.5 GHz = 6.14 TFLOPs theoretical peak.
#
# Modeling: per-iter time = max(N * D / W, D * L, T_floor) cycles. With D >>
# T_floor / L the floor drops out and saturation arrives where the ALU-bound
# term equals the chain critical path: N * D / W = D * L, i.e. N* = W * L —
# independent of D once D is big enough. With W=4, L=2: saturation at N=8.
#
# A constant ``c`` near 1 (one ULP below) keeps add and mul from folding to
# identity in bf16. ``mul`` uses ``1 - 2⁻⁷`` and ``add`` uses ``2⁻⁷``; both
# are exactly representable in bf16.

SUBLANE: int = 8
LANE: int = 128
DEPTH: int = 64  # serial ops per chain per body iter — see notes above


def _measure_throughput(
    io_dtype: jnp.dtype,
    op: str,  # "add" or "mul"
    n_chains: int,
    k: int,
    carry_dtype: jnp.dtype | None = None,
    depth: int = DEPTH,
) -> float:
    """Sustained ops/sec for ``N`` parallel chains, ``depth`` serial ops per chain, ``k`` iters.

    The body runs ``depth`` ops in series per chain (Mosaic preserves source
    order, no barrier needed). Across chains the ops are SSA-independent so
    the scheduler can issue them on as many ALU slots as the hardware exposes.
    Each chain's iter-to-iter recurrence is serial through fori_loop carry.

    ``depth`` defaults to the module-level ``DEPTH = 64`` — enough body work
    that loop overhead is negligible. Probes that need depth=1 specifically
    (e.g. to study loop-overhead behavior) can override.

    ``carry_dtype`` defaults to ``io_dtype`` (carry has the same dtype as I/O).
    Setting it different (e.g. ``io_dtype=bf16``, ``carry_dtype=f32``) inserts
    one cast at load and one at store but keeps the body in ``carry_dtype``.
    Used by the carry-dtype cross-check to show that bf16's plateau deficit
    is per-iter conversion overhead, not anything visible in the Mosaic IR.
    """
    from benchmarks.runner import bench

    if op not in ("add", "mul"):
        raise ValueError(f"op must be 'add' or 'mul', got {op!r}")
    if carry_dtype is None:
        carry_dtype = io_dtype

    def kernel(x_ref, y_ref):
        # Constants are 1 ULP off identity (0 for add, 1 for mul) and exactly
        # representable in bf16. Anything inexact (e.g. 0.999) rounds to the
        # identity in bf16 and Mosaic folds the op away — the probe still
        # counts a flop/iter, so the bf16 reading silently inflates.
        c_add = jnp.asarray(2.0**-7, dtype=carry_dtype)  # 0.0078125
        c_mul = jnp.asarray(1.0 - 2.0**-7, dtype=carry_dtype)  # 0.9921875 = 1 - 2⁻⁷
        c = c_add if op == "add" else c_mul

        chains_init = tuple(
            x_ref[i * SUBLANE : (i + 1) * SUBLANE].astype(carry_dtype) for i in range(n_chains)
        )

        def body(_i: jax.Array, accs: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
            new_accs = []
            for acc in accs:
                for _ in range(depth):
                    acc = acc + c if op == "add" else acc * c
                new_accs.append(acc)
            return tuple(new_accs)

        chains_final = jax.lax.fori_loop(0, k, body, chains_init)
        for i, chain in enumerate(chains_final):
            y_ref[i * SUBLANE : (i + 1) * SUBLANE] = chain.astype(io_dtype)

    @jax.jit
    def run(x):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        )(x)

    shape = (n_chains * SUBLANE, LANE)
    key = jax.random.key(0)
    base = jax.random.normal(key, shape, dtype=jnp.float32)
    x = base if io_dtype == jnp.float32 else base.astype(io_dtype)

    result = bench("vpu_probe", run, args=(x,), timing="device", warmup=3, iters=20)
    elements = n_chains * SUBLANE * LANE  # depth ops per element per iter
    flops = elements * k * depth
    return flops / result.median_s


def _saturation_n(sweep: dict[int, float], tol: float = 0.03) -> int:
    """Smallest N where throughput is within ``tol`` of that sweep's plateau."""
    peak = max(sweep.values())
    return min(n for n, tflops in sweep.items() if tflops >= (1 - tol) * peak)


def _run_vpu_probe(
    n_values: tuple[int, ...] = (1, 2, 4, 8, 12, 16, 24),
    k: int = 4096,
) -> dict[str, dict[str, dict[int, float]]]:
    """Parallel-ops throughput sweep for {f32, bf16} x {add, mul}; print + return.

    Each cell is sustained TFLOPs at that (dtype, op, N) for body depth = DEPTH
    serial ops per chain per iter. The plateau across N is the **total VPU
    bandwidth** for that op/dtype; saturation lands at N* = W * L (the
    intrinsic latency-bandwidth product, ~8 chains for v5e VPU once body
    work dominates the loop floor). add and mul should match if the ALU
    pool is uniform across op kind; f32 and bf16 deviate at saturation
    because the VPU has no native bf16 ALU (see carry-dtype cross-check
    below).
    """
    print(f"Empirical probe (VPU compute, parallel-ops throughput sweep, K={k}, DEPTH={DEPTH}):")
    print(f"  Each chain = one ({SUBLANE}, {LANE}) accumulator (1 vec reg).")
    print(f"  Body = {DEPTH} serial ops per chain per iter, repeated K times.")
    print()

    results: dict[str, dict[str, dict[int, float]]] = {}
    header = "    " + f"{'variant':<10}" + "".join(f"{f'N={n}':>8}" for n in n_values)
    for dtype_name, dtype in (("f32", jnp.float32), ("bf16", jnp.bfloat16)):
        print(f"  {dtype_name} TFLOPs:")
        print(header)
        per_variant: dict[str, dict[int, float]] = {}
        for op in ("add", "mul"):
            sweep = {n: _measure_throughput(dtype, op, n, k) / 1e12 for n in n_values}
            per_variant[op] = sweep
            peak = max(sweep.values())
            sat_n = _saturation_n(sweep)
            cells = "".join(
                f"{tflops:>7.2f}{'*' if tflops >= 0.97 * peak else ' '}"
                for tflops in sweep.values()
            )
            print(f"    {op:<10}{cells}  (peak {peak:.2f}, sat N={sat_n})")
        results[dtype_name] = per_variant
        print()
    print("  (* = within 3% of that op/dtype's plateau)")
    return results


def _run_vpu_carry_probe(
    n_values: tuple[int, ...] = (1, 8, 12, 16, 20, 24),
    k: int = 4096,
) -> dict[str, dict[int, float]]:
    """Carry-dtype cross-check: bf16 I/O with bf16 vs f32 carry.

    Holds I/O as bf16, varies only the loop-carry dtype. Both columns share
    the same kernel signature, the same op, and the same K — only the carry
    differs. f32 carry should reproduce the f32 plateau (~2.77 TFLOPs);
    bf16 carry should stay at the bf16 plateau (~1.86 TFLOPs). The split
    proves the bf16 deficit is per-iter conversion below Mosaic, not a
    hardware-level bf16 throughput limit.

    N range is narrower than the main sweep — we just need to span the
    saturation region so the divergence is visible.
    """
    print(f"Empirical probe (VPU bf16 carry-dtype cross-check, K={k}):")
    print("  Same bf16 I/O, vary in-loop carry dtype. f32-carry should match f32 plateau.")
    print()

    header = "    " + f"{'carry':<6}" + "".join(f"{f'N={n}':>8}" for n in n_values)
    print(header)
    results: dict[str, dict[int, float]] = {}
    for carry_name, carry_dtype in (("bf16", jnp.bfloat16), ("f32", jnp.float32)):
        sweep = {
            n: _measure_throughput(jnp.bfloat16, "add", n, k, carry_dtype=carry_dtype) / 1e12
            for n in n_values
        }
        results[carry_name] = sweep
        peak = max(sweep.values())
        cells = "".join(
            f"{tflops:>7.2f}{'*' if tflops >= 0.97 * peak else ' '}" for tflops in sweep.values()
        )
        print(f"    {carry_name:<6}{cells}  (peak {peak:.2f})")
    print()
    return results


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

    # 2. Scratch probes — kernel allocations against the live compiler.
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
    print("Note: probe ceiling is below API capacity by however much internal scratch")
    print("each kernel reserves (a few hundred bytes for these probes).")
    print()

    # 3. VPU compute — no API counterpart for VPU peak, so this is the only
    # source. Parallel-ops throughput sweep: N independent (8, 128) chains,
    # DEPTH serial ops per chain per iter, loop-carry forces serial within each
    # chain. With DEPTH big enough (we use 64) the body work dominates the
    # fori_loop floor and the plateau lands within ~3% of the chip's W=4 ALUs
    # * 1.5 GHz theoretical peak.
    results = _run_vpu_probe()
    elem_per_op = SUBLANE * LANE  # one vec inst spans this many elements
    # v5e VPU clock = 1.5 GHz, citing https://jax-ml.github.io/scaling-book/tpus/.
    # Cross-checks against the API's MXU peak: 197 TFLOPs / (256*256*2 flops/cycle)
    # = 1.50 GHz exactly, confirming the scaling-book figure (and that the MXU and
    # VPU share a clock domain on v5e).
    vpu_clock_hz = 1.5e9
    theoretical_peak_tflops = 4 * elem_per_op * vpu_clock_hz / 1e12  # W=4 ALUs from book
    for dt in ("f32", "bf16"):
        per_op = results[dt]
        peaks = {op: max(sweep.values()) for op, sweep in per_op.items()}
        sat_ns = {op: _saturation_n(sweep) for op, sweep in per_op.items()}
        peak_tflops = max(peaks.values())
        utilization = peak_tflops / theoretical_peak_tflops
        print(
            f"  → v5e VPU {dt:<4} bandwidth ≈ {peak_tflops:.2f} TFLOPs  "
            f"(add {peaks['add']:.2f} @ sat N={sat_ns['add']}, "
            f"mul {peaks['mul']:.2f} @ sat N={sat_ns['mul']})"
        )
        print(
            f"      = {utilization * 100:.0f}% of W=4 ALUs * 1.5 GHz peak "
            f"({theoretical_peak_tflops:.2f} TFLOPs)"
        )
    print()

    # 4. Carry-dtype cross-check. The bf16 deficit above comes from per-iter
    # bf16<->f32 conversions the TPU backend inserts below Mosaic — the v5e
    # VPU has no native bf16 ALU. Holding I/O as bf16 but carrying in f32
    # amortizes the conversion (once at load, once at store) over K iters and
    # recovers the f32 plateau — a ~3x speedup at DEPTH=64 (matches the "1
    # bf16 op = 3 hw ops" expectation from the chain serialising extf+addf+
    # truncf).
    carry = _run_vpu_carry_probe()
    bf16_peak = max(carry["bf16"].values())
    f32_peak = max(carry["f32"].values())
    print(
        f"  → bf16 I/O with f32 carry ≈ {f32_peak:.2f} TFLOPs "
        f"(matches f32 plateau); with bf16 carry ≈ {bf16_peak:.2f} TFLOPs "
        f"({bf16_peak / f32_peak * 100:.0f}% — per-iter conversion cost)"
    )


if __name__ == "__main__":
    main()
