"""``Workload``: the "what is being benched" bundle shared by compare/sweep.

Both ``compare()`` and ``sweep()`` need the same five things to do their job:
the op name (for logging and history paths), the input arrays, the per-call
flop and byte counts, and the dtype the flops are counted in. Every suite
builds these together — they're tightly coupled (you can't change ``flops``
without rethinking ``args`` and ``nbytes``) — so bundling them as a single
parameter makes the call sites short and the suites' two harness calls
share one source of truth.

This is the only such bundle. Hardware (``hw``) defaults at the harness
level via ``v5e()``; bench knobs (``warmup``, ``iters``, ``timing``) and
output knobs (``dump_hlo``, ``profile_dir``, ``write_history``) stay as
explicit kwargs because they're rarely overridden together and grouping
them would just shuffle complexity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from benchmarks.roofline import FlopDtype


@dataclass(frozen=True, slots=True)
class Workload:
    """The "what is being benched" bundle for compare()/sweep().

    Attributes:
        op: Op name. Used in bench-row labels (``f"{op}::{variant}"``)
            and the ``bench_history/<op>/`` directory.
        flops: Floating-point ops per call. Drives the MFU% / SoL
            denominator and the regime classification.
        nbytes: Bytes moved to/from HBM per call. Drives the BW% / SoL
            denominator. Includes both reads and writes.
        args: Input arrays passed to every variant. Same for all variants
            in a compare(), same across all swept configs in a sweep().
        flop_dtype: Which peak rate to compare against — bf16 / f32 / int8.
            v5e's MXU is bf16-native; f32 emulation runs at roughly half
            rate; int8 gets the highest peak.
    """

    op: str
    flops: int
    nbytes: int
    args: Sequence[Any] = ()
    flop_dtype: FlopDtype = "bf16"
