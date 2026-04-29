"""Auto-skip TPU-marked tests when no TPU is available.

CLAUDE.md says "CPU-only runs skip TPU-marked tests"; this conftest is
what makes that true. Tests marked `@pytest.mark.tpu` (which includes
everything in `tests/perf/`) are skipped unless `jax.devices()` reports
a TPU backend.
"""

from __future__ import annotations

import jax
import pytest


def _has_tpu() -> bool:
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except RuntimeError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _has_tpu():
        return
    skip_tpu = pytest.mark.skip(reason="no TPU available")
    for item in items:
        if "tpu" in item.keywords:
            item.add_marker(skip_tpu)
