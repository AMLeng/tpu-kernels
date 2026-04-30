"""Shared bits for ``bench_history/`` writers (compare + sweep).

Both ``compare()`` and ``sweep()`` write timestamped JSON records to
``bench_history/<op>/`` and stamp each record with the current git SHA, so
the path resolution and SHA lookup live here rather than being duplicated.

Pulling these out also gives tests a single module to monkeypatch when
they need to redirect history writes to ``tmp_path``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = REPO_ROOT / "bench_history"


def git_sha() -> str | None:
    """Resolve HEAD's SHA, with a "-dirty" suffix if the working tree has changes.

    Trend tracking from ``bench_history/`` is only meaningful when each record
    can be tied to a specific tree state — a clean SHA on a dirty tree
    silently lies about that. The dirty check uses ``git status --porcelain``
    so untracked files count too (a forgotten ``.py`` in ``benchmarks/``
    would affect the run).
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if head.returncode != 0:
            return None
        sha = head.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if status.returncode == 0 and status.stdout.strip():
            sha = f"{sha}-dirty"
        return sha
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
