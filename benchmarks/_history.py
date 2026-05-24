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

    Three classes of path are excluded from the porcelain check:

    * ``bench_history/`` — output, not input. A JSON record from a previous
      run can't have affected the run we're about to stamp; without the
      exclude, back-to-back compare() calls would falsely mark the second
      one dirty.
    * ``tests/`` — also not bench input. Test code doesn't change kernel
      behavior or bench output, so iterating on a test (e.g. tightening a
      threshold, adding ``@pytest.mark.xfail``) shouldn't invalidate
      JSONs taken under the same kernel code.
    * dotfiles (any path whose *basename* starts with ``.``) — editor/OS
      cruft like a vim ``.foo.py.swp`` swap or ``.DS_Store``. These aren't
      bench input, and an open editor session would otherwise mark every
      run dirty. Only the basename is checked, so a real source file inside
      a dot-*directory* (e.g. ``.github/workflows/ci.yml``) still counts.
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
            [
                "git",
                "status",
                "--porcelain",
                "--",
                ".",
                ":(exclude)bench_history",
                ":(exclude)tests",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if status.returncode == 0 and _has_dirtying_entry(status.stdout):
            sha = f"{sha}-dirty"
        return sha
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _has_dirtying_entry(porcelain: str) -> bool:
    """True if any ``git status --porcelain`` line names a non-dotfile.

    Each porcelain v1 line is ``XY <path>`` (path from column 3); a rename is
    ``old -> new``, in which case the new path is what's on disk. We ignore
    entries whose basename starts with ``.`` so editor swap files and the like
    don't taint the SHA.
    """
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        path = line[3:]
        if " -> " in path:  # rename/copy: "old -> new"
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"')
        if not Path(path).name.startswith("."):
            return True
    return False
