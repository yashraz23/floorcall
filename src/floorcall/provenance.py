"""Where a number came from: the commit that produced it."""

from __future__ import annotations

import os
import subprocess

from floorcall.config import REPO_ROOT


def git_head() -> str:
    """Short HEAD sha, suffixed `-dirty` if tracked files have uncommitted changes.

    Outside a git checkout (the Kaggle bundle, docs/runbook-kaggle.md) it is FLOORCALL_CODE, which
    the bundle sets to the clean commit it was built from.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("FLOORCALL_CODE", "unknown")
    return f"{sha}{'-dirty' if dirty else ''}"
