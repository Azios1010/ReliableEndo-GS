"""Read-only Git metadata helpers."""

import subprocess
from collections.abc import Sequence
from pathlib import Path


def _run_git(repo_root: Path, arguments: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def get_git_commit(repo_root: Path) -> str | None:
    """Return the current commit, or ``None`` when it cannot be determined."""

    commit = _run_git(repo_root, ["rev-parse", "HEAD"])
    return commit or None


def is_git_dirty(repo_root: Path) -> bool | None:
    """Return whether Git reports changes, or ``None`` if Git is unavailable."""

    status = _run_git(repo_root, ["status", "--porcelain"])
    if status is None:
        return None
    return bool(status)
