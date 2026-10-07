"""Canonical first-party source identity used by experimental artifacts."""

import hashlib
from pathlib import Path


def source_sha256(root: Path) -> str:
    """Hash ordered relative Python paths plus bytes, matching student v1."""
    digest = hashlib.sha256()
    for path in sorted((root / "src/reliable_endo_gs").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
