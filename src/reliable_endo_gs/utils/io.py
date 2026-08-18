"""Focused filesystem helpers for infrastructure outputs."""

import json
import os
import tempfile
from pathlib import Path

import yaml


def ensure_directory(path: Path) -> Path:
    """Create ``path`` and its parents if needed, then return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def create_unique_directory(output_root: Path, base_name: str) -> Path:
    """Create a new run directory without overwriting an existing run."""

    ensure_directory(output_root)
    for index in range(1000):
        suffix = "" if index == 0 else f"-{index:02d}"
        candidate = output_root / f"{base_name}{suffix}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"Unable to allocate a unique run directory under {output_root}")


def _write_text_atomic(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, data: object) -> None:
    """Serialize ``data`` as deterministic, human-readable JSON atomically."""

    content = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_text_atomic(path, content)


def write_yaml_atomic(path: Path, data: object) -> None:
    """Serialize ``data`` as readable YAML atomically."""

    content = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    _write_text_atomic(path, content)
