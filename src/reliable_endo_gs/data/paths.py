"""Explicit external dataset root resolution."""

import os
from pathlib import Path

DATA_ROOT_ENV_VAR = "RELIABLE_ENDO_DATA_ROOT"


class DataRootResolutionError(ValueError):
    """Raised when a portable dataset root cannot be resolved safely."""


def _require_absolute_global_root(root: Path, source: str) -> Path:
    if not root.is_absolute():
        raise DataRootResolutionError(
            f"{source} must be an absolute path; received {root!s}. "
            "Relative global roots would depend on the current working directory."
        )
    return root


def resolve_data_root(
    dataset_root: Path,
    *,
    global_root: Path | None = None,
    env_var: str = DATA_ROOT_ENV_VAR,
) -> Path:
    """Resolve a dataset root without consulting or validating the filesystem.

    Absolute dataset roots are explicit overrides and are preserved. Relative
    roots require either an explicit absolute ``global_root`` or an absolute
    path supplied through ``env_var``.
    """

    if dataset_root.is_absolute():
        return dataset_root

    if global_root is not None:
        base_root = _require_absolute_global_root(global_root, "global_root")
        return base_root / dataset_root

    environment_value = os.environ.get(env_var)
    if environment_value is None or not environment_value.strip():
        raise DataRootResolutionError(
            f"Cannot resolve relative dataset root {dataset_root!s}: set {env_var} "
            "to an absolute mounted-data directory or pass an explicit global_root."
        )

    base_root = _require_absolute_global_root(Path(environment_value), env_var)
    return base_root / dataset_root
