"""Explicit checkpoint hashing and upstream-compatible inference loading."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch


class CheckpointError(RuntimeError):
    """Raised when a baseline checkpoint is missing, changed, or incompatible."""


@dataclass(frozen=True, slots=True)
class CheckpointLoadResult:
    """Identity and state-dict compatibility observed during one load."""

    path: Path
    sha256: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of an existing regular file."""

    if not path.is_file():
        raise CheckpointError(f"checkpoint is not a readable file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise CheckpointError(f"unable to hash checkpoint {path}: {error}") from error
    return digest.hexdigest()


def load_upstream_checkpoint(
    model: torch.nn.Module,
    path: Path,
    *,
    expected_sha256: str,
    map_location: str | torch.device,
    strict: bool = True,
) -> CheckpointLoadResult:
    """Load the official ``checkpoint['network']`` state dictionary explicitly.

    Only trusted upstream checkpoints should be passed to ``torch.load``.
    Mutable names such as ``latest.pth`` never replace the required digest.
    """

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise CheckpointError(
            f"checkpoint SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise CheckpointError(f"unable to load checkpoint {path}: {error}") from error
    if not isinstance(checkpoint, dict) or "network" not in checkpoint:
        raise CheckpointError("upstream checkpoint must be a mapping containing 'network'")
    state_dict = checkpoint["network"]
    if not isinstance(state_dict, dict):
        raise CheckpointError("upstream checkpoint field 'network' must be a state dictionary")
    try:
        incompatible = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as error:
        raise CheckpointError(f"checkpoint is incompatible with the model: {error}") from error
    return CheckpointLoadResult(
        path=path,
        sha256=actual_sha256,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )
