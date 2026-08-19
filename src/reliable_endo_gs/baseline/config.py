"""Strict, machine-portable configuration for Endo-E2E-GS integration."""

from dataclasses import dataclass
from pathlib import Path

import yaml


class BaselineConfigError(ValueError):
    """Raised when a baseline integration configuration is invalid."""


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Configuration that identifies an unmodified pinned baseline execution."""

    upstream_root: Path
    upstream_commit: str
    upstream_config: Path
    entry_point: str
    device: str
    enforce_clean: bool
    checkpoint_id: str | None = None
    checkpoint_path: Path | None = None
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if len(self.upstream_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.upstream_commit
        ):
            raise BaselineConfigError("upstream_commit must be a full lowercase 40-character SHA")
        if not self.entry_point.strip():
            raise BaselineConfigError("entry_point must be non-empty")
        if not self.device.strip():
            raise BaselineConfigError("device must be non-empty")
        checkpoint_fields = (
            self.checkpoint_id,
            self.checkpoint_path,
            self.checkpoint_sha256,
        )
        if any(value is not None for value in checkpoint_fields) and not all(
            value is not None for value in checkpoint_fields
        ):
            raise BaselineConfigError(
                "checkpoint_id, checkpoint_path, and checkpoint_sha256 must be set together"
            )
        if self.checkpoint_sha256 is not None and (
            len(self.checkpoint_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.checkpoint_sha256)
        ):
            raise BaselineConfigError("checkpoint_sha256 must be a lowercase SHA-256 digest")


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise BaselineConfigError(f"{context} must be a string-keyed mapping")
    return value


def _optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise BaselineConfigError(f"{context} must be null or a non-empty string")
    return value


def load_baseline_config(path: Path) -> BaselineConfig:
    """Load a Plan-01 baseline config without changing the project config schema."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BaselineConfigError(f"unable to read baseline config {path}: {error}") from error
    except yaml.YAMLError as error:
        raise BaselineConfigError(f"invalid YAML in baseline config {path}: {error}") from error

    root = _mapping(raw, "baseline configuration")
    if set(root) != {"schema_version", "baseline"}:
        raise BaselineConfigError("baseline configuration requires schema_version and baseline")
    if root["schema_version"] != "1.0":
        raise BaselineConfigError("unsupported baseline configuration schema_version")
    baseline = _mapping(root["baseline"], "baseline")
    required = {
        "upstream_root",
        "upstream_commit",
        "upstream_config",
        "entry_point",
        "device",
        "enforce_clean",
        "checkpoint_id",
        "checkpoint_path",
        "checkpoint_sha256",
    }
    if set(baseline) != required:
        missing = sorted(required - set(baseline))
        unexpected = sorted(set(baseline) - required)
        raise BaselineConfigError(
            f"baseline fields mismatch; missing={missing}, unexpected={unexpected}"
        )

    strings: dict[str, str] = {}
    for key in ("upstream_root", "upstream_commit", "upstream_config", "entry_point", "device"):
        value = baseline[key]
        if not isinstance(value, str) or not value.strip():
            raise BaselineConfigError(f"baseline.{key} must be a non-empty string")
        strings[key] = value
    if not isinstance(baseline["enforce_clean"], bool):
        raise BaselineConfigError("baseline.enforce_clean must be boolean")

    checkpoint_id = _optional_string(baseline["checkpoint_id"], "baseline.checkpoint_id")
    checkpoint_path = _optional_string(baseline["checkpoint_path"], "baseline.checkpoint_path")
    checkpoint_sha256 = _optional_string(
        baseline["checkpoint_sha256"], "baseline.checkpoint_sha256"
    )
    return BaselineConfig(
        upstream_root=Path(strings["upstream_root"]),
        upstream_commit=strings["upstream_commit"],
        upstream_config=Path(strings["upstream_config"]),
        entry_point=strings["entry_point"],
        device=strings["device"],
        enforce_clean=baseline["enforce_clean"],
        checkpoint_id=checkpoint_id,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path is not None else None,
        checkpoint_sha256=checkpoint_sha256,
    )
