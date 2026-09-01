"""Immutable development artifact and provenance schema for baseline runs."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

ARTIFACT_SCHEMA_NAME = "reliable_endo_gs.baseline_artifact"
ARTIFACT_SCHEMA_VERSION = "1.0"


class ArtifactError(ValueError):
    """Raised when a baseline artifact is incomplete or invalid."""


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class ArtifactPayload:
    """A referenced payload with a content digest and byte size."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.path or Path(self.path).is_absolute():
            raise ArtifactError("payload path must be a non-empty relative path")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ArtifactError("payload sha256 must be a lowercase 64-character digest")
        if self.size_bytes < 0:
            raise ArtifactError("payload size_bytes must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class BaselineArtifact:
    """Validated identity-bearing manifest; payloads are never overwritten."""

    artifact_id: str
    created_utc: str
    project_git_commit: str | None
    project_git_dirty: bool | None
    config: Mapping[str, object]
    config_hash: str
    dataset: Mapping[str, object]
    split: Mapping[str, object]
    sample_selection: Mapping[str, object]
    seeds: Mapping[str, object]
    hardware: Mapping[str, object]
    upstream: Mapping[str, object]
    checkpoint: Mapping[str, object]
    metric_schema: Mapping[str, object]
    metrics: Mapping[str, object]
    profiling: Mapping[str, object]
    payloads: Sequence[ArtifactPayload] = ()
    warnings: Sequence[str] = ()
    gate: Mapping[str, object] = field(default_factory=dict)
    artifact_type: str = "baseline_reproduction"
    schema_name: str = ARTIFACT_SCHEMA_NAME
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.artifact_type != "baseline_reproduction":
            raise ArtifactError("unsupported artifact_type")
        if (
            self.schema_name != ARTIFACT_SCHEMA_NAME
            or self.schema_version != ARTIFACT_SCHEMA_VERSION
        ):
            raise ArtifactError("unsupported baseline artifact schema")
        if (
            not self.artifact_id
            or any(ch not in "0123456789abcdef" for ch in self.artifact_id)
            or len(self.artifact_id) != 64
        ):
            raise ArtifactError("artifact_id must be a lowercase SHA-256 digest")
        if len(self.config_hash) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.config_hash
        ):
            raise ArtifactError("config_hash must be a lowercase SHA-256 digest")
        if self.project_git_dirty not in (True, False, None):
            raise ArtifactError("project_git_dirty must be boolean or null")
        if not isinstance(self.created_utc, str) or not self.created_utc:
            raise ArtifactError("created_utc must be a non-empty ISO-8601 string")
        if not isinstance(self.payloads, Sequence) or any(
            not isinstance(payload, ArtifactPayload) for payload in self.payloads
        ):
            raise ArtifactError("payloads must contain ArtifactPayload values")
        if not isinstance(self.warnings, Sequence) or any(
            not isinstance(warning, str) for warning in self.warnings
        ):
            raise ArtifactError("warnings must contain strings")
        mappings = {
            "config": self.config,
            "dataset": self.dataset,
            "split": self.split,
            "sample_selection": self.sample_selection,
            "seeds": self.seeds,
            "hardware": self.hardware,
            "upstream": self.upstream,
            "checkpoint": self.checkpoint,
            "metric_schema": self.metric_schema,
            "metrics": self.metrics,
            "profiling": self.profiling,
            "gate": self.gate,
        }
        if any(not isinstance(value, Mapping) for value in mappings.values()):
            raise ArtifactError("artifact metadata fields must be mappings")
        object.__setattr__(self, "config", _freeze(self.config))
        object.__setattr__(self, "dataset", _freeze(self.dataset))
        object.__setattr__(self, "split", _freeze(self.split))
        object.__setattr__(self, "sample_selection", _freeze(self.sample_selection))
        object.__setattr__(self, "seeds", _freeze(self.seeds))
        object.__setattr__(self, "hardware", _freeze(self.hardware))
        object.__setattr__(self, "upstream", _freeze(self.upstream))
        object.__setattr__(self, "checkpoint", _freeze(self.checkpoint))
        object.__setattr__(self, "metric_schema", _freeze(self.metric_schema))
        object.__setattr__(self, "metrics", _freeze(self.metrics))
        object.__setattr__(self, "profiling", _freeze(self.profiling))
        object.__setattr__(self, "gate", _freeze(self.gate))
        object.__setattr__(self, "payloads", tuple(self.payloads))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if artifact_digest(self.to_dict(include_id=False)) != self.artifact_id:
            raise ArtifactError("artifact_id does not match manifest content")

    def to_dict(self, *, include_id: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "artifact_type": self.artifact_type,
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "project_git_commit": self.project_git_commit,
            "project_git_dirty": self.project_git_dirty,
            "config": _jsonable(self.config),
            "config_hash": self.config_hash,
            "dataset": _jsonable(self.dataset),
            "split": _jsonable(self.split),
            "sample_selection": _jsonable(self.sample_selection),
            "seeds": _jsonable(self.seeds),
            "hardware": _jsonable(self.hardware),
            "upstream": _jsonable(self.upstream),
            "checkpoint": _jsonable(self.checkpoint),
            "metric_schema": _jsonable(self.metric_schema),
            "metrics": _jsonable(self.metrics),
            "profiling": _jsonable(self.profiling),
            "payloads": [payload.to_dict() for payload in self.payloads],
            "warnings": list(self.warnings),
            "gate": _jsonable(self.gate),
        }
        if include_id:
            result["artifact_id"] = self.artifact_id
        return result


def artifact_digest(fields: Mapping[str, object]) -> str:
    encoded = json.dumps(
        _jsonable(fields), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_artifact(
    *,
    config: Mapping[str, object],
    config_hash: str,
    dataset: Mapping[str, object],
    split: Mapping[str, object],
    sample_selection: Mapping[str, object],
    seeds: Mapping[str, object],
    upstream: Mapping[str, object],
    checkpoint: Mapping[str, object],
    metric_schema: Mapping[str, object],
    metrics: Mapping[str, object],
    profiling: Mapping[str, object],
    project_git_commit: str | None = None,
    project_git_dirty: bool | None = None,
    hardware: Mapping[str, object] | None = None,
    payloads: Sequence[ArtifactPayload] = (),
    warnings: Sequence[str] = (),
    gate: Mapping[str, object] | None = None,
    created_utc: str | None = None,
) -> BaselineArtifact:
    body = {
        "artifact_type": "baseline_reproduction",
        "schema_name": ARTIFACT_SCHEMA_NAME,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": created_utc or _timestamp(),
        "project_git_commit": project_git_commit,
        "project_git_dirty": project_git_dirty,
        "config": config,
        "config_hash": config_hash,
        "dataset": dataset,
        "split": split,
        "sample_selection": sample_selection,
        "seeds": seeds,
        "hardware": (
            hardware
            if hardware is not None
            else {"python": sys.version, "platform": platform.platform()}
        ),
        "upstream": upstream,
        "checkpoint": checkpoint,
        "metric_schema": metric_schema,
        "metrics": metrics,
        "profiling": profiling,
        "payloads": [item.to_dict() for item in payloads],
        "warnings": list(warnings),
        "gate": gate if gate is not None else {},
    }
    constructor_body: dict[str, object] = dict(body)
    constructor_body["payloads"] = tuple(payloads)
    return BaselineArtifact(
        artifact_id=artifact_digest(body),
        **constructor_body,  # type: ignore[arg-type]
    )


def write_artifact(path: Path, artifact: BaselineArtifact) -> None:
    """Write a finalized artifact atomically and refuse replacement."""

    if path.exists():
        raise ArtifactError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging")
    temporary.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_artifact(path: Path) -> BaselineArtifact:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"unable to read artifact: {error}") from error
    if not isinstance(raw, dict):
        raise ArtifactError("artifact must contain a JSON object")
    artifact_id = raw.pop("artifact_id", None)
    if not isinstance(artifact_id, str):
        raise ArtifactError("artifact is missing artifact_id")
    computed = artifact_digest(raw)
    if computed != artifact_id:
        raise ArtifactError("artifact_id does not match manifest content")
    try:
        payloads = tuple(ArtifactPayload(**item) for item in raw.pop("payloads", []))
        return BaselineArtifact(artifact_id=artifact_id, payloads=payloads, **raw)
    except (ArtifactError, TypeError, KeyError) as error:
        raise ArtifactError(f"invalid baseline artifact fields: {error}") from error


create_baseline_artifact = make_artifact
BaselineArtifactManifest = BaselineArtifact
ArtifactManifest = BaselineArtifact
