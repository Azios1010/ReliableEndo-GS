"""Resumable, externally stored A01 disparity target cache."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch

from reliable_endo_gs.teachers.roles import DEFAULT_A01_SCARED_ROLES, A01ScaredDataRoles
from reliable_endo_gs.teachers.targets import AlignedTeacherTarget
from reliable_endo_gs.utils.io import write_json_atomic


class TargetCacheError(ValueError):
    """Raised when a target cache manifest or item is invalid."""


TARGET_CACHE_MANIFEST_SCHEMA = 2


def _canonicalize(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TargetCacheError(f"cache metadata value {type(value).__name__} is not JSON compatible")


def _canonical_json(value: object) -> bytes:
    try:
        normalized = _canonicalize(value)
        return json.dumps(
            normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TargetCacheError(f"cache metadata is not JSON-compatible: {error}") from error


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class TargetSchema:
    """Initial target schema; large/derived signals are deliberately absent."""

    name: str = "a01_teacher_disparity"
    version: int = 1
    fields: tuple[str, ...] = ("disparity", "validity", "support", "metadata")

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version, "fields": list(self.fields)}

    @property
    def schema_hash(self) -> str:
        return _sha256(_canonical_json(self.to_dict()))


DEFAULT_TARGET_SCHEMA = TargetSchema()


@dataclass(frozen=True, slots=True)
class TargetCacheKey:
    """All identities that must match before one target can be reused."""

    sample_id: str
    split_identity: str
    data_role: str
    image_transform_hash: str
    calibration_transform_hash: str
    teacher_identity_hash: str
    target_schema_hash: str = field(default_factory=lambda: DEFAULT_TARGET_SCHEMA.schema_hash)
    role_contract_hash: str = field(default_factory=lambda: DEFAULT_A01_SCARED_ROLES.role_hash)

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "split_identity",
            "data_role",
            "image_transform_hash",
            "calibration_transform_hash",
            "teacher_identity_hash",
            "target_schema_hash",
            "role_contract_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TargetCacheError(f"cache key field {name!r} must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "split_identity": self.split_identity,
            "data_role": self.data_role,
            "image_transform_hash": self.image_transform_hash,
            "calibration_transform_hash": self.calibration_transform_hash,
            "teacher_identity_hash": self.teacher_identity_hash,
            "target_schema_hash": self.target_schema_hash,
            "role_contract_hash": self.role_contract_hash,
        }

    @property
    def digest(self) -> str:
        """Return the content-addressed item path identity."""

        return _sha256(_canonical_json(self.to_dict()))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TargetCacheKey:
        expected = {
            "sample_id",
            "split_identity",
            "data_role",
            "image_transform_hash",
            "calibration_transform_hash",
            "teacher_identity_hash",
            "target_schema_hash",
            "role_contract_hash",
        }
        if set(value) != expected:
            raise TargetCacheError("cache key has unsupported or missing fields")
        return cls(**{key: str(value[key]) for key in expected})


@dataclass(frozen=True, slots=True)
class TargetCacheResult:
    """Result of a resumable lookup or computation."""

    target: AlignedTeacherTarget
    cache_hit: bool


class TargetCache:
    """Atomic manifest-backed cache stored outside the repository.

    One item is stored under ``targets/<digest>.pt`` and contains only CPU
    disparity, validity, support and JSON preprocessing/calibration metadata.
    The manifest records the complete key, item checksum, size and shape.  A
    key change therefore produces a cache miss without overwriting an earlier
    successful target.
    """

    def __init__(
        self,
        root: Path,
        *,
        target_schema: TargetSchema = DEFAULT_TARGET_SCHEMA,
        roles: A01ScaredDataRoles = DEFAULT_A01_SCARED_ROLES,
        read_only: bool = False,
    ) -> None:
        if not isinstance(target_schema, TargetSchema):
            raise TargetCacheError("target_schema must be a TargetSchema")
        if not isinstance(roles, A01ScaredDataRoles):
            raise TargetCacheError("A01 target cache requires the validated SCARED role contract")
        self.root = Path(root)
        self.target_schema = target_schema
        self.roles = roles
        self.read_only = read_only
        self.manifest_path = self.root / "manifest.json"
        self._entries: dict[str, dict[str, object]] = {}
        if read_only:
            if not self.root.is_dir() or not self.manifest_path.is_file():
                raise TargetCacheError("read-only target cache requires an existing manifest")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            payload: object = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TargetCacheError(f"unable to read target cache manifest: {error}") from error
        if not isinstance(payload, Mapping):
            raise TargetCacheError("target cache manifest must be an object")
        if payload.get("manifest_schema_version") != TARGET_CACHE_MANIFEST_SCHEMA:
            raise TargetCacheError("unsupported target cache manifest schema")
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise TargetCacheError("target cache manifest entries must be a list")
        if payload.get("target_schema") != self.target_schema.to_dict():
            raise TargetCacheError("target cache target schema does not match configuration")
        if payload.get("role_contract_hash") != self.roles.role_hash:
            raise TargetCacheError("target cache role contract does not match configuration")
        manifest_hash = payload.get("manifest_sha256")
        if not isinstance(manifest_hash, str) or not manifest_hash:
            raise TargetCacheError("target cache manifest checksum is required")
        without_hash = {
            "manifest_schema_version": payload.get("manifest_schema_version"),
            "target_schema": payload.get("target_schema"),
            "role_contract_hash": payload.get("role_contract_hash"),
            "entries": raw_entries,
        }
        if manifest_hash != _sha256(_canonical_json(without_hash)):
            raise TargetCacheError("target cache manifest checksum mismatch")
        entries: dict[str, dict[str, object]] = {}
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                raise TargetCacheError("target cache manifest entry must be an object")
            entry = dict(cast(Mapping[str, object], raw_entry))
            raw_key = entry.get("key")
            digest = entry.get("digest")
            if not isinstance(raw_key, Mapping) or not isinstance(digest, str):
                raise TargetCacheError("target cache manifest entry is missing key/digest")
            key = TargetCacheKey.from_dict(cast(Mapping[str, object], raw_key))
            if digest != key.digest or digest in entries:
                raise TargetCacheError("target cache manifest has an invalid or duplicate digest")
            required = {"digest", "key", "relative_path", "sha256", "size", "shape"}
            if set(entry) != required:
                raise TargetCacheError("target cache manifest entry has unsupported fields")
            relative_path = entry["relative_path"]
            entry_sha256 = entry["sha256"]
            size = entry["size"]
            shape = entry["shape"]
            if (
                not isinstance(relative_path, str)
                or not relative_path
                or Path(relative_path).is_absolute()
                or ".." in Path(relative_path).parts
            ):
                raise TargetCacheError("target cache manifest entry has an invalid relative path")
            if not isinstance(entry_sha256, str) or not entry_sha256:
                raise TargetCacheError("target cache manifest entry checksum is required")
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise TargetCacheError("target cache manifest entry size is invalid")
            if (
                not isinstance(shape, list)
                or not shape
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item < 1
                    for item in shape
                )
            ):
                raise TargetCacheError("target cache manifest entry shape is invalid")
            self._validate_key(key)
            entries[digest] = entry
        self._entries = entries

    def _manifest_payload(self) -> dict[str, object]:
        without_hash: dict[str, object] = {
            "manifest_schema_version": TARGET_CACHE_MANIFEST_SCHEMA,
            "target_schema": self.target_schema.to_dict(),
            "role_contract_hash": self.roles.role_hash,
            "entries": [self._entries[digest] for digest in sorted(self._entries)],
        }
        payload = dict(without_hash)
        payload["manifest_sha256"] = _sha256(_canonical_json(without_hash))
        return payload

    def _item_path(self, digest: str) -> Path:
        return self.root / "targets" / digest[:2] / f"{digest}.pt"

    def _validate_role(self, key: TargetCacheKey) -> None:
        if "/" not in key.sample_id:
            raise TargetCacheError("role validation requires a portable sample_id")
        dataset_id = key.sample_id.split("/", maxsplit=1)[0]
        try:
            self.roles.validate_sample_id(key.sample_id, role=key.data_role, cache=True)
        except ValueError as error:
            raise TargetCacheError(str(error)) from error
        if dataset_id not in self.roles.datasets_for_role(key.data_role):
            raise TargetCacheError("cache key dataset does not match declared data role")

    def _validate_key(self, key: TargetCacheKey) -> None:
        if key.target_schema_hash != self.target_schema.schema_hash:
            raise TargetCacheError("cache key target schema does not match configuration")
        if key.role_contract_hash != self.roles.role_hash:
            raise TargetCacheError("cache key role contract does not match configuration")
        self._validate_role(key)

    @staticmethod
    def _target_tensors(
        target: AlignedTeacherTarget,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if target.disparity.ndim != 4 or target.disparity.shape[0] != 1:
            raise TargetCacheError("cache items must contain one sample [1,1,H,W]")
        if not target.disparity.is_floating_point():
            raise TargetCacheError("cache disparity must be floating point")
        disparity = target.disparity.detach().to(device="cpu", dtype=torch.float32).squeeze(0)
        validity = target.validity.detach().to(device="cpu", dtype=torch.bool).squeeze(0)
        support = target.support.detach().to(device="cpu", dtype=torch.bool).squeeze(0)
        return disparity, validity, support

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(path)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    def put(self, key: TargetCacheKey, target: AlignedTeacherTarget) -> None:
        """Atomically store one target and update its checksummed manifest."""
        if self.read_only:
            raise TargetCacheError("cannot write a read-only target cache")
        self._validate_key(key)
        disparity, validity, support = self._target_tensors(target)
        try:
            metadata = target.cache_metadata()
            _canonical_json(metadata)
        except (TypeError, ValueError) as error:
            raise TargetCacheError(f"target metadata cannot be cached: {error}") from error
        payload = {
            "schema_version": self.target_schema.version,
            "disparity": disparity,
            "validity": validity,
            "support": support,
            "metadata": metadata,
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        serialized = buffer.getvalue()
        digest = key.digest
        item_path = self._item_path(digest)
        self._atomic_write_bytes(item_path, serialized)
        entry = {
            "digest": digest,
            "key": key.to_dict(),
            "relative_path": item_path.relative_to(self.root).as_posix(),
            "sha256": _sha256(serialized),
            "size": len(serialized),
            "shape": list(disparity.shape),
        }
        self._entries[digest] = entry
        write_json_atomic(self.manifest_path, self._manifest_payload())

    def get(self, key: TargetCacheKey) -> AlignedTeacherTarget | None:
        """Return a verified hit or ``None`` for a miss/stale/corrupt item."""

        self._validate_key(key)
        entry = self._entries.get(key.digest)
        if entry is None:
            return None
        relative_path = entry.get("relative_path")
        if not isinstance(relative_path, str):
            return None
        item_path = self.root / Path(relative_path)
        try:
            resolved_root = self.root.resolve()
            if resolved_root not in item_path.resolve().parents:
                return None
            serialized = item_path.read_bytes()
        except OSError:
            return None
        size = entry.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or len(serialized) != size:
            return None
        if _sha256(serialized) != entry.get("sha256"):
            return None
        try:
            buffer = io.BytesIO(serialized)
            try:
                payload: object = torch.load(buffer, map_location="cpu", weights_only=True)
            except TypeError:
                buffer.seek(0)
                payload = torch.load(buffer, map_location="cpu")
        except Exception:
            return None
        if not isinstance(payload, Mapping):
            return None
        if set(payload) != {"schema_version", "disparity", "validity", "support", "metadata"}:
            return None
        if payload.get("schema_version") != self.target_schema.version:
            return None
        disparity = payload.get("disparity")
        validity = payload.get("validity")
        support = payload.get("support")
        metadata = payload.get("metadata")
        if not isinstance(disparity, torch.Tensor) or not isinstance(validity, torch.Tensor):
            return None
        if not isinstance(support, torch.Tensor) or not isinstance(metadata, Mapping):
            return None
        if (
            disparity.ndim != 3
            or validity.shape != disparity.shape
            or support.shape != disparity.shape
        ):
            return None
        if (
            disparity.shape[0] != 1
            or not disparity.is_floating_point()
            or validity.dtype != torch.bool
            or support.dtype != torch.bool
        ):
            return None
        if entry.get("shape") != list(disparity.shape):
            return None
        return AlignedTeacherTarget(
            disparity=disparity.unsqueeze(0),
            validity=validity.unsqueeze(0),
            support=support.unsqueeze(0),
            metadata=dict(cast(Mapping[str, object], metadata)),
        )

    lookup = get

    def get_or_compute(
        self,
        key: TargetCacheKey,
        producer: Callable[[], AlignedTeacherTarget],
    ) -> TargetCacheResult:
        """Resume from a verified hit or compute/store exactly one missing item."""

        cached = self.get(key)
        if cached is not None:
            return TargetCacheResult(cached, cache_hit=True)
        if self.read_only:
            raise TargetCacheError("read-only cache miss; producer will not be called")
        target = producer()
        self.put(key, target)
        return TargetCacheResult(target, cache_hit=False)

    @property
    def entries(self) -> tuple[dict[str, object], ...]:
        """Return a deterministic read-only snapshot of manifest entries."""

        return tuple(self._entries[digest].copy() for digest in sorted(self._entries))


__all__ = [
    "DEFAULT_TARGET_SCHEMA",
    "TARGET_CACHE_MANIFEST_SCHEMA",
    "TargetCache",
    "TargetCacheError",
    "TargetCacheKey",
    "TargetCacheResult",
    "TargetSchema",
]
