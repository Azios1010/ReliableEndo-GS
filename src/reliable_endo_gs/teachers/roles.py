"""A01 SCARED-only data roles and final-set leakage guards.

This module is intentionally independent of all historical dataset adapters.
It validates portable dataset identifiers and role semantics only; external
data roots are resolved by an explicit caller at collection time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import yaml


class A01RoleError(ValueError):
    """Raised when the active A01 data-role contract is unsafe."""


A01_DATASET_NAME = "scared"
A01_DATASET_IDS: tuple[str, ...] = tuple(f"dataset_{index}" for index in range(1, 8))
A01_FINAL_DATASET_IDS: tuple[str, ...] = ()
A01_CACHE_ROLES: tuple[str, ...] = ("teacher_fit", "qualification")


def _as_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise A01RoleError(f"{context} must be a string-keyed mapping")
    return dict(cast(Mapping[str, object], value))


def _normal_ids(values: object, context: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise A01RoleError(f"{context} must be a non-empty sequence")
    if not all(isinstance(value, str) for value in values):
        raise A01RoleError(f"{context} must contain strings")
    result = tuple(value.strip() for value in values)
    if not result or any(not value for value in result):
        raise A01RoleError(f"{context} must contain non-empty identifiers")
    if len(set(result)) != len(result):
        raise A01RoleError(f"{context} contains duplicate identifiers")
    if any(value not in A01_DATASET_IDS for value in result):
        raise A01RoleError(f"{context} contains an identifier outside dataset_1..dataset_7")
    return result


@dataclass(frozen=True, slots=True)
class A01ScaredDataRoles:
    """Explicit SCARED fold roles for A01 target collection.

    ``teacher_fit`` is adaptation/cache-development data, ``qualification``
    is held out from fitting for candidate comparison.  All seven user-directed
    SCARED folds are active for training/development; later final-evaluation
    freeze rules are not applied to this A01 target cache.
    """

    dataset_name: str = A01_DATASET_NAME
    root_env_var: str = "RELIABLE_ENDO_DATA_ROOT"
    root_subdir: str = "scared"
    dataset_ids: tuple[str, ...] = A01_DATASET_IDS
    roles: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "teacher_fit": ("dataset_1", "dataset_2", "dataset_3"),
            "qualification": ("dataset_4", "dataset_5", "dataset_6", "dataset_7"),
        }
    )
    cache_roles: tuple[str, ...] = A01_CACHE_ROLES
    final_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dataset_name != A01_DATASET_NAME:
            raise A01RoleError("A01 roles must use the SCARED dataset")
        if not self.root_env_var.strip() or not self.root_subdir.strip():
            raise A01RoleError("root environment variable and relative subdirectory are required")
        root_parts = self.root_subdir.replace("\\", "/").split("/")
        if (
            self.root_subdir.replace("\\", "/").startswith("/")
            or ":" in self.root_subdir
            or any(part in {"", ".", ".."} for part in root_parts)
        ):
            raise A01RoleError("dataset root must be a portable relative subdirectory")
        dataset_ids = tuple(self.dataset_ids)
        if dataset_ids != A01_DATASET_IDS:
            raise A01RoleError("active A01 roles must enumerate dataset_1 through dataset_7")
        normalized_roles = {str(name): tuple(values) for name, values in self.roles.items()}
        if set(normalized_roles) != {"teacher_fit", "qualification"}:
            raise A01RoleError("roles must define teacher_fit and qualification")
        seen: set[str] = set()
        for role_name, values in normalized_roles.items():
            if not values:
                raise A01RoleError(f"role {role_name!r} cannot be empty")
            if any(value not in A01_DATASET_IDS for value in values):
                raise A01RoleError(f"role {role_name!r} contains an unknown dataset")
            overlap = seen & set(values)
            if overlap:
                raise A01RoleError(f"dataset role overlap detected: {sorted(overlap)}")
            seen.update(values)
        if seen != set(A01_DATASET_IDS):
            raise A01RoleError("roles must cover every active dataset exactly once")
        if tuple(self.final_roles):
            raise A01RoleError("A01 target collection has no protected final role")
        if set(self.cache_roles) != set(normalized_roles):
            raise A01RoleError("all active A01 roles must be approved for target caching")
        object.__setattr__(self, "roles", normalized_roles)

    @property
    def final_dataset_ids(self) -> tuple[str, ...]:
        return tuple(dataset_id for role in self.final_roles for dataset_id in self.roles[role])

    def datasets_for_role(self, role: str) -> tuple[str, ...]:
        try:
            return self.roles[role]
        except KeyError as error:
            raise A01RoleError(f"unknown A01 data role {role!r}") from error

    def role_for_dataset(self, dataset_id: str) -> str:
        for role, dataset_ids in self.roles.items():
            if dataset_id in dataset_ids:
                return role
        raise A01RoleError(f"unknown dataset identifier {dataset_id!r}")

    def assert_cache_allowed(self, *, dataset_id: str, role: str) -> None:
        """Reject unknown or otherwise unapproved folds before cache access."""

        expected_role = self.role_for_dataset(dataset_id)
        if role != expected_role:
            raise A01RoleError(
                f"dataset {dataset_id!r} belongs to role {expected_role!r}, not {role!r}"
            )
        if role not in self.cache_roles:
            raise A01RoleError(f"role {role!r} is not approved for A01 target caching")

    def validate_sample_id(self, sample_id: str, *, role: str, cache: bool = True) -> str:
        """Validate ``dataset_N/...`` provenance without reading the dataset."""

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise A01RoleError("sample_id must be a non-empty string")
        normalized = sample_id.strip().replace("\\", "/")
        dataset_id = normalized.split("/", maxsplit=1)[0]
        if dataset_id not in self.dataset_ids:
            raise A01RoleError("sample_id must begin with dataset_1 through dataset_7")
        if "/" not in normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
            raise A01RoleError("sample_id must be a portable relative identifier")
        if cache:
            self.assert_cache_allowed(dataset_id=dataset_id, role=role)
        elif role != self.role_for_dataset(dataset_id):
            raise A01RoleError(f"sample_id role does not match {dataset_id}")
        return normalized

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "dataset_name": self.dataset_name,
            "root_env_var": self.root_env_var,
            "root_subdir": self.root_subdir,
            "dataset_ids": list(self.dataset_ids),
            "roles": {role: list(values) for role, values in self.roles.items()},
            "cache_roles": list(self.cache_roles),
            "final_roles": list(self.final_roles),
            "semantics": {
                "teacher_fit": "candidate adaptation and development target cache",
                "qualification": "held-out candidate comparison before reference freeze",
            },
        }

    @property
    def role_hash(self) -> str:
        """Return a stable role-contract identity for cache keys."""

        import hashlib

        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()


DEFAULT_A01_SCARED_ROLES = A01ScaredDataRoles()


def load_a01_roles(path: Path) -> A01ScaredDataRoles:
    """Load and strictly validate the active A01 YAML role contract."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise A01RoleError(f"unable to read A01 role config {path}: {error}") from error
    except yaml.YAMLError as error:
        raise A01RoleError(f"invalid YAML in A01 role config {path}: {error}") from error
    mapping = _as_mapping(raw, "A01 role configuration")
    required = {
        "schema_version",
        "dataset_name",
        "root_env_var",
        "root_subdir",
        "dataset_ids",
        "roles",
        "cache_roles",
        "final_roles",
        "semantics",
    }
    if set(mapping) != required:
        raise A01RoleError(f"A01 role configuration keys must be exactly {sorted(required)}")
    if mapping["schema_version"] != 1:
        raise A01RoleError("unsupported A01 role schema version")
    dataset_ids = _normal_ids(mapping["dataset_ids"], "dataset_ids")
    if dataset_ids != A01_DATASET_IDS:
        raise A01RoleError("A01 config must enumerate dataset_1 through dataset_7")
    roles_mapping = _as_mapping(mapping["roles"], "roles")
    roles = {role: _normal_ids(values, f"roles.{role}") for role, values in roles_mapping.items()}
    cache_roles_raw = mapping["cache_roles"]
    final_roles_raw = mapping["final_roles"]
    if not isinstance(cache_roles_raw, Sequence) or isinstance(cache_roles_raw, (str, bytes)):
        raise A01RoleError("cache_roles must be a sequence")
    if not isinstance(final_roles_raw, Sequence) or isinstance(final_roles_raw, (str, bytes)):
        raise A01RoleError("final_roles must be a sequence")
    semantics = _as_mapping(mapping["semantics"], "semantics")
    if set(semantics) != {"teacher_fit", "qualification"}:
        raise A01RoleError("semantics must document both active A01 roles")
    return A01ScaredDataRoles(
        dataset_name=str(mapping["dataset_name"]),
        root_env_var=str(mapping["root_env_var"]),
        root_subdir=str(mapping["root_subdir"]),
        dataset_ids=dataset_ids,
        roles=roles,
        cache_roles=tuple(str(value) for value in cache_roles_raw),
        final_roles=tuple(str(value) for value in final_roles_raw),
    )


__all__ = [
    "A01_CACHE_ROLES",
    "A01_DATASET_IDS",
    "A01_DATASET_NAME",
    "A01_FINAL_DATASET_IDS",
    "A01RoleError",
    "A01ScaredDataRoles",
    "DEFAULT_A01_SCARED_ROLES",
    "load_a01_roles",
]
