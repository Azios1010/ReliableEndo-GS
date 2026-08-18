"""Versioned split manifests, leakage validation, and scientific split hashing."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

SPLIT_SCHEMA_VERSION = 1


class SplitManifestError(ValueError):
    """Raised when a split manifest is malformed or leaks identifiers."""


@dataclass(frozen=True)
class SplitManifest:
    """Versioned train/validation/test sequence assignment."""

    schema_version: int
    dataset: str
    dataset_version: str | None
    split_name: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    notes: str


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SplitManifestError("split manifest must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise SplitManifestError("split manifest keys must be strings")
    return cast(dict[str, object], value)


def _required_string(mapping: dict[str, object], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise SplitManifestError(f"split manifest field {key!r} must be a non-empty string")
    return value.strip()


def _parse_ids(mapping: dict[str, object], key: str) -> tuple[str, ...]:
    raw_ids = mapping[key]
    if not isinstance(raw_ids, list):
        raise SplitManifestError(f"split manifest field {key!r} must be a list")
    if not all(isinstance(item, str) and item.strip() for item in raw_ids):
        raise SplitManifestError(f"all identifiers in {key!r} must be non-empty strings")

    identifiers = tuple(cast(str, item).strip() for item in raw_ids)
    duplicates = sorted(
        {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
    )
    if duplicates:
        raise SplitManifestError(f"duplicate identifier(s) in {key}: {', '.join(duplicates)}")
    return identifiers


def validate_split_manifest(manifest: SplitManifest) -> None:
    """Validate schema support, within-split uniqueness, and cross-split isolation."""

    if manifest.schema_version != SPLIT_SCHEMA_VERSION:
        raise SplitManifestError(
            f"unsupported split schema version {manifest.schema_version}; "
            f"supported version is {SPLIT_SCHEMA_VERSION}"
        )
    if not manifest.dataset.strip():
        raise SplitManifestError("dataset name must be present")
    if not manifest.split_name.strip():
        raise SplitManifestError("split name must be present")

    assignments: dict[str, list[str]] = {}
    for split_name, identifiers in (
        ("train", manifest.train),
        ("validation", manifest.validation),
        ("test", manifest.test),
    ):
        if len(identifiers) != len(set(identifiers)):
            raise SplitManifestError(f"duplicate identifier detected inside {split_name}")
        for identifier in identifiers:
            assignments.setdefault(identifier, []).append(split_name)

    overlaps = {
        identifier: locations for identifier, locations in assignments.items() if len(locations) > 1
    }
    if overlaps:
        details = "; ".join(
            f"{identifier} appears in {', '.join(locations)}"
            for identifier, locations in sorted(overlaps.items())
        )
        raise SplitManifestError(f"split overlap detected: {details}")


def parse_split_manifest(value: object) -> SplitManifest:
    """Parse and validate a split manifest from decoded JSON."""

    mapping = _as_mapping(value)
    required = {
        "schema_version",
        "dataset",
        "dataset_version",
        "split_name",
        "train",
        "validation",
        "test",
        "notes",
    }
    missing = sorted(required - mapping.keys())
    if missing:
        raise SplitManifestError(f"split manifest missing field(s): {', '.join(missing)}")
    unexpected = sorted(mapping.keys() - required)
    if unexpected:
        raise SplitManifestError(
            f"split manifest has unsupported field(s): {', '.join(unexpected)}"
        )

    schema_version = mapping["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise SplitManifestError("schema_version must be an integer")

    dataset_version_value = mapping["dataset_version"]
    if dataset_version_value is not None and (
        not isinstance(dataset_version_value, str) or not dataset_version_value.strip()
    ):
        raise SplitManifestError("dataset_version must be null or a non-empty string")

    notes = mapping["notes"]
    if not isinstance(notes, str):
        raise SplitManifestError("notes must be a string")

    manifest = SplitManifest(
        schema_version=schema_version,
        dataset=_required_string(mapping, "dataset"),
        dataset_version=(
            dataset_version_value.strip() if isinstance(dataset_version_value, str) else None
        ),
        split_name=_required_string(mapping, "split_name"),
        train=_parse_ids(mapping, "train"),
        validation=_parse_ids(mapping, "validation"),
        test=_parse_ids(mapping, "test"),
        notes=notes,
    )
    validate_split_manifest(manifest)
    return manifest


def load_split_manifest(path: Path) -> SplitManifest:
    """Load and validate a committed JSON split manifest."""

    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SplitManifestError(f"Unable to read split manifest {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise SplitManifestError(f"Invalid JSON in split manifest {path}: {error}") from error
    return parse_split_manifest(decoded)


def hash_split_manifest(manifest: SplitManifest) -> str:
    """Hash scientific split identity with order-insensitive ID lists.

    The split name, notes, timestamp, file location, and developer paths are
    intentionally excluded.
    """

    validate_split_manifest(manifest)
    identity = {
        "schema_version": manifest.schema_version,
        "dataset": manifest.dataset,
        "dataset_version": manifest.dataset_version,
        "train": sorted(manifest.train),
        "validation": sorted(manifest.validation),
        "test": sorted(manifest.test),
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
