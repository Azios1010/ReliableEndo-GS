"""Typed dataset identity and portable configuration schema."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias, cast

import yaml

DatasetOptionValue: TypeAlias = str | int | float | bool | None


class DatasetConfigError(ValueError):
    """Raised when dataset identity configuration is missing or invalid."""


def _empty_options() -> Mapping[str, DatasetOptionValue]:
    return MappingProxyType({})


@dataclass(frozen=True)
class DatasetConfig:
    """Portable identity and external-root configuration for one dataset."""

    name: str
    root: Path
    version: str | None
    split_manifest: Path | None
    options: Mapping[str, DatasetOptionValue] = field(default_factory=_empty_options)


def _as_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise DatasetConfigError(f"{context} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise DatasetConfigError(f"{context} keys must be strings")
    return cast(dict[str, object], value)


def _required_string(mapping: dict[str, object], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise DatasetConfigError(f"dataset.{key} must be a non-empty string")
    return value.strip()


def _optional_string(mapping: dict[str, object], key: str) -> str | None:
    value = mapping[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DatasetConfigError(f"dataset.{key} must be null or a non-empty string")
    return value.strip()


def _parse_options(value: object) -> Mapping[str, DatasetOptionValue]:
    raw_options = _as_mapping(value, "dataset.options")
    parsed: dict[str, DatasetOptionValue] = {}
    for key, option_value in raw_options.items():
        if not isinstance(option_value, (str, int, float, bool)) and option_value is not None:
            raise DatasetConfigError(
                f"dataset.options.{key} must be a scalar value; nested structures are unsupported"
            )
        parsed[key] = option_value
    return MappingProxyType(parsed)


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load a strict dataset YAML without resolving or inspecting its root."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DatasetConfigError(f"Unable to read dataset config {path}: {error}") from error
    except yaml.YAMLError as error:
        raise DatasetConfigError(f"Invalid YAML in dataset config {path}: {error}") from error

    mapping = _as_mapping(raw, "dataset configuration")
    required = {"name", "root", "version", "split_manifest"}
    allowed = required | {"options"}
    missing = sorted(required - mapping.keys())
    if missing:
        raise DatasetConfigError(
            f"dataset configuration is missing required field(s): {', '.join(missing)}"
        )
    unexpected = sorted(mapping.keys() - allowed)
    if unexpected:
        raise DatasetConfigError(
            f"dataset configuration contains unsupported field(s): {', '.join(unexpected)}"
        )

    root_value = _required_string(mapping, "root")
    split_value = _optional_string(mapping, "split_manifest")
    options = _parse_options(mapping.get("options", {}))
    return DatasetConfig(
        name=_required_string(mapping, "name"),
        root=Path(root_value),
        version=_optional_string(mapping, "version"),
        split_manifest=Path(split_value) if split_value is not None else None,
        options=options,
    )


def dataset_config_to_dict(config: DatasetConfig) -> dict[str, object]:
    """Return the portable dataset configuration used for identity hashing."""

    return {
        "name": config.name,
        "root": config.root.as_posix(),
        "version": config.version,
        "split_manifest": (
            config.split_manifest.as_posix() if config.split_manifest is not None else None
        ),
        "options": dict(config.options),
    }


def hash_dataset_config(config: DatasetConfig) -> str:
    """Hash dataset configuration, not mounted dataset contents."""

    canonical = json.dumps(
        dataset_config_to_dict(config),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
