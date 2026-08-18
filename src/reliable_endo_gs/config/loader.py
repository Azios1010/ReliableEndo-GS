"""Load and validate the small YAML configuration foundation."""

from pathlib import Path
from typing import cast

import yaml

from reliable_endo_gs.config.schema import ExperimentConfig, ProjectConfig, RuntimeConfig


class ConfigError(ValueError):
    """Raised when a project configuration is missing or invalid."""


def _as_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{context} keys must be strings")
    return cast(dict[str, object], value)


def _validate_keys(mapping: dict[str, object], required: set[str], context: str) -> None:
    missing = sorted(required - mapping.keys())
    if missing:
        raise ConfigError(f"{context} is missing required field(s): {', '.join(missing)}")

    unexpected = sorted(mapping.keys() - required)
    if unexpected:
        raise ConfigError(f"{context} contains unsupported field(s): {', '.join(unexpected)}")


def _required_string(mapping: dict[str, object], key: str, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context}.{key} must be a non-empty string")
    return value


def _required_seed(mapping: dict[str, object], key: str, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context}.{key} must be an integer")
    if value < 0:
        raise ConfigError(f"{context}.{key} must be non-negative")
    return value


def load_config(path: Path) -> ProjectConfig:
    """Load a validated project configuration from ``path``.

    Relative output paths remain relative so resolved configurations do not
    acquire developer-specific absolute paths.
    """

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"Unable to read configuration {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in configuration {path}: {error}") from error

    root = _as_mapping(raw, "configuration")
    _validate_keys(root, {"experiment", "runtime"}, "configuration")

    experiment_raw = _as_mapping(root["experiment"], "experiment")
    _validate_keys(experiment_raw, {"name", "seed", "output_root"}, "experiment")

    runtime_raw = _as_mapping(root["runtime"], "runtime")
    _validate_keys(runtime_raw, {"device"}, "runtime")

    output_root = Path(_required_string(experiment_raw, "output_root", "experiment"))
    return ProjectConfig(
        experiment=ExperimentConfig(
            name=_required_string(experiment_raw, "name", "experiment"),
            seed=_required_seed(experiment_raw, "seed", "experiment"),
            output_root=output_root,
        ),
        runtime=RuntimeConfig(device=_required_string(runtime_raw, "device", "runtime")),
    )
