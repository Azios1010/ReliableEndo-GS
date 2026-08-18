"""Minimal, infrastructure-only project configuration schema."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for identifying and storing one infrastructure run."""

    name: str
    seed: int
    output_root: Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for the execution environment."""

    device: str


@dataclass(frozen=True)
class ProjectConfig:
    """Complete configuration currently supported by the project."""

    experiment: ExperimentConfig
    runtime: RuntimeConfig


def config_to_dict(config: ProjectConfig) -> dict[str, object]:
    """Convert a resolved config to a JSON/YAML-compatible representation."""

    return {
        "experiment": {
            "name": config.experiment.name,
            "seed": config.experiment.seed,
            "output_root": config.experiment.output_root.as_posix(),
        },
        "runtime": {"device": config.runtime.device},
    }
