"""Tests for canonical configuration hashing."""

from pathlib import Path

import yaml

from reliable_endo_gs.config.hashing import hash_config
from reliable_endo_gs.config.loader import load_config
from reliable_endo_gs.config.schema import ExperimentConfig, ProjectConfig, RuntimeConfig


def test_same_config_has_same_hash() -> None:
    config = ProjectConfig(
        experiment=ExperimentConfig("smoke", 42, Path("outputs")),
        runtime=RuntimeConfig("cpu"),
    )

    assert hash_config(config) == hash_config(config)


def test_reordered_yaml_mapping_has_same_hash(tmp_path: Path) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "smoke", "seed": 42, "output_root": "outputs"},
                "runtime": {"device": "cpu"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    second_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {"device": "cpu"},
                "experiment": {"output_root": "outputs", "seed": 42, "name": "smoke"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert hash_config(load_config(first_path)) == hash_config(load_config(second_path))


def test_changed_seed_changes_hash() -> None:
    first = ProjectConfig(
        experiment=ExperimentConfig("smoke", 42, Path("outputs")),
        runtime=RuntimeConfig("cpu"),
    )
    second = ProjectConfig(
        experiment=ExperimentConfig("smoke", 43, Path("outputs")),
        runtime=RuntimeConfig("cpu"),
    )

    assert hash_config(first) != hash_config(second)
