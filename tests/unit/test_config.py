"""Tests for typed YAML configuration loading."""

from pathlib import Path

import pytest
import yaml

from reliable_endo_gs.config.loader import ConfigError, load_config


def _write_yaml(path: Path, data: object) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_valid_yaml_loads(tmp_path: Path, valid_config_data: dict[str, object]) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, valid_config_data)

    config = load_config(config_path)

    assert config.experiment.name == "infrastructure_smoke"
    assert config.experiment.seed == 42
    assert config.experiment.output_root == Path("outputs")
    assert config.runtime.device == "cpu"


def test_missing_required_key_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.yaml"
    _write_yaml(
        config_path,
        {
            "experiment": {"name": "smoke", "output_root": "outputs"},
            "runtime": {"device": "cpu"},
        },
    )

    with pytest.raises(ConfigError, match="missing required field.*seed"):
        load_config(config_path)


def test_invalid_type_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    _write_yaml(
        config_path,
        {
            "experiment": {"name": "smoke", "seed": "42", "output_root": "outputs"},
            "runtime": {"device": "cpu"},
        },
    )

    with pytest.raises(ConfigError, match="experiment.seed must be an integer"):
        load_config(config_path)
