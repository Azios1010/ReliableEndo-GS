"""Tests for strict, portable dataset configuration."""

from pathlib import Path

import pytest
import yaml

from reliable_endo_gs.data.schema import (
    DatasetConfigError,
    hash_dataset_config,
    load_dataset_config,
)


def _write_yaml(path: Path, data: object) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_valid_dataset_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "scared.yaml"
    _write_yaml(
        path,
        {"name": "scared", "root": "SCARED", "version": None, "split_manifest": None},
    )

    config = load_dataset_config(path)

    assert config.name == "scared"
    assert config.root == Path("SCARED")
    assert config.version is None
    assert config.split_manifest is None
    assert dict(config.options) == {}


def test_missing_name_fails(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"
    _write_yaml(path, {"root": "SCARED", "version": None, "split_manifest": None})

    with pytest.raises(DatasetConfigError, match="missing required field.*name"):
        load_dataset_config(path)


def test_invalid_root_type_fails(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    _write_yaml(
        path,
        {"name": "scared", "root": ["SCARED"], "version": None, "split_manifest": None},
    )

    with pytest.raises(DatasetConfigError, match="dataset.root must be a non-empty string"):
        load_dataset_config(path)


def test_optional_version_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "versioned.yaml"
    _write_yaml(
        path,
        {
            "name": "c3vd",
            "root": "C3VD",
            "version": "approved-release-label",
            "split_manifest": None,
        },
    )

    assert load_dataset_config(path).version == "approved-release-label"


def test_dataset_config_hash_is_stable_across_yaml_key_order(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    _write_yaml(
        first,
        {"name": "scared", "root": "SCARED", "version": None, "split_manifest": None},
    )
    _write_yaml(
        second,
        {"split_manifest": None, "version": None, "root": "SCARED", "name": "scared"},
    )

    assert hash_dataset_config(load_dataset_config(first)) == hash_dataset_config(
        load_dataset_config(second)
    )
