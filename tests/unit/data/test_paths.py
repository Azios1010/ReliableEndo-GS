"""Tests for explicit external dataset root resolution."""

import importlib
from pathlib import Path

import pytest

import reliable_endo_gs.data as data_module
from reliable_endo_gs.data.paths import DataRootResolutionError, resolve_data_root


def test_relative_root_uses_environment_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RELIABLE_ENDO_DATA_ROOT", str(tmp_path))

    assert resolve_data_root(Path("SCARED")) == tmp_path / "SCARED"


def test_explicit_global_root_takes_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment_root = tmp_path / "environment"
    explicit_root = tmp_path / "explicit"
    monkeypatch.setenv("RELIABLE_ENDO_DATA_ROOT", str(environment_root))

    assert resolve_data_root(Path("C3VD"), global_root=explicit_root) == explicit_root / "C3VD"


def test_absolute_dataset_root_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RELIABLE_ENDO_DATA_ROOT", raising=False)
    absolute_root = tmp_path / "EndoNeRF"

    assert resolve_data_root(absolute_root) == absolute_root


def test_missing_environment_root_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RELIABLE_ENDO_DATA_ROOT", raising=False)

    with pytest.raises(DataRootResolutionError, match="RELIABLE_ENDO_DATA_ROOT"):
        resolve_data_root(Path("SCARED"))


def test_import_does_not_inspect_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(path: Path) -> bool:
        raise AssertionError(f"filesystem inspected during import: {path}")

    monkeypatch.setattr(Path, "exists", fail_if_called)
    importlib.reload(data_module)
