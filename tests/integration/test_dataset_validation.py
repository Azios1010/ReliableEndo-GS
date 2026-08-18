"""Integration tests for adapter-level external root validation."""

from pathlib import Path

from reliable_endo_gs.data.adapters.scared import ScaredAdapter


def test_missing_root_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "missing"

    report = ScaredAdapter().validate_root(root)

    assert report.exists is False
    assert report.is_directory is False
    assert report.valid is False
    assert report.messages == ("DATASET NOT PRESENT: configured root does not exist.",)


def test_existing_directory_passes_only_basic_validation(tmp_path: Path) -> None:
    root = tmp_path / "SCARED"
    root.mkdir()

    report = ScaredAdapter().validate_root(root)

    assert report.exists is True
    assert report.is_directory is True
    assert report.valid is True
    assert report.layout_checked is False
    assert "DATASET PRESENT, BASIC VALIDATION PASSED." in report.messages
    assert "DETAILED LAYOUT VALIDATION NOT IMPLEMENTED YET." in report.messages


def test_existing_file_is_invalid(tmp_path: Path) -> None:
    root = tmp_path / "not-a-directory"
    root.write_text("not dataset content", encoding="utf-8")

    report = ScaredAdapter().validate_root(root)

    assert report.exists is True
    assert report.is_directory is False
    assert report.valid is False
    assert report.messages == ("DATASET PRESENT BUT INVALID: configured root is not a directory.",)
