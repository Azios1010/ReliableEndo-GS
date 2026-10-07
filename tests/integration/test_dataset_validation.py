"""Integration tests for adapter-level external root validation."""

from pathlib import Path

from reliable_endo_gs.data.adapters.scared import ScaredAdapter


def test_missing_root_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "missing"

    report = ScaredAdapter().validate_root(root)

    assert report.exists is False
    assert report.is_directory is False
    assert report.valid is False
    assert report.messages == ("STATUS: MISSING", "Configured SCARED root does not exist.")


def test_existing_directory_requires_a00_processed_layout(tmp_path: Path) -> None:
    root = tmp_path / "SCARED"
    root.mkdir()

    report = ScaredAdapter().validate_root(root)

    assert report.exists is True
    assert report.is_directory is True
    assert report.valid is False
    assert report.layout_checked is True
    assert report.status == "INVALID"
    assert "found 0/9 active folds and 0 frame records" in report.messages
    assert "no processed frame_data/*.json samples found" in report.messages


def test_existing_file_is_invalid(tmp_path: Path) -> None:
    root = tmp_path / "not-a-directory"
    root.write_text("not dataset content", encoding="utf-8")

    report = ScaredAdapter().validate_root(root)

    assert report.exists is True
    assert report.is_directory is False
    assert report.valid is False
    assert report.messages == ("STATUS: INVALID", "Configured SCARED root is not a directory.")
