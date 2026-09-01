"""Deterministic SCARED-C index tests."""

from pathlib import Path

import pytest

from reliable_endo_gs.data.index import (
    IndexError,
    build_scared_c_index,
    build_scared_index,
    make_sample_id,
)


def test_scared_c_index_is_sorted_and_path_independent(
    scared_c_root: Path, make_keyframe, tmp_path: Path
) -> None:
    first = build_scared_c_index(scared_c_root)
    single_hash = first.index_hash
    make_keyframe(scared_c_root, dataset_name="dataset_10", keyframe_name="keyframe_2")
    first = build_scared_c_index(scared_c_root)
    second = build_scared_c_index(scared_c_root)

    assert first.sample_ids == (
        "scared_c/dataset_1/keyframe_1/reference",
        "scared_c/dataset_10/keyframe_2/reference",
    )
    assert first.index_hash == second.index_hash
    assert all(str(scared_c_root) not in sample_id for sample_id in first.sample_ids)

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    make_keyframe(relocated, dataset_name="dataset_1", keyframe_name="keyframe_1")
    relocated_index = build_scared_c_index(relocated)
    assert relocated_index.sample_ids == ("scared_c/dataset_1/keyframe_1/reference",)
    assert relocated_index.index_hash == single_hash


def test_scared_c_index_reports_missing_members(scared_c_root: Path) -> None:
    keyframe = scared_c_root / "dataset_1" / "keyframe_1"
    (keyframe / "Right_Image.png").unlink()

    index = build_scared_c_index(scared_c_root, strict=False)

    assert index.records == ()
    assert "missing member(s): Right_Image.png" in index.issues[0]
    with pytest.raises(IndexError, match="Right_Image.png"):
        build_scared_c_index(scared_c_root)


def test_scared_c_index_reports_stereo_dimension_mismatch(scared_c_root: Path) -> None:
    from PIL import Image

    Image.new("RGBA", (2, 2), color=(0, 0, 0, 255)).save(
        scared_c_root / "dataset_1" / "keyframe_1" / "Right_Image.png"
    )

    index = build_scared_c_index(scared_c_root, strict=False)

    assert any("dimensions do not match" in issue for issue in index.issues)


def test_logical_id_rejects_absolute_or_nested_components() -> None:
    assert make_sample_id("scared_c", "dataset_1", "keyframe_1") == (
        "scared_c/dataset_1/keyframe_1/reference"
    )
    with pytest.raises(ValueError):
        make_sample_id("scared_c", "E:\\private", "keyframe_1")


def test_legacy_index_remains_separate_from_scared_c_layout(tmp_path: Path) -> None:
    legacy = tmp_path / "dataset_01" / "keyframe_01" / "data"
    (legacy / "left").mkdir(parents=True)
    (legacy / "right").mkdir()
    (legacy / "depth").mkdir()
    (legacy / "left" / "000000.png").touch()
    (legacy / "right" / "000000.png").touch()
    (legacy / "depth" / "000000.tiff").touch()

    samples = build_scared_index(tmp_path)

    assert len(samples) == 1
    assert samples[0].logical_id == "dataset_01_keyframe_01_000000"
