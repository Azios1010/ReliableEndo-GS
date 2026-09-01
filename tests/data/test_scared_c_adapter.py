"""SCARED-C adapter semantics and validation tests."""

from pathlib import Path

import pytest

from reliable_endo_gs.data.adapters.scared import ScaredAdapter
from reliable_endo_gs.data.adapters.scared_c import ScaredCAdapter


def test_scared_c_adapter_has_no_filesystem_access_on_construction() -> None:
    adapter = ScaredCAdapter()

    assert adapter.name == "scared_c"
    assert adapter.protocol == "scared_c_endoscope_stereo_calibration_v1"


def test_scared_c_adapter_reports_missing_root(tmp_path: Path) -> None:
    report = ScaredCAdapter().validate_root(tmp_path / "missing")

    assert report.status == "MISSING"
    assert report.valid is False
    assert report.to_dict()["root"] == "<REDACTED_ABSOLUTE_PATH>"


def test_scared_c_adapter_reports_incomplete_keyframe(tmp_path: Path, make_keyframe) -> None:
    make_keyframe(tmp_path, missing=("right_depth_map.tiff",))

    report = ScaredCAdapter().validate_root(tmp_path)

    assert report.status == "INCOMPLETE"
    assert report.valid is False
    assert "right_depth_map.tiff" in " ".join(report.messages)


def test_scared_c_adapter_reports_invalid_frame_log(tmp_path: Path, make_keyframe) -> None:
    keyframe = make_keyframe(tmp_path)
    (keyframe / "frame_log.json").write_text("{invalid", encoding="utf-8")

    report = ScaredCAdapter().validate_root(tmp_path)

    assert report.status == "INVALID"
    assert report.valid is False


def test_scared_c_adapter_enumerates_and_inspects_metadata(scared_c_root: Path) -> None:
    adapter = ScaredCAdapter()

    assert adapter.enumerate_sequences(scared_c_root) == ("dataset_1",)
    metadata = adapter.inspect_metadata(scared_c_root, "dataset_1")
    assert metadata["protocol"] == adapter.protocol
    assert metadata["keyframes"] == [
        {
            "keyframe_id": "keyframe_1",
            "frame_log_key": "1_1",
            "total_frames_on_disk": 2,
            "included_count": 2,
            "has_frame_data_archive": False,
        }
    ]


def test_scared_c_adapter_validates_real_contract_fixture(scared_c_root: Path) -> None:
    report = ScaredCAdapter().validate_root(scared_c_root)

    assert report.status == "USABLE"
    assert report.valid is True
    assert report.contract_valid is True
    assert report.stereo_available is True
    assert report.sequence_count == 1
    assert report.sample_count == 1
    assert report.index_hash is not None


def test_original_scared_adapter_is_not_an_alias() -> None:
    original = ScaredAdapter()
    corrected = ScaredCAdapter()

    assert original.name == "scared"
    assert corrected.name == "scared_c"
    with pytest.raises(NotImplementedError):
        original.enumerate_sequences(Path("unused"))
