"""Synthetic contract tests for the audited lazy SCARED-C loader."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from reliable_endo_gs.data.scared_c import (
    CORRECTED_VIDEO_MODE,
    FINAL_DATASET_ID,
    STATIC_MODE,
    FinalDatasetGuardError,
    InactiveSplitError,
    ScaredCLazyDataset,
    build_scared_c_index,
    load_scared_c_role_manifest,
    parse_frame_log,
    resolve_scared_c_root,
)
from reliable_endo_gs.data.scared_c_stereo import (
    StereoCalibrationContract,
    StereoContractError,
    archive_member_map,
    create_rectification_maps,
    direct_projected_disparity,
    index_archive_members,
    project_xyz,
    rasterize_nearest_depth,
    reproject_with_q,
    split_stacked_stereo_frame,
    transform_left_to_right,
    valid_xyz_mask,
    validate_archive_frame_identity,
)


def _write_tar(path: Path, members: list[str]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for name in members:
            payload = b"{}" if name.endswith(".json") else b"synthetic"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _analytic_calibration() -> StereoCalibrationContract:
    M1 = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]])
    M2 = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]])
    return StereoCalibrationContract(
        M1=M1,
        D1=np.zeros((1, 5)),
        M2=M2,
        D2=np.zeros((1, 5)),
        R=np.eye(3),
        T=np.array([[-2.0], [0.0], [0.0]]),
        K_colmap=M1.copy(),
        image_size=(8, 6),
    )


def test_stacked_video_split_and_frame_index_contract() -> None:
    frame = np.zeros((2048, 1280, 3), dtype=np.uint8)
    frame[:1024] = 17
    frame[1024:] = 231

    left, right = split_stacked_stereo_frame(frame)

    assert left.shape == (1024, 1280, 3)
    assert right.shape == (1024, 1280, 3)
    assert np.all(left == 17)
    assert np.all(right == 231)
    assert frame[0, 0, 0] == 17
    assert frame[1024, 0, 0] == 231


def test_frame_log_and_archive_identity_support_noncontiguous_ids(tmp_path: Path) -> None:
    frame_log_path = tmp_path / "frame_log.json"
    frame_log_path.write_text(
        json.dumps(
            {
                "key": "2_1",
                "total_frames_on_disk": 88,
                "included_frames": [60, 45, 46],
                "excluded_frames": [47],
            }
        ),
        encoding="utf-8",
    )
    parsed = parse_frame_log(
        frame_log_path,
        sequence_id="dataset_2",
        keyframe_id="keyframe_1",
    )
    assert parsed.included_frames == (45, 46, 60)
    assert parsed.included_frames[0] - 1 == 44

    archives = {
        "frame_data": tmp_path / "frame_data.tar.gz",
        "rgb_frames": tmp_path / "rgb_frames.tar.gz",
        "scene_points": tmp_path / "scene_points.tar.gz",
    }
    _write_tar(archives["frame_data"], ["frame_data000060.json", "frame_data000045.json"])
    _write_tar(archives["rgb_frames"], ["frame000045.png", "frame000060.png"])
    _write_tar(archives["scene_points"], ["scene_points000060.tiff", "scene_points000045.tiff"])
    indexed = {
        kind: index_archive_members(path, kind=kind)  # type: ignore[arg-type]
        for kind, path in archives.items()
    }
    assert tuple(item.frame_id for item in indexed["rgb_frames"]) == (45, 60)
    common, issues = validate_archive_frame_identity(set(parsed.included_frames), indexed)
    assert common == (45, 60)
    assert issues == ("frame 46 missing from frame_data, rgb_frames, scene_points",)
    assert archive_member_map(indexed["frame_data"])[45] == "frame_data000045.json"


def test_archive_duplicate_frame_id_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "rgb_frames.tar.gz"
    _write_tar(archive_path, ["frame000001.png", "nested/frame000001.png"])

    with pytest.raises(StereoContractError, match="duplicate rgb_frames frame ID"):
        index_archive_members(archive_path, kind="rgb_frames")


def test_geometry_direction_zbuffer_and_direct_disparity() -> None:
    calibration = _analytic_calibration()
    rectification = create_rectification_maps(calibration)
    xyz = np.zeros((6, 8, 3), dtype=np.float32)
    yy, xx = np.indices((6, 8), dtype=np.float32)
    xyz[..., 0] = (xx - 2.0) * 100.0 / 100.0
    xyz[..., 1] = (yy - 2.0) * 100.0 / 100.0
    xyz[..., 2] = 100.0

    transformed = transform_left_to_right(
        np.array([[[1.0, 2.0, 10.0]]]), calibration.R, calibration.T
    )
    assert np.allclose(transformed, [[[-1.0, 2.0, 10.0]]])
    direct = direct_projected_disparity(xyz, calibration, rectification)
    simplified = rectification.fx_rect * calibration.baseline / direct.left_rectified_xyz[..., 2]
    assert (
        np.max(np.abs(direct.disparity[direct.valid_mask] - simplified[direct.valid_mask])) < 1e-5
    )
    assert np.all(direct.disparity[direct.valid_mask] > 0)
    left_projection = project_xyz(
        direct.left_rectified_xyz,
        np.concatenate((rectification.P1[:, :3], np.zeros((3, 1))), axis=1),
    )
    q_xyz = reproject_with_q(
        left_projection.u[direct.valid_mask],
        left_projection.v[direct.valid_mask],
        direct.disparity[direct.valid_mask],
        rectification.Q,
    )
    assert np.max(np.abs(q_xyz - direct.left_rectified_xyz[direct.valid_mask])) < 1e-5

    collision_xyz = np.array([[[6.0, 0.0, 5.0], [4.2, 0.0, 3.0]]], dtype=np.float32)
    projection = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    raster = rasterize_nearest_depth(
        collision_xyz,
        projection,
        output_size=(4, 2),
    )
    assert raster.valid_mask[0, 1]
    assert raster.depth[0, 1] == pytest.approx(3.0)


def test_xyz_validity_uses_finite_coordinates_and_positive_z() -> None:
    xyz = np.array(
        [
            [[1.0, 2.0, 3.0], [np.nan, 2.0, 3.0]],
            [[1.0, 2.0, 0.0], [1.0, 2.0, -1.0]],
        ],
        dtype=np.float32,
    )

    assert np.array_equal(valid_xyz_mask(xyz), [[True, False], [False, False]])


def test_static_mode_does_not_require_temporal_assets(tmp_path: Path, make_keyframe) -> None:
    keyframe = make_keyframe(tmp_path)
    (keyframe / "frame_log.json").unlink()
    (keyframe / "intrinsics_colmap.yaml").unlink()

    index = build_scared_c_index(tmp_path, mode=STATIC_MODE)

    assert len(index.records) == 1
    assert index.records[0].source_type == "STATIC_KEYFRAME"
    assert index.records[0].frame_id == "reference"
    with ScaredCLazyDataset(tmp_path, mode=STATIC_MODE) as dataset:
        sample = dataset[0]
        assert sample["source_type"] == "STATIC_KEYFRAME"
        assert sample["K_colmap"] is None


def test_final_dataset_guard_precedes_static_content_decode(tmp_path: Path, make_keyframe) -> None:
    make_keyframe(tmp_path, dataset_name=FINAL_DATASET_ID)
    dataset = ScaredCLazyDataset(tmp_path, mode=STATIC_MODE)

    assert len(dataset) == 1
    assert dataset.metadata(0).final_role is True
    with pytest.raises(FinalDatasetGuardError):
        _ = dataset[0]
    dataset.close()


def test_inactive_manifest_and_root_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = load_scared_c_role_manifest()
    assert manifest.active is False
    assert manifest.roles["MEAN_MODEL_TRAIN"] == ("1_1", "2_2", "3_2")
    with pytest.raises(InactiveSplitError):
        manifest.require_active_role("MEAN_MODEL_TRAIN")
    monkeypatch.setenv("RELIABLE_ENDO_DATA_ROOT", r"E:\mounted-data")
    assert resolve_scared_c_root(config_root="scared_c") == Path(r"E:\mounted-data\scared_c")


def test_corrected_mode_name_is_explicit() -> None:
    assert CORRECTED_VIDEO_MODE == "corrected_video"
