"""Mounted non-final SCARED-C contract checks.

The test intentionally materializes only the four approved representatives
``1_1``, ``2_2``, ``3_1``, and ``7_2``.  Dataset 6 is inspected only through
lazy metadata and guard checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reliable_endo_gs.data.scared_c import (
    CORRECTED_VIDEO_MODE,
    STATIC_MODE,
    FinalDatasetGuardError,
    ScaredCLazyDataset,
)
from reliable_endo_gs.data.scared_c_stereo import (
    RectificationMaps,
    StereoCalibrationContract,
    direct_projected_disparity,
    project_xyz,
    read_archive_rgb,
)


def _mounted_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "scared_c"


def _sample_rectification(sample: dict[str, object]) -> RectificationMaps:
    empty_map = np.empty((1024, 1280), dtype=np.float32)
    return RectificationMaps(
        image_size=(1280, 1024),
        R1=sample["R1"],  # type: ignore[arg-type]
        R2=sample["R2"],  # type: ignore[arg-type]
        P1=sample["P1"],  # type: ignore[arg-type]
        P2=sample["P2"],  # type: ignore[arg-type]
        Q=sample["Q"],  # type: ignore[arg-type]
        left_map_x=empty_map,
        left_map_y=empty_map,
        right_map_x=empty_map,
        right_map_y=empty_map,
    )


def _sample_calibration(sample: dict[str, object]) -> StereoCalibrationContract:
    return StereoCalibrationContract(
        M1=sample["M1"],  # type: ignore[arg-type]
        D1=sample["D1"],  # type: ignore[arg-type]
        M2=sample["M2"],  # type: ignore[arg-type]
        D2=sample["D2"],  # type: ignore[arg-type]
        R=sample["R"],  # type: ignore[arg-type]
        T=sample["T"],  # type: ignore[arg-type]
        K_colmap=sample["K_colmap"],  # type: ignore[arg-type]
        image_size=(1280, 1024),
    )


@pytest.mark.skipif(not _mounted_root().is_dir(), reason="mounted SCARED-C root is unavailable")
def test_real_scared_c_contract_on_non_final_representatives() -> None:
    root = _mounted_root()
    representatives = ("1_1", "2_2", "3_1", "7_2")

    with ScaredCLazyDataset(
        root,
        mode=CORRECTED_VIDEO_MODE,
        archive_validation="lazy",
    ) as dataset:
        assert len(dataset) == 17129
        assert len(dataset.index.final_records) == 4657
        final_record = next(
            record for record in dataset.index.records if record.sequence_key == "6_1"
        )
        with pytest.raises(FinalDatasetGuardError):
            _ = dataset[final_record.sample_id]

        for sequence_key in representatives:
            record = next(
                item
                for item in dataset.index.records
                if item.sequence_key == sequence_key and item.frame_id == 1
            )
            sample = dataset[record.sample_id]
            assert sample["source_type"] == "CORRECTED_VIDEO_FRAME"
            assert sample["frame_id"] == 1
            assert sample["video_index"] == 0
            assert sample["camera_pose_direction"] == "WORLD_TO_CAMERA_RELATIVE_KEYFRAME_LEFT"
            assert sample["pose_reference"] == "KEYFRAME_LEFT"
            assert sample["camera_pose_raw"].shape == (4, 4)
            assert sample["geometry_unit"] == "MILLIMETRES"
            assert sample["upstream_revision"] == ("44baac1187c8729c96db0d1def569bd94c9d9417")
            assert np.array_equal(
                sample["rgb_left_raw"],
                read_archive_rgb(record.rgb_frames_archive_path, record.rgb_frames_member),
            )

            xyz = sample["xyz_left_corrected"]
            valid_xyz = sample["valid_xyz_mask"]
            K = sample["K_colmap"]
            assert K is not None
            with np.errstate(divide="ignore", invalid="ignore"):
                z = xyz[..., 2].astype(np.float64)
                u = K[0, 0] * xyz[..., 0] / z + K[0, 2]
                v = K[1, 1] * xyz[..., 1] / z + K[1, 2]
            yy, xx = np.indices(xyz.shape[:2], dtype=np.float64)
            projection_valid = valid_xyz & np.isfinite(u) & np.isfinite(v)
            projection_error = np.sqrt(
                (u[projection_valid] - xx[projection_valid]) ** 2
                + (v[projection_valid] - yy[projection_valid]) ** 2
            )
            assert float(np.max(projection_error)) < 1e-3

            calibration = _sample_calibration(sample)
            rectification = _sample_rectification(sample)
            direct = direct_projected_disparity(xyz, calibration, rectification)
            with np.errstate(divide="ignore", invalid="ignore"):
                simplified = (
                    rectification.fx_rect
                    * calibration.baseline
                    / (direct.left_rectified_xyz[..., 2])
                )
            equivalence_valid = direct.valid_mask & np.isfinite(simplified)
            equivalence_error = np.abs(
                direct.disparity[equivalence_valid].astype(np.float64)
                - simplified[equivalence_valid]
            )
            assert float(np.max(equivalence_error)) < 1e-3

            left_projection = project_xyz(
                direct.left_rectified_xyz,
                np.column_stack((sample["P1"][:, :3], np.zeros(3))),  # type: ignore[index]
            )
            right_projection = project_xyz(
                direct.right_rectified_xyz,
                np.column_stack((sample["P2"][:, :3], np.zeros(3))),  # type: ignore[index]
            )
            rectified_valid = equivalence_valid & left_projection.finite & right_projection.finite
            vertical_error = np.abs(
                left_projection.v[rectified_valid] - right_projection.v[rectified_valid]
            )
            assert float(np.percentile(vertical_error, 95)) < 1e-3

    with ScaredCLazyDataset(root, mode=STATIC_MODE) as static_dataset:
        for sequence_key in representatives:
            record = next(
                item for item in static_dataset.index.records if item.sequence_key == sequence_key
            )
            sample = static_dataset[record.sample_id]
            assert sample["source_type"] == "STATIC_KEYFRAME"
            assert sample["rgb_left_raw"].shape == (1024, 1280, 3)
            assert sample["xyz_left_corrected"].shape == (1024, 1280, 3)
            assert np.array_equal(
                sample["valid_xyz_mask"],
                np.isfinite(sample["xyz_left_corrected"]).all(axis=-1)
                & (sample["xyz_left_corrected"][..., 2] > 0),
            )
