"""SCARED-C tensor decoding and contract tests."""

from pathlib import Path

import pytest
import torch

from reliable_endo_gs.data.index import build_scared_c_index
from reliable_endo_gs.data.loaders import DataDecodeError, load_scared_c_sample


def test_loader_decodes_real_fixture_with_explicit_masks(scared_c_root: Path) -> None:
    record = build_scared_c_index(scared_c_root).records[0]

    batch = load_scared_c_sample(record)

    assert batch.sample_ids == ("scared_c/dataset_1/keyframe_1/reference",)
    assert batch.sequence_ids == ("dataset_1",)
    assert batch.left.shape == (1, 3, 3, 4)
    assert batch.right.shape == (1, 3, 3, 4)
    assert batch.left.dtype == torch.float32
    assert batch.left.min() >= 0.0
    assert batch.left.max() <= 1.0
    assert batch.gt_depth is None
    assert batch.gt_depth_xyz is not None
    assert batch.gt_right_depth_xyz is not None
    assert batch.gt_depth_xyz.shape == (1, 3, 3, 4)
    assert batch.gt_right_depth_xyz.shape == (1, 3, 3, 4)
    assert batch.masks["left_depth_valid"].dtype == torch.bool
    assert batch.masks["right_depth_valid"].dtype == torch.bool
    assert int(batch.masks["left_depth_valid"].sum()) == 10
    assert int(batch.masks["right_depth_valid"].sum()) == 10
    assert batch.metadata["protocol"] == "scared_c_endoscope_stereo_calibration_v1"
    assert batch.metadata["depth_representation"] == "per-pixel XYZ coordinate map"
    assert batch.metadata["depth_units"] == "UNRESOLVED"
    assert batch.metadata["pose_applied"] is False
    assert batch.metadata["stereo_extrinsics_applied"] is False
    assert "E:\\" not in " ".join(str(value) for value in batch.metadata.values())


def test_loader_uses_both_endoscope_stereo_intrinsics(scared_c_root: Path) -> None:
    record = build_scared_c_index(scared_c_root).records[0]
    batch = load_scared_c_sample(record)

    assert batch.left_camera.intrinsics[0, 0, 0] == 100.0
    assert batch.right_camera.intrinsics[0, 0, 0] == 110.0
    assert torch.equal(batch.left_camera.world_from_camera, torch.eye(4).unsqueeze(0))
    assert batch.metadata["intrinsics_source"] == "endoscope_calibration.yaml:M1,M2"
    assert batch.metadata["pose_source"].startswith("frame_data.tar.gz")


def test_loader_applies_resize_and_keeps_camera_intrinsics_analytic(scared_c_root: Path) -> None:
    record = build_scared_c_index(scared_c_root).records[0]

    batch = load_scared_c_sample(record, target_size=(2, 2))

    assert batch.spatial_shape == (2, 2)
    assert batch.left_camera.intrinsics[0, 0, 0] == 50.0
    assert batch.left_camera.intrinsics[0, 1, 1] == pytest.approx(101.0 * 2 / 3)
    assert batch.left_camera.intrinsics[0, 0, 2] == 1.0
    assert batch.gt_depth_xyz is not None
    assert batch.gt_depth_xyz.shape == (1, 3, 2, 2)


def test_loader_rejects_non_index_record(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="ScaredCRecord"):
        load_scared_c_sample(tmp_path)  # type: ignore[arg-type]


def test_loader_rejects_wrong_tiff_shape(scared_c_root: Path) -> None:
    import numpy as np
    import tifffile

    record = build_scared_c_index(scared_c_root).records[0]
    tifffile.imwrite(record.right_depth_path, np.ones((2, 2), dtype=np.float32))

    with pytest.raises(DataDecodeError, match="XYZ input"):
        load_scared_c_sample(record)
