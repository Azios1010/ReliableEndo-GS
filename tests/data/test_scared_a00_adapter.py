"""A00 SCARED deployment adapter and active-manifest tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from reliable_endo_gs.data.scared_a00 import (
    ScaredA00Adapter,
    ScaredA00ProtocolError,
    ScaredA00SampleError,
    build_a01_manifest,
)


def _fixture(root: Path, *, dataset_id: str = "dataset_1", with_protocol: bool = True) -> Path:
    data = root / dataset_id / "keyframe_1" / "data"
    for name in ("frame_data", "left_finalpass", "right_finalpass", "disparity", "newpram_data"):
        (data / name).mkdir(parents=True, exist_ok=True)
    frame_id = "frame_data000000"
    Image.fromarray(np.full((4, 5, 3), 128, dtype=np.uint8)).save(
        data / "left_finalpass" / f"{frame_id}.png"
    )
    Image.fromarray(np.full((4, 5, 3), 64, dtype=np.uint8)).save(
        data / "right_finalpass" / f"{frame_id}.png"
    )
    tifffile.imwrite(
        data / "disparity" / f"{frame_id}.tiff", np.full((4, 5), 8.0, dtype=np.float32)
    )
    (data / "frame_data" / f"{frame_id}.json").write_text("{}", encoding="utf-8")
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = -0.1
    (data / "newpram_data" / f"{frame_id}.json").write_text(
        json.dumps(
            {
                "intr0": [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
                "intr1": [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
                "T_R_from_L": transform.tolist(),
            }
        ),
        encoding="utf-8",
    )
    if with_protocol:
        (data / "a00_protocol.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "image_range": "uint8_0_255",
                    "rectified": True,
                    "distortion_removed": True,
                    "pixel_center": "integer",
                    "disparity": {
                        "units": "pixels",
                        "sign": "u_left_minus_right",
                        "reference": "left",
                    },
                    "extrinsics": {
                        "source": "newpram.T_R_from_L",
                        "direction": "right_from_left",
                        "units": "m",
                    },
                    "preprocessing": {"images_grid": "processed"},
                }
            ),
            encoding="utf-8",
        )
    return root


def test_a00_adapter_emits_deployment_and_separate_supervision(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    manifest_path = tmp_path / "a01.json"
    manifest = build_a01_manifest(root, manifest_path, require_all_folds=False)
    sample = next(iter(ScaredA00Adapter(root, manifest)))
    assert sample.batch.left.min() >= 0 and sample.batch.left.max() <= 1
    assert sample.batch.left.mean() == pytest.approx(128 / 255, abs=1e-4)
    assert sample.batch.camera.baseline.tolist() == pytest.approx([0.1])
    assert sample.supervision is not None
    assert sample.supervision.disparity.shape == (1, 1, 4, 5)
    assert not hasattr(sample.batch, "gt_disparity")


@pytest.mark.parametrize("right_crop_x", [0, 1])
def test_empty_supervision_keeps_deployment_rgb_and_sensor_support(
    tmp_path: Path, right_crop_x: int
) -> None:
    root = _fixture(tmp_path)
    path = root / "dataset_1/keyframe_1/data/disparity/frame_data000000.tiff"
    tifffile.imwrite(path, np.zeros((4, 5), dtype=np.float32))
    manifest = build_a01_manifest(root, tmp_path / "a01.json", require_all_folds=False)
    adapter = ScaredA00Adapter(root, manifest)
    kwargs = dict(crop_xy_right=(right_crop_x, 0), crop_hw=(4, 4), output_hw=(2, 2))
    sample = adapter.load(manifest.entries[0].sample_id, include_supervision=True, **kwargs)
    assert sample.supervision is not None
    assert not sample.supervision.valid_mask.any()
    assert sample.batch.support.all()
    deployment = adapter.load(manifest.entries[0].sample_id, include_supervision=False, **kwargs)
    assert deployment.supervision is None
    assert torch.equal(deployment.batch.left, sample.batch.left)
    assert not hasattr(deployment.batch, "gt_disparity")


def test_a00_adapter_applies_crop_resize_and_disparity_scaling(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    manifest = build_a01_manifest(root, tmp_path / "a01.json", require_all_folds=False)
    sample = ScaredA00Adapter(root, manifest).load(
        manifest.entries[0].sample_id,
        crop_xy_left=(1, 0),
        crop_xy_right=(0, 0),
        crop_hw=(4, 4),
        output_hw=(2, 2),
    )
    assert sample.batch.left.shape == (1, 3, 2, 2)
    assert sample.batch.camera.K_L[0, 0, 0] == pytest.approx(50.0)
    assert sample.batch.camera.K_L[0, 0, 2] == pytest.approx(0.25)
    assert sample.batch.camera.K_R[0, 0, 2] == pytest.approx(0.75)
    assert sample.supervision is not None
    assert torch.allclose(sample.supervision.disparity, torch.full((1, 1, 2, 2), 3.5))


def test_a00_deployment_load_does_not_require_ground_truth(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    disparity = next((root / "dataset_1" / "keyframe_1" / "data" / "disparity").glob("*.tiff"))
    disparity.unlink()
    manifest = build_a01_manifest(root, tmp_path / "a01.json", require_all_folds=False)
    sample = ScaredA00Adapter(root, manifest).load(
        manifest.entries[0].sample_id, include_supervision=False
    )
    assert sample.supervision is None
    assert sample.batch.left.shape == (1, 3, 4, 5)


@pytest.mark.parametrize("output_hw", [(2, 2), (3, 3), (6, 7)])
def test_sparse_gt_resize_uses_rgb_pixel_centres_and_same_validity_index(
    tmp_path: Path, output_hw: tuple[int, int]
) -> None:
    root = _fixture(tmp_path)
    # Crop 4x4 from a 4x5 source. A varying disparity and holes expose the
    # old nearest offset; constant-disparity fixtures cannot detect it.
    raw = np.arange(20, dtype=np.float32).reshape(4, 5) + 10
    raw[0, 0], raw[1, 2], raw[3, 4] = 0, float("nan"), float("inf")
    tifffile.imwrite(root / "dataset_1/keyframe_1/data/disparity/frame_data000000.tiff", raw)
    manifest = build_a01_manifest(root, tmp_path / "a01.json", require_all_folds=False)
    adapter = ScaredA00Adapter(root, manifest)
    sample = adapter.load(
        manifest.entries[0].sample_id,
        crop_xy_left=(1, 0),
        crop_xy_right=(0, 0),
        crop_hw=(4, 4),
        output_hw=output_hw,
    )
    assert sample.supervision is not None
    # Independent inverse of u' = s*(u+0.5)-0.5. Nearest centre ties go
    # toward the larger index, matching nearest-exact; do not call interpolate.
    yy = np.floor((np.arange(output_hw[0]) + 0.5) * 4 / output_hw[0]).astype(int)
    xx = np.floor((np.arange(output_hw[1]) + 0.5) * 4 / output_hw[1]).astype(int)
    expected = raw[yy[:, None], xx[None, :] + 1]
    valid = np.isfinite(expected) & (expected > 0)
    expected = (expected - 1) * (output_hw[1] / 4)
    valid &= np.isfinite(expected) & (expected > 0)
    expected[~np.isfinite(expected)] = 0
    torch.testing.assert_close(sample.supervision.disparity[0, 0], torch.from_numpy(expected))
    assert torch.equal(sample.supervision.valid_mask[0, 0], torch.from_numpy(valid))
    assert sample.calibration_metadata["input_hw"] == [4, 5]
    assert sample.calibration_metadata["output_hw"] == list(output_hw)
    assert sample.calibration_metadata["runtime_resampling"]["supervision"] == (
        "nearest_exact_half_pixel_centres"
    )
    deployment = adapter.load(
        manifest.entries[0].sample_id,
        include_supervision=False,
        crop_xy_left=(1, 0),
        crop_xy_right=(0, 0),
        crop_hw=(4, 4),
        output_hw=output_hw,
    )
    assert deployment.supervision is None
    assert torch.equal(deployment.batch.left, sample.batch.left)
    assert torch.equal(deployment.batch.camera.K_L, sample.batch.camera.K_L)
    assert deployment.calibration_metadata == sample.calibration_metadata


def test_a01_manifest_requires_all_active_folds_by_default(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    with pytest.raises(ScaredA00SampleError, match="dataset_2"):
        build_a01_manifest(root, tmp_path / "a01.json")


def test_a00_protocol_missing_fails_closed(tmp_path: Path) -> None:
    root = _fixture(tmp_path, with_protocol=False)
    manifest = build_a01_manifest(root, tmp_path / "a01.json", require_all_folds=False)
    with pytest.raises(ScaredA00ProtocolError, match="missing a00_protocol.json"):
        ScaredA00Adapter(root, manifest).load(manifest.entries[0].sample_id)


def test_active_manifest_assigns_dataset_9_to_final_test(tmp_path: Path) -> None:
    root = _fixture(tmp_path, dataset_id="dataset_9")
    manifest = build_a01_manifest(root, tmp_path / "a01.json", require_all_folds=False)
    assert len(manifest.entries) == 1
    assert manifest.entries[0].sample_id == "dataset_9/keyframe_1/frame_data000000"
    assert manifest.entries[0].data_role == "final_test"
    sample = ScaredA00Adapter(root, manifest).load(
        manifest.entries[0].sample_id, include_supervision=False
    )
    assert sample.data_role == "final_test"


def test_active_adapter_rejects_dataset_10(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    manifest = build_a01_manifest(root, tmp_path / "a01.json", require_all_folds=False)
    with pytest.raises(ScaredA00SampleError, match="dataset_1 through dataset_9"):
        ScaredA00Adapter(root, manifest).load(
            "dataset_10/keyframe_1/frame_data000000", include_supervision=False
        )
