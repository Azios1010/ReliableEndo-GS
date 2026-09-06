"""Focused CPU tests for ReliableEndo-GS Stage-2 adapter.

Verifies:
- Q matrix decomposition and disparity constant derivation: disp_const = Q[2, 3] / Q[3, 2]
- Safe depth calculation, znear/zfar thresholding, and upstream min-max normalization
- Camera center C = -R.T @ t and parity with inverse world_view_transform
- Projection matrix equivalence to lib.graphics_utils.getProjectionMatrix transposed
- World-view transform and full projection transform
- Finite and positive disparity masking (handling NaNs, Infs, zeros, negative values)
- Complete output shapes, dtypes, and upstream dictionary schema
- Multi-sequence dataset wrapper and sequence/sample identity preservation (no hardcoded dataset_3)
- Collation and integration with NativeInferenceInput
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from reliable_endo_gs.baseline.native import NativeInferenceInput
from reliable_endo_gs.data import (
    DEFAULT_ZFAR,
    DEFAULT_ZNEAR,
    ScaredKeyframeDataset,
    ScaredMultiSequenceDataset,
    ScaredStage1Sample,
    ScaredStage2Dataset,
    ScaredStage2Sample,
    SplitManifest,
    compute_camera_center,
    compute_disp_const,
    compute_fov,
    compute_projection_matrix,
    compute_safe_depth,
    compute_world_view_transform,
    stage1_to_stage2_sample,
    stage2_batch_to_native_input,
    stage2_collate_fn,
    validate_stage2_sample,
)


# ---------------------------------------------------------------------------
# Helpers & Synthetic Fixtures
# ---------------------------------------------------------------------------

def _make_stage1_sample(
    *,
    h: int = 32,
    w: int = 48,
    dataset_id: str = "dataset_1",
    keyframe_id: str = "keyframe_1",
    frame_id: str = "frame_data000000",
    disp_val: float = 20.0,
    fx: float = 200.0,
    fy: float = 200.0,
    baseline: float = 4.0,
    R: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
) -> ScaredStage1Sample:
    cx = float(w) / 2.0
    cy = float(h) / 2.0

    intr = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    right_intr = intr.clone()

    if R is None:
        R = torch.eye(3, dtype=torch.float32)
    if t is None:
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    extr = torch.cat([R, t.unsqueeze(1)], dim=1)

    Q = torch.tensor(
        [
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 0.0, fx],
            [0.0, 0.0, 1.0 / baseline, 0.0],
        ],
        dtype=torch.float32,
    )

    left = torch.zeros((3, h, w), dtype=torch.float32)
    right = torch.zeros((3, h, w), dtype=torch.float32)
    disparity = torch.full((h, w), disp_val, dtype=torch.float32)

    return ScaredStage1Sample(
        dataset_id=dataset_id,
        keyframe_id=keyframe_id,
        frame_id=frame_id,
        left=left,
        right=right,
        disparity=disparity,
        intr=intr,
        right_intr=right_intr,
        extr=extr,
        Q=Q,
    )


def _create_synthetic_keyframe(
    root: Path,
    dataset_id: str,
    keyframe_id: str,
    num_frames: int = 3,
    height: int = 16,
    width: int = 24,
) -> Path:
    kf_dir = root / dataset_id / keyframe_id / "data"
    for subdir in (
        "frame_data",
        "left_finalpass",
        "right_finalpass",
        "disparity",
        "newpram_data",
        "reprojection_data",
    ):
        (kf_dir / subdir).mkdir(parents=True, exist_ok=True)

    for i in range(num_frames):
        fid = f"frame_data{i:06d}"
        left_img = Image.fromarray(np.full((height, width, 3), 128, dtype=np.uint8))
        left_img.save(kf_dir / "left_finalpass" / f"{fid}.png")
        right_img = Image.fromarray(np.full((height, width, 3), 128, dtype=np.uint8))
        right_img.save(kf_dir / "right_finalpass" / f"{fid}.png")

        disp = np.full((height, width), 25.0, dtype=np.float32)
        tifffile.imwrite(kf_dir / "disparity" / f"{fid}.tiff", disp)

        calib_data = {
            "camera-calibration": {
                "KL": [[100.0, 0.0, 12.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]],
                "KR": [[100.0, 0.0, 12.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]],
                "DL": [[0.0] * 5],
                "DR": [[0.0] * 5],
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "T": [[-4.0], [0.0], [0.0]],
            },
            "camera-pose": np.eye(4).tolist(),
        }
        (kf_dir / "frame_data" / f"{fid}.json").write_text(json.dumps(calib_data), encoding="utf-8")

        newpram = {
            "intr0": [[100.0, 0.0, 12.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]],
            "intr1": [[100.0, 0.0, 12.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]],
            "extr0": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        }
        (kf_dir / "newpram_data" / f"{fid}.json").write_text(json.dumps(newpram), encoding="utf-8")

        reproj = {
            "reprojection-matrix": [
                [1.0, 0.0, 0.0, -12.0],
                [0.0, 1.0, 0.0, -8.0],
                [0.0, 0.0, 0.0, 100.0],
                [0.0, 0.0, 0.25, 0.0],
            ]
        }
        (kf_dir / "reprojection_data" / f"{fid}.json").write_text(json.dumps(reproj), encoding="utf-8")

    return root / dataset_id / keyframe_id


# ---------------------------------------------------------------------------
# 1. Q Matrix & Disparity Constant Tests
# ---------------------------------------------------------------------------

def test_compute_disp_const_canonical() -> None:
    Q = torch.tensor(
        [
            [1.0, 0.0, 0.0, -100.0],
            [0.0, 1.0, 0.0, -100.0],
            [0.0, 0.0, 0.0, 500.0],
            [0.0, 0.0, 0.25, 0.0],
        ],
        dtype=torch.float32,
    )
    # disp_const = Q[2, 3] / Q[3, 2] = 500.0 / 0.25 = 2000.0
    disp_const = compute_disp_const(Q)
    assert isinstance(disp_const, float)
    assert pytest.approx(disp_const, rel=1e-6) == 2000.0


def test_compute_disp_const_zero_denominator_raises() -> None:
    Q = torch.eye(4, dtype=torch.float32)  # Q[3, 2] == 0.0
    with pytest.raises(ValueError, match="zero or near-zero"):
        compute_disp_const(Q)


def test_compute_disp_const_invalid_shape_raises() -> None:
    Q = torch.eye(3, dtype=torch.float32)
    with pytest.raises(ValueError, match="shape \\[4, 4\\]"):
        compute_disp_const(Q)


# ---------------------------------------------------------------------------
# 2. Camera Center Tests (C = -R.T @ t)
# ---------------------------------------------------------------------------

def test_compute_camera_center_identity_pose() -> None:
    extr = torch.tensor(
        [
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, -3.0],
            [0.0, 0.0, 1.0, 10.0],
        ],
        dtype=torch.float32,
    )
    # C = -I @ t = -t
    c = compute_camera_center(extr)
    assert c.shape == (3,)
    assert torch.allclose(c, torch.tensor([-5.0, 3.0, -10.0]))


def test_compute_camera_center_with_rotation_and_parity_with_world_view_inv() -> None:
    # 90-degree yaw rotation around Y
    theta = math.pi / 2.0
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32)
    t = torch.tensor([10.0, -20.0, 30.0], dtype=torch.float32)
    extr = torch.cat([R, t.unsqueeze(1)], dim=1)

    c_formula = compute_camera_center(extr)
    w2v = compute_world_view_transform(extr)
    c_inv = w2v.inverse()[3, :3]

    assert c_formula.shape == (3,)
    assert torch.allclose(c_formula, c_inv, atol=1e-5)


def test_compute_camera_center_arbitrary_pose() -> None:
    # Arbitrary orthogonal rotation matrix from Rodrigues
    axis = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    axis = axis / torch.norm(axis)
    angle = 0.75
    K = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=torch.float32,
    )
    R = torch.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    t = torch.tensor([1.5, -3.2, 8.4], dtype=torch.float32)
    extr = torch.cat([R, t.unsqueeze(1)], dim=1)

    c_formula = compute_camera_center(extr)
    w2v = compute_world_view_transform(extr)
    c_inv = w2v.inverse()[3, :3]

    assert torch.allclose(c_formula, c_inv, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. Projection Matrix & World-View Transform Tests
# ---------------------------------------------------------------------------

def test_compute_projection_matrix_matches_upstream() -> None:
    from lib.graphics_utils import getProjectionMatrix

    h, w = 1024, 1280
    intr = torch.tensor(
        [[1050.2, 0.0, 642.5], [0.0, 1049.8, 511.3], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    znear, zfar = 0.03, 300.0

    # Upstream computes getProjectionMatrix then transposes
    expected_P = getProjectionMatrix(znear, zfar, intr, h, w).transpose(0, 1)
    actual_P = compute_projection_matrix(intr, h, w, znear=znear, zfar=zfar)

    assert actual_P.shape == (4, 4)
    assert actual_P.dtype == torch.float32
    assert torch.allclose(actual_P, expected_P, atol=1e-7)


def test_compute_world_view_transform_matches_upstream() -> None:
    from lib.graphics_utils import getWorld2View2

    extr = torch.tensor(
        [
            [0.9998, 0.0012, 0.0182, 12.3],
            [-0.0014, 0.9999, 0.0091, -4.5],
            [-0.0182, -0.0091, 0.9998, 56.7],
        ],
        dtype=torch.float32,
    )
    R_np = extr[:, :3].numpy().transpose(1, 0)
    T_np = extr[:, 3].numpy()

    expected_w2v = torch.tensor(getWorld2View2(R_np, T_np)).transpose(0, 1)
    actual_w2v = compute_world_view_transform(extr)

    assert actual_w2v.shape == (4, 4)
    assert actual_w2v.dtype == torch.float32
    assert torch.allclose(actual_w2v, expected_w2v, atol=1e-6)


def test_full_proj_transform_composition() -> None:
    intr = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    extr = torch.tensor([[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 5.0]])

    w2v = compute_world_view_transform(extr)
    proj = compute_projection_matrix(intr, 480, 640)
    fproj = w2v @ proj

    # Check equivalent to bmm
    fproj_bmm = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    assert torch.allclose(fproj, fproj_bmm)


def test_compute_fov() -> None:
    fov_x = compute_fov(500.0, 1000.0)
    # 2 * atan(1000 / (2 * 500)) = 2 * atan(1) = pi / 2
    assert pytest.approx(fov_x, rel=1e-6) == math.pi / 2.0


# ---------------------------------------------------------------------------
# 4. Safe Disparity & Depth Masking Tests
# ---------------------------------------------------------------------------

def test_compute_safe_depth_valid_positive() -> None:
    disp = torch.tensor([[10.0, 20.0], [40.0, 50.0]], dtype=torch.float32)
    disp_const = 1000.0
    # raw depths: [100.0, 50.0, 25.0, 20.0]
    # min=20.0, max=100.0, range=80.0
    # norm depths: (depth - 20) / 80 = [1.0, 0.375, 0.0625, 0.0]
    clean_disp, mask, depth = compute_safe_depth(disp, disp_const, znear=0.03, zfar=300.0)

    assert clean_disp.shape == (1, 2, 2)
    assert mask.shape == (1, 2, 2)
    assert depth.shape == (1, 2, 2)

    assert torch.equal(mask, torch.ones((1, 2, 2)))
    assert torch.allclose(clean_disp.squeeze(0), disp)
    expected_depth = torch.tensor([[1.0, 0.375], [0.0625, 0.0]])
    assert torch.allclose(depth.squeeze(0), expected_depth)


def test_compute_safe_depth_masks_zeros_negatives_nans_infs() -> None:
    disp = torch.tensor(
        [
            [0.0, -10.0, float("nan")],
            [float("inf"), float("-inf"), 20.0],
        ],
        dtype=torch.float32,
    )
    disp_const = 500.0

    clean_disp, mask, depth = compute_safe_depth(disp, disp_const, znear=0.03, zfar=300.0)

    # Only element (1, 2) is valid with disparity 20.0
    expected_mask = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    assert torch.equal(mask, expected_mask)

    # Clean disp must be 0 for invalid and 20.0 for valid
    expected_disp = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]]])
    assert torch.equal(clean_disp, expected_disp)

    # Depth must contain NO NaNs or Infs anywhere
    assert torch.isfinite(depth).all()
    assert torch.isfinite(clean_disp).all()
    assert torch.isfinite(mask).all()

    # Invalid locations have depth 0.0
    assert depth[0, 0, 0] == 0.0
    assert depth[0, 0, 1] == 0.0
    assert depth[0, 0, 2] == 0.0
    assert depth[0, 1, 0] == 0.0
    assert depth[0, 1, 1] == 0.0


def test_compute_safe_depth_clip_znear_zfar() -> None:
    # disp_const = 100.0
    # disp = 10000.0 -> depth = 0.01 < znear (0.03) -> zeroed out
    # disp = 0.2 -> depth = 500.0 > zfar (300.0) -> zeroed out
    # disp = 10.0 -> depth = 10.0 -> in range
    disp = torch.tensor([[10000.0, 0.2], [10.0, 20.0]], dtype=torch.float32)
    disp_const = 100.0

    clean_disp, mask, depth = compute_safe_depth(disp, disp_const, znear=0.03, zfar=300.0)
    assert torch.isfinite(depth).all()
    # element (0, 0) clipped because < znear, element (0, 1) clipped because > zfar
    assert depth[0, 0, 0] == 0.0
    assert depth[0, 0, 1] == 0.0


def test_compute_safe_depth_all_invalid_no_crash() -> None:
    disp = torch.zeros((10, 10), dtype=torch.float32)
    clean_disp, mask, depth = compute_safe_depth(disp, 500.0)

    assert torch.equal(mask, torch.zeros((1, 10, 10)))
    assert torch.equal(clean_disp, torch.zeros((1, 10, 10)))
    assert torch.equal(depth, torch.zeros((1, 10, 10)))
    assert torch.isfinite(depth).all()


def test_compute_safe_depth_constant_valid_no_nan() -> None:
    # All pixels have identical disparity 20.0 -> max - min == 0
    disp = torch.full((8, 8), 20.0, dtype=torch.float32)
    clean_disp, mask, depth = compute_safe_depth(disp, 500.0)

    assert torch.equal(mask, torch.ones((1, 8, 8)))
    assert torch.isfinite(depth).all()
    assert (depth == 0.0).all()  # safely zeroed rather than NaN


# ---------------------------------------------------------------------------
# 5. Full Stage-1 to Stage-2 Sample Transformation & Validation
# ---------------------------------------------------------------------------

def test_stage1_to_stage2_sample_structure_and_types() -> None:
    h, w = 32, 48
    s1 = _make_stage1_sample(h=h, w=w, dataset_id="dataset_1", keyframe_id="keyframe_1", frame_id="frame_001")
    s2 = stage1_to_stage2_sample(s1)

    assert isinstance(s2, ScaredStage2Sample)
    assert isinstance(s2, dict)
    validate_stage2_sample(s2)

    # Check top-level keys
    assert "name" in s2
    assert "lmain" in s2
    assert "rmain" in s2
    assert s2.name == "dataset_1/keyframe_1/frame_001"
    assert s2.sample_id == "dataset_1/keyframe_1/frame_001"
    assert s2.dataset_id == "dataset_1"
    assert s2.keyframe_id == "keyframe_1"
    assert s2.frame_id == "frame_001"

    # Check lmain contents
    lmain = s2.lmain
    assert lmain["img"].shape == (3, h, w)
    assert lmain["img"].dtype == torch.float32
    assert lmain["mask"].shape == (1, h, w)
    assert lmain["mask"].dtype == torch.float32
    assert lmain["disp"].shape == (1, h, w)
    assert lmain["disp"].dtype == torch.float32
    assert lmain["depth"].shape == (1, h, w)
    assert lmain["depth"].dtype == torch.float32
    assert isinstance(lmain["disp_const"], float)
    assert lmain["intr"].shape == (3, 3)
    assert lmain["intr"].dtype == torch.float32
    assert lmain["extr"].shape == (3, 4)
    assert lmain["extr"].dtype == torch.float32
    assert isinstance(lmain["FovX"], float)
    assert isinstance(lmain["FovY"], float)
    assert lmain["width"] == w
    assert lmain["height"] == h
    assert lmain["world_view_transform"].shape == (4, 4)
    assert lmain["world_view_transform"].dtype == torch.float32
    assert lmain["full_proj_transform"].shape == (4, 4)
    assert lmain["full_proj_transform"].dtype == torch.float32
    assert lmain["camera_center"].shape == (3,)
    assert lmain["camera_center"].dtype == torch.float32

    # Check rmain contents
    rmain = s2.rmain
    assert rmain["img"].shape == (3, h, w)
    assert rmain["img"].dtype == torch.float32
    assert rmain["intr"].shape == (3, 3)
    assert rmain["intr"].dtype == torch.float32

    # Check property shortcuts
    assert torch.equal(s2.left, lmain["img"])
    assert torch.equal(s2.right, rmain["img"])
    assert torch.equal(s2.mask, lmain["mask"])
    assert torch.equal(s2.disparity, lmain["disp"])
    assert torch.equal(s2.depth, lmain["depth"])
    assert torch.equal(s2.world_view_transform, lmain["world_view_transform"])
    assert torch.equal(s2.full_proj_transform, lmain["full_proj_transform"])
    assert torch.equal(s2.camera_center, lmain["camera_center"])


def test_validate_stage2_sample_rejects_missing_keys_or_bad_shapes() -> None:
    s1 = _make_stage1_sample(h=16, w=16)
    s2 = stage1_to_stage2_sample(s1)

    # Missing lmain
    bad = dict(s2)
    del bad["lmain"]
    with pytest.raises(ValueError, match="missing 'lmain'"):
        validate_stage2_sample(bad)

    # Bad camera center shape
    bad = dict(s2)
    bad["lmain"] = dict(s2.lmain)
    bad["lmain"]["camera_center"] = torch.zeros(4)
    with pytest.raises(ValueError, match="camera_center"):
        validate_stage2_sample(bad)

    # Bad depth dimensions
    bad = dict(s2)
    bad["lmain"] = dict(s2.lmain)
    bad["lmain"]["depth"] = torch.zeros((16, 16))
    with pytest.raises(ValueError, match="depth"):
        validate_stage2_sample(bad)


# ---------------------------------------------------------------------------
# 6. Multi-Sequence Dataset Wrapper & Sequence Identity (No Hardcoded dataset_3)
# ---------------------------------------------------------------------------

def test_stage2_dataset_multi_sequence_identity(tmp_path: Path) -> None:
    # Build synthetic keyframes for dataset_1, dataset_2, dataset_7, dataset_4 (matching v3 manifest sequences)
    kf1 = _create_synthetic_keyframe(tmp_path, "dataset_1", "keyframe_1", num_frames=2)
    kf2 = _create_synthetic_keyframe(tmp_path, "dataset_2", "keyframe_1", num_frames=2)
    kf7 = _create_synthetic_keyframe(tmp_path, "dataset_7", "keyframe_2", num_frames=2)
    kf4 = _create_synthetic_keyframe(tmp_path, "dataset_4", "keyframe_4", num_frames=2)

    manifest = SplitManifest(
        schema_version=1,
        dataset="scared",
        dataset_version="v3_test",
        split_name="v3_multi_test",
        train=("dataset_1/keyframe_1", "dataset_2/keyframe_1"),
        validation=("dataset_7/keyframe_2", "dataset_4/keyframe_4"),
        test=(),
        notes="Synthetic test manifest for v3 multi-sequence testing",
    )

    skip_map = {
        "dataset_1/keyframe_1": 1,
        "dataset_2/keyframe_1": 1,
        "dataset_7/keyframe_2": 1,
        "dataset_4/keyframe_4": 1,
    }

    # Train split wrapper
    train_ds = ScaredStage2Dataset(
        scared_root=tmp_path,
        split_manifest=manifest,
        split="train",
        skip_every_map=skip_map,
    )
    assert len(train_ds) == 4  # 2 frames * 2 keyframes
    assert train_ds.keyframe_entries == ("dataset_1/keyframe_1", "dataset_2/keyframe_1")

    # Check sample identities
    expected_sample_ids = (
        "dataset_1/keyframe_1/frame_data000000",
        "dataset_1/keyframe_1/frame_data000001",
        "dataset_2/keyframe_1/frame_data000000",
        "dataset_2/keyframe_1/frame_data000001",
    )
    assert train_ds.sample_ids == expected_sample_ids

    for i in range(len(train_ds)):
        prov = train_ds.get_sample_provenance(i)
        expected_ds, expected_kf, expected_fid = expected_sample_ids[i].split("/")
        assert prov == (expected_ds, expected_kf, expected_fid)

        sample = train_ds[i]
        validate_stage2_sample(sample)
        assert sample.name == expected_sample_ids[i]
        assert sample.dataset_id == expected_ds
        assert sample.keyframe_id == expected_kf
        assert sample.frame_id == expected_fid
        assert sample.sample_id == expected_sample_ids[i]

    # Validation split wrapper (completely held out dataset_7 and dataset_4)
    val_ds = ScaredStage2Dataset(
        scared_root=tmp_path,
        split_manifest=manifest,
        split="validation",
        skip_every_map=skip_map,
    )
    assert len(val_ds) == 4
    assert val_ds.keyframe_entries == ("dataset_7/keyframe_2", "dataset_4/keyframe_4")
    val_sample_0 = val_ds[0]
    validate_stage2_sample(val_sample_0)
    assert val_sample_0.dataset_id == "dataset_7"
    assert val_sample_0.keyframe_id == "keyframe_2"


def test_stage2_dataset_wrapping_existing_stage1_dataset() -> None:
    s1 = _make_stage1_sample(dataset_id="dataset_6", keyframe_id="keyframe_3", frame_id="frame_042")

    class MockStage1Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> ScaredStage1Sample:
            return s1

        @property
        def sample_ids(self) -> tuple[str, ...]:
            return ("dataset_6/keyframe_3/frame_042",)

        def get_sample_provenance(self, index: int) -> tuple[str, str, str]:
            return "dataset_6", "keyframe_3", "frame_042"

    ds = ScaredStage2Dataset(stage1_dataset=MockStage1Dataset())
    assert len(ds) == 1
    assert ds.sample_ids == ("dataset_6/keyframe_3/frame_042",)
    assert ds.get_sample_provenance(0) == ("dataset_6", "keyframe_3", "frame_042")

    sample = ds[0]
    validate_stage2_sample(sample)
    assert sample.dataset_id == "dataset_6"
    assert sample.keyframe_id == "keyframe_3"
    assert sample.frame_id == "frame_042"


# ---------------------------------------------------------------------------
# 7. Batch Collation & NativeInferenceInput Integration Tests
# ---------------------------------------------------------------------------

def test_stage2_collate_fn_and_native_inference_input() -> None:
    s1_a = _make_stage1_sample(h=16, w=24, dataset_id="dataset_1", frame_id="f0")
    s1_b = _make_stage1_sample(h=16, w=24, dataset_id="dataset_2", frame_id="f1")

    s2_a = stage1_to_stage2_sample(s1_a)
    s2_b = stage1_to_stage2_sample(s1_b)

    batch = stage2_collate_fn([s2_a, s2_b])

    assert batch["name"] == ["dataset_1/keyframe_1/f0", "dataset_2/keyframe_1/f1"]
    assert batch["lmain"]["img"].shape == (2, 3, 16, 24)
    assert batch["rmain"]["img"].shape == (2, 3, 16, 24)
    assert batch["lmain"]["mask"].shape == (2, 1, 16, 24)
    assert batch["lmain"]["disp"].shape == (2, 1, 16, 24)
    assert batch["lmain"]["depth"].shape == (2, 1, 16, 24)
    assert batch["lmain"]["disp_const"].shape == (2,)
    assert batch["lmain"]["intr"].shape == (2, 3, 3)
    assert batch["lmain"]["extr"].shape == (2, 3, 4)
    assert batch["lmain"]["world_view_transform"].shape == (2, 4, 4)
    assert batch["lmain"]["full_proj_transform"].shape == (2, 4, 4)
    assert batch["lmain"]["camera_center"].shape == (2, 3)

    # Convert to NativeInferenceInput
    native_input = stage2_batch_to_native_input(batch)
    assert isinstance(native_input, NativeInferenceInput)
    assert native_input.batch_size == 2
    assert native_input.sample_names == ("dataset_1/keyframe_1/f0", "dataset_2/keyframe_1/f1")

    # Check upstream data mapping
    upstream_dict = native_input.to_upstream_data(torch.device("cpu"))
    assert upstream_dict["name"] == ("dataset_1/keyframe_1/f0", "dataset_2/keyframe_1/f1")
    assert "lmain" in upstream_dict and "rmain" in upstream_dict
    assert upstream_dict["lmain"]["img"].shape == (2, 3, 16, 24)


def test_single_sample_to_native_inference_input() -> None:
    s1 = _make_stage1_sample(h=16, w=24, dataset_id="dataset_3", frame_id="f0")
    s2 = stage1_to_stage2_sample(s1)

    native_input = s2.to_native_inference_input()
    assert isinstance(native_input, NativeInferenceInput)
    assert native_input.batch_size == 1
    assert native_input.sample_names == ("dataset_3/keyframe_1/f0",)
    assert native_input.left_image.shape == (1, 3, 16, 24)
    assert native_input.left_camera_center.shape == (1, 3)
