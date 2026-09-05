"""Focused tests for SCARED multi-sequence datasets, manifests, and Stage-1 tensor contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image
from torch.utils.data import DataLoader

from reliable_endo_gs.data import (
    KeyframeValidationReport,
    ScaredKeyframeDataset,
    ScaredMultiSequenceDataset,
    ScaredStage1Sample,
    ScaredUpstreamWrapper,
    SplitManifest,
    load_scared_split_manifest,
    make_scared_sample_id,
    parse_scared_keyframe_id,
    validate_keyframe_processed_root,
    validate_scared_split_manifest,
    validate_stage1_sample,
    write_scared_split_manifest,
)
from reliable_endo_gs.data.scared_manifest import (
    SCARED_FIVE_KEYFRAME_SPLIT,
    SCARED_FIVE_KEYFRAME_TRAIN,
    SCARED_FIVE_KEYFRAME_VALIDATION,
    SCARED_FOUR_KEYFRAME_SPLIT,
    SCARED_FOUR_KEYFRAME_TRAIN,
    SCARED_FOUR_KEYFRAME_VALIDATION,
    validate_sample_provenance_leakage,
)
from reliable_endo_gs.data.splits import SplitManifestError


# ---------------------------------------------------------------------------
# 1. Manifest Provenance & Portable Path Tests
# ---------------------------------------------------------------------------

def test_parse_scared_keyframe_id_valid() -> None:
    ds_id, kf_id = parse_scared_keyframe_id("dataset_1/keyframe_1")
    assert ds_id == "dataset_1"
    assert kf_id == "keyframe_1"

    ds_id, kf_id = parse_scared_keyframe_id("dataset_9\\keyframe_3")
    assert ds_id == "dataset_9"
    assert kf_id == "keyframe_3"


@pytest.mark.parametrize(
    "invalid_path",
    [
        "/dataset_1/keyframe_1",
        "C:\\datasets\\dataset_1\\keyframe_1",
        "dataset_1/../dataset_2/keyframe_1",
        "dataset_1/keyframe_1/extra",
        "case_1/keyframe_1",
        "dataset_1/sequence_1",
        "",
        "   ",
    ],
)
def test_parse_scared_keyframe_id_rejects_invalid_or_absolute_paths(invalid_path: str) -> None:
    with pytest.raises(SplitManifestError):
        parse_scared_keyframe_id(invalid_path)


def test_make_scared_sample_id() -> None:
    sample_id = make_scared_sample_id("dataset_3", "keyframe_1", "frame_data000000")
    assert sample_id == "dataset_3/keyframe_1/frame_data000000"

    with pytest.raises(ValueError):
        make_scared_sample_id("", "keyframe_1", "frame_data000000")


# ---------------------------------------------------------------------------
# 2. Grouped Split & Leakage Tests
# ---------------------------------------------------------------------------

def test_committed_five_keyframe_manifest_validity() -> None:
    manifest_path = Path("splits/scared/five_keyframe_v1.json")
    assert manifest_path.is_file(), f"missing committed split manifest at {manifest_path}"

    manifest = load_scared_split_manifest(manifest_path)
    assert manifest.dataset == "scared"
    assert manifest.split_name == SCARED_FIVE_KEYFRAME_SPLIT
    assert manifest.train == SCARED_FIVE_KEYFRAME_TRAIN
    assert manifest.validation == SCARED_FIVE_KEYFRAME_VALIDATION
    assert manifest.test == ()


def test_committed_four_keyframe_manifest_validity() -> None:
    manifest_path = Path("splits/scared/four_keyframe_v2.json")
    assert manifest_path.is_file(), f"missing committed split manifest at {manifest_path}"

    manifest = load_scared_split_manifest(manifest_path)
    assert manifest.dataset == "scared"
    assert manifest.split_name == SCARED_FOUR_KEYFRAME_SPLIT
    assert manifest.train == SCARED_FOUR_KEYFRAME_TRAIN
    assert manifest.validation == SCARED_FOUR_KEYFRAME_VALIDATION
    assert manifest.test == ()
    assert "dataset_9/keyframe_3" not in manifest.validation
    assert "dataset_9/keyframe_3" not in manifest.train



def test_keyframe_overlap_leakage_is_rejected() -> None:
    leaky_manifest = SplitManifest(
        schema_version=1,
        dataset="scared",
        dataset_version="v1",
        split_name="leaky",
        train=("dataset_1/keyframe_1", "dataset_2/keyframe_1"),
        validation=("dataset_1/keyframe_1",),
        test=(),
        notes="",
    )
    with pytest.raises(SplitManifestError, match="split overlap detected"):
        validate_scared_split_manifest(leaky_manifest)


def test_case_level_leakage_is_rejected() -> None:
    case_leaky_manifest = SplitManifest(
        schema_version=1,
        dataset="scared",
        dataset_version="v1",
        split_name="case_leaky",
        train=("dataset_1/keyframe_1",),
        validation=("dataset_1/keyframe_2",),
        test=(),
        notes="",
    )
    with pytest.raises(SplitManifestError, match="case-level leakage detected"):
        validate_scared_split_manifest(case_leaky_manifest)


def test_sample_provenance_leakage_detector() -> None:
    train_samples = ["dataset_1/keyframe_1/f0", "dataset_1/keyframe_1/f1"]
    val_samples = ["dataset_7/keyframe_2/f0"]
    validate_sample_provenance_leakage(train_samples, val_samples)

    with pytest.raises(SplitManifestError, match="sample leakage detected"):
        validate_sample_provenance_leakage(train_samples, ["dataset_1/keyframe_1/f0"])


# ---------------------------------------------------------------------------
# 3. Stage-1 Tensor Contract & Validation Tests
# ---------------------------------------------------------------------------

def _valid_sample_dict(h: int = 16, w: int = 24) -> dict[str, Any]:
    return {
        "dataset_id": "dataset_3",
        "keyframe_id": "keyframe_1",
        "frame_id": "frame_data000000",
        "left": torch.zeros((3, h, w), dtype=torch.float32),
        "right": torch.zeros((3, h, w), dtype=torch.float32),
        "disparity": torch.full((h, w), 10.0, dtype=torch.float32),
        "intr": torch.eye(3, dtype=torch.float32),
        "right_intr": torch.eye(3, dtype=torch.float32),
        "extr": torch.zeros((3, 4), dtype=torch.float32),
        "Q": torch.eye(4, dtype=torch.float32),
    }


def test_validate_stage1_sample_passes_on_valid_sample() -> None:
    sample = ScaredStage1Sample(**_valid_sample_dict())
    validate_stage1_sample(sample)
    assert sample.dataset_id == "dataset_3"
    assert sample.keyframe_id == "keyframe_1"
    assert sample.frame_id == "frame_data000000"
    assert sample.sample_id == "dataset_3/keyframe_1/frame_data000000"
    assert sample.left.shape == (3, 16, 24)
    assert sample.disparity.shape == (16, 24)
    assert sample.intr.shape == (3, 3)
    assert sample.extr.shape == (3, 4)
    assert sample.Q.shape == (4, 4)


def test_validate_stage1_sample_rejects_non_finite() -> None:
    d = _valid_sample_dict()
    d["disparity"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_stage1_sample(d)

    d = _valid_sample_dict()
    d["left"][0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        validate_stage1_sample(d)

    d = _valid_sample_dict()
    d["Q"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_stage1_sample(d)


def test_validate_stage1_sample_rejects_out_of_range_images() -> None:
    d = _valid_sample_dict()
    d["left"][0, 0, 0] = 1.5
    with pytest.raises(ValueError, match="within \\[-1.0, 1.0\\]"):
        validate_stage1_sample(d)

    d = _valid_sample_dict()
    d["right"][0, 0, 0] = -1.2
    with pytest.raises(ValueError, match="within \\[-1.0, 1.0\\]"):
        validate_stage1_sample(d)


def test_validate_stage1_sample_rejects_shape_mismatches() -> None:
    d = _valid_sample_dict(h=16, w=24)
    d["right"] = torch.zeros((3, 16, 30), dtype=torch.float32)
    with pytest.raises(ValueError, match="must match 'left' shape"):
        validate_stage1_sample(d)

    d = _valid_sample_dict(h=16, w=24)
    d["disparity"] = torch.zeros((16, 30), dtype=torch.float32)
    with pytest.raises(ValueError, match="must have shape \\[H, W\\]"):
        validate_stage1_sample(d)

    d = _valid_sample_dict()
    d["intr"] = torch.zeros((4, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="shape \\[3, 3\\]"):
        validate_stage1_sample(d)

    d = _valid_sample_dict()
    d["extr"] = torch.zeros((3, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="shape \\[3, 4\\]"):
        validate_stage1_sample(d)


# ---------------------------------------------------------------------------
# 4. Synthetic Keyframe Fixtures & Multi-Sequence Dataset Tests
# ---------------------------------------------------------------------------

def _create_synthetic_keyframe(
    root: Path,
    dataset_id: str,
    keyframe_id: str,
    num_frames: int = 4,
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
        # Images (RGB 8-bit)
        left_img = Image.fromarray(np.full((height, width, 3), 128, dtype=np.uint8))
        left_img.save(kf_dir / "left_finalpass" / f"{fid}.png")
        right_img = Image.fromarray(np.full((height, width, 3), 128, dtype=np.uint8))
        right_img.save(kf_dir / "right_finalpass" / f"{fid}.png")

        # Disparity (float32 TIFF)
        disp = np.full((height, width), 25.0, dtype=np.float32)
        tifffile.imwrite(kf_dir / "disparity" / f"{fid}.tiff", disp)

        # Calibration JSON
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

        # Newpram JSON
        newpram = {
            "intr0": [[100.0, 0.0, 12.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]],
            "intr1": [[100.0, 0.0, 12.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]],
            "extr0": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        }
        (kf_dir / "newpram_data" / f"{fid}.json").write_text(json.dumps(newpram), encoding="utf-8")

        # Reprojection JSON
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


def test_keyframe_validation_report(tmp_path: Path) -> None:
    kf_dir = _create_synthetic_keyframe(tmp_path, "dataset_1", "keyframe_1", num_frames=3)
    report = validate_keyframe_processed_root(kf_dir)
    assert report.valid is True
    assert report.frame_count == 3
    assert report.image_size_wh == (24, 16)
    assert report.issues == ()


def test_scared_keyframe_dataset_reads_and_validates(tmp_path: Path) -> None:
    kf_dir = _create_synthetic_keyframe(tmp_path, "dataset_1", "keyframe_1", num_frames=6)
    ds = ScaredKeyframeDataset(kf_dir, "dataset_1", "keyframe_1", skip_every=2)
    assert len(ds) == 3

    sample = ds[0]
    assert isinstance(sample, ScaredStage1Sample)
    assert sample.dataset_id == "dataset_1"
    assert sample.keyframe_id == "keyframe_1"
    assert sample.frame_id == "frame_data000000"
    assert sample.sample_id == "dataset_1/keyframe_1/frame_data000000"
    assert sample.left.shape == (3, 16, 24)
    assert sample.disparity.shape == (16, 24)
    assert sample.intr.shape == (3, 3)
    assert sample.extr.shape == (3, 4)
    assert sample.Q.shape == (4, 4)


def test_scared_multi_sequence_dataset_composition_and_dataloader(tmp_path: Path) -> None:
    # Setup 3 train keyframes and 2 validation keyframes
    _create_synthetic_keyframe(tmp_path, "dataset_1", "keyframe_1", num_frames=4)
    _create_synthetic_keyframe(tmp_path, "dataset_2", "keyframe_1", num_frames=3)
    _create_synthetic_keyframe(tmp_path, "dataset_3", "keyframe_1", num_frames=8)

    _create_synthetic_keyframe(tmp_path, "dataset_7", "keyframe_2", num_frames=8)
    _create_synthetic_keyframe(tmp_path, "dataset_9", "keyframe_3", num_frames=4)

    manifest = SplitManifest(
        schema_version=1,
        dataset="scared",
        dataset_version="synthetic",
        split_name="synthetic_5kf",
        train=("dataset_1/keyframe_1", "dataset_2/keyframe_1", "dataset_3/keyframe_1"),
        validation=("dataset_7/keyframe_2", "dataset_9/keyframe_3"),
        test=(),
        notes="synthetic test manifest",
    )

    manifest_file = tmp_path / "split.json"
    write_scared_split_manifest(manifest, manifest_file)

    # Instantiate train dataset (with skip_every=1 to count all frames)
    train_ds = ScaredMultiSequenceDataset(
        scared_root=tmp_path,
        split_manifest=manifest_file,
        split="train",
        skip_every_map={"dataset_1": 1, "dataset_2": 1, "dataset_3": 1},
    )
    assert len(train_ds) == 4 + 3 + 8

    # Verify provenance sequence
    assert train_ds.get_sample_provenance(0) == ("dataset_1", "keyframe_1", "frame_data000000")
    assert train_ds.get_sample_provenance(4) == ("dataset_2", "keyframe_1", "frame_data000000")
    assert train_ds.get_sample_provenance(7) == ("dataset_3", "keyframe_1", "frame_data000000")

    # Instantiate validation dataset
    val_ds = ScaredMultiSequenceDataset(
        scared_root=tmp_path,
        split_manifest=manifest_file,
        split="validation",
        skip_every_map={"dataset_7": 1, "dataset_9": 1},
    )
    assert len(val_ds) == 8 + 4
    assert val_ds.get_sample_provenance(0) == ("dataset_7", "keyframe_2", "frame_data000000")
    assert val_ds.get_sample_provenance(8) == ("dataset_9", "keyframe_3", "frame_data000000")

    # Verify DataLoader with batch_size=1, num_workers=0
    loader = DataLoader(train_ds, batch_size=1, num_workers=0)
    batch = next(iter(loader))

    assert "left" in batch and batch["left"].shape == (1, 3, 16, 24)
    assert "right" in batch and batch["right"].shape == (1, 3, 16, 24)
    assert "disparity" in batch and batch["disparity"].shape == (1, 16, 24)
    assert "intr" in batch and batch["intr"].shape == (1, 3, 3)
    assert "extr" in batch and batch["extr"].shape == (1, 3, 4)
    assert "Q" in batch and batch["Q"].shape == (1, 4, 4)
    assert batch["sample_id"] == ["dataset_1/keyframe_1/frame_data000000"]
    assert batch["dataset_id"] == ["dataset_1"]
    assert batch["keyframe_id"] == ["keyframe_1"]


# ---------------------------------------------------------------------------
# 5. Real Keyframe Artifact Test (Optional / Environment Dependent)
# ---------------------------------------------------------------------------

def test_real_dataset_3_keyframe_1_sample() -> None:
    real_path = Path(r"E:\datasets\reliable-endo-gs\scared\dataset_3\keyframe_1")
    if not real_path.is_dir():
        pytest.skip("real SCARED dataset_3/keyframe_1 directory is not mounted")

    report = validate_keyframe_processed_root(real_path)
    assert report.valid is True
    assert report.frame_count == 329

    ds = ScaredKeyframeDataset(real_path, "dataset_3", "keyframe_1")
    assert len(ds) == 83  # 329 frames with default dataset_3 skip_every=4: ceil(329/4) = 83

    sample = ds[0]
    validate_stage1_sample(sample)
    assert sample.dataset_id == "dataset_3"
    assert sample.keyframe_id == "keyframe_1"
    assert sample.left.shape == (3, 1024, 1280)
    assert sample.disparity.shape == (1024, 1280)
    assert sample.intr.shape == (3, 3)
    assert sample.extr.shape == (3, 4)
    assert sample.Q.shape == (4, 4)
    assert torch.isfinite(sample.left).all()
    assert torch.isfinite(sample.disparity).all()


def test_upstream_wrapper_converts_upstream_dict() -> None:
    mock_upstream_item = {
        "name": "frame_data000000",
        "lmain": {
            "img": torch.full((3, 16, 24), 0.5, dtype=torch.float32),
            "disp": torch.full((1, 16, 24), 12.0, dtype=torch.float32),
            "intr": torch.eye(3, dtype=torch.float32),
            "extr": torch.zeros((3, 4), dtype=torch.float32),
            "disp_const": 100.0,
        },
        "rmain": {
            "img": torch.full((3, 16, 24), -0.5, dtype=torch.float32),
            "intr": torch.eye(3, dtype=torch.float32),
        },
    }

    class MockUpstreamDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return mock_upstream_item

    wrapper = ScaredUpstreamWrapper(MockUpstreamDataset(), "dataset_3", "keyframe_1")
    assert len(wrapper) == 1
    sample = wrapper[0]
    validate_stage1_sample(sample)
    assert sample.dataset_id == "dataset_3"
    assert sample.keyframe_id == "keyframe_1"
    assert sample.frame_id == "frame_data000000"
    assert sample.left.shape == (3, 16, 24)
    assert sample.disparity.shape == (16, 24)
    assert sample.Q.shape == (4, 4)