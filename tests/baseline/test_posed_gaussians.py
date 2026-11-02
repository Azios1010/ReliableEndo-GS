"""Tiny CPU boundary checks; synthetic predictions are not baseline measurements."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from reliable_endo_gs.baseline.posed_gaussians import (
    PosedGaussianConfig,
    PosedGaussianPredictor,
    posed_context,
)
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample, ScaredSupervision


def batch() -> DeploymentStereoBatch:
    """Synthetic 4×8, 2 cm baseline, unequal principal points, optical depth 20 cm."""
    left_k = torch.tensor([[[100.0, 0, 3.0], [0, 100.0, 2.0], [0, 0, 1.0]]])
    right_k = left_k.clone()
    right_k[:, 0, 2] = 1.0
    transform = torch.eye(4)[None]
    transform[:, 0, 3] = -0.02
    camera = RectifiedStereoCamera(left_k, right_k, transform, True)
    return DeploymentStereoBatch(
        torch.full((1, 3, 4, 8), 0.25),
        torch.full((1, 3, 4, 8), 0.75),
        camera,
        torch.ones(1, 1, 4, 8, dtype=torch.bool),
    )


class Encoder(nn.Module):
    def __init__(self, as_dict: bool):
        super().__init__()
        self.as_dict = as_dict

    def forward(self, context, step, deterministic, visualization_dump):
        assert deterministic and step == 300001
        assert set(context) == {"image", "intrinsics", "extrinsics", "near", "far", "index"}
        _, _, _, h, w = context["image"].shape
        count = 2 * h * w
        visualization_dump.update(
            depth=torch.full((1, 2, h, w, 1, 1), 0.2),
            scales=torch.full((1, count, 3), 0.001),
            rotations=torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(1, count, 4),
        )
        self.gaussians = SimpleNamespace(
            means=torch.full((1, count, 3), 0.1),
            harmonics=torch.ones(1, count, 3, 4),
            opacities=torch.full((1, count), 0.5),
            covariances=torch.eye(3).expand(1, count, 3, 3) * 1e-6,
        )
        return {"gaussians": self.gaussians} if self.as_dict else self.gaussians


class Decoder(nn.Module):
    def forward(self, gaussians, extrinsics, intrinsics, near, far, hw):
        self.seen = gaussians
        self.poses = extrinsics.clone()
        h, w = hw
        color = torch.full((1, 2, 3, h, w), 0.4)
        color[:, 1] = 0.6
        return SimpleNamespace(color=color)


def test_camera_direction_half_pixel_and_no_input_mutation():
    b = batch()
    k = b.camera.K_L.clone()
    context = posed_context(b, PosedGaussianConfig(0.01, 1.0, 300001))
    torch.testing.assert_close(context["extrinsics"][0, 1, 0, 3], torch.tensor(0.02))
    point = torch.tensor([0.0, 0.0, 0.2, 1.0])
    in_right = torch.linalg.solve(context["extrinsics"][0, 1], point)
    torch.testing.assert_close(in_right[:3], torch.tensor([-0.02, 0, 0.2]))
    torch.testing.assert_close(context["intrinsics"][0, 0, 0, 2], torch.tensor(3.5 / 8))
    pixel = torch.tensor([3.5 / 8, 2.5 / 4, 1.0])
    ray = torch.linalg.solve(context["intrinsics"][0, 0], pixel)
    torch.testing.assert_close(ray, torch.tensor([0.0, 0.0, 1.0]))
    context["intrinsics"][0, 0, 0, 2] = 99
    torch.testing.assert_close(b.camera.K_L, k)


@pytest.mark.parametrize("as_dict", [False, True], ids=["mvsplat", "depthsplat"])
def test_forward_native_renderer_depth_conversion_and_source_provenance(as_dict):
    b = batch()
    encoder, decoder = Encoder(as_dict), Decoder()
    predictor = PosedGaussianPredictor(encoder, decoder, PosedGaussianConfig(0.01, 1.0, 300001))
    sample = ScaredA00Sample("dataset_7/keyframe_1/frame_data000000", "final_train", b, {})
    out = predictor(sample)
    assert decoder.seen is encoder.gaussians
    torch.testing.assert_close(out.disparity, torch.full((1, 1, 4, 8), 12.0))
    torch.testing.assert_close(out.left.image, torch.full((1, 3, 4, 8), 0.4))
    torch.testing.assert_close(out.right.image, torch.full((1, 3, 4, 8), 0.6))
    torch.testing.assert_close(out.field.cov_surface, encoder.gaussians.covariances)
    assert torch.equal(out.source_index[0, :32], torch.arange(32))
    assert (out.source_index[0, 32:] == -1).all()
    torch.testing.assert_close(out.field.rotations[..., 0], torch.ones(1, 64))
    assert out.left.depth is None
    torch.testing.assert_close(predictor.predict_disparity(b), out.disparity)
    contaminated = replace(sample, supervision=ScaredSupervision(out.disparity, b.support))
    with pytest.raises(ValueError, match="ground truth"):
        predictor(contaminated)


def test_principal_point_crop_shift_has_correct_raw_disparity_sign():
    b = batch()
    predictor = PosedGaussianPredictor(
        Encoder(False), Decoder(), PosedGaussianConfig(0.01, 1, 300001)
    )
    original = predictor.predict_disparity(b)
    camera = b.camera
    left_k, right_k = camera.K_L.clone(), camera.K_R.clone()
    left_k[:, 0, 2] -= 3
    right_k[:, 0, 2] -= 1
    cropped = replace(b, camera=RectifiedStereoCamera(left_k, right_k, camera.T_R_from_L, True))
    torch.testing.assert_close(predictor.predict_disparity(cropped), original - 2)


@pytest.mark.parametrize("near,far", [(0, 1), (1, 1), (-1, 1), (1, float("nan"))])
def test_bounds_require_explicit_valid_metres(near, far):
    with pytest.raises(ValueError):
        PosedGaussianConfig(near, far, 1)


@pytest.mark.parametrize("commit,dirty", [("different", ""), ("expected", " M source.py")])
def test_native_loader_rejects_changed_upstream_before_import(tmp_path, monkeypatch, commit, dirty):
    from reliable_endo_gs.baseline import posed_gaussians_native as native

    results = iter([commit, dirty])
    monkeypatch.setattr(native.subprocess, "check_output", lambda *a, **k: next(results))
    with pytest.raises(ValueError, match="upstream commit"):
        native.build_posed_predictor(
            "mvsplat", tmp_path, "expected", tmp_path / "unused.ckpt", near_m=0.01, far_m=1
        )


def test_cohorts_follow_fixed_rule_and_reject_test_dataset(tmp_path: Path):
    from reliable_endo_gs.baseline.posed_benchmark import development_cohorts, digest

    val_ids = [f"dataset_7/keyframe_1/frame_data{k:06d}" for k in range(56)]
    train_ids = [
        f"dataset_{d}/keyframe_1/frame_data{k:06d}" for d in (1, 2, 3, 6) for k in range(28)
    ]
    files = {
        "samples": {"sample_ids": val_ids},
        "validation_manifest": {"entries": []},
        "training_manifest": {"entries": [{"sample_id": s} for s in train_ids]},
    }
    original = {"pinned_files": []}
    for name, payload in files.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(payload))
        original[name] = str(path)
        original["pinned_files"].append({"path": str(path), "sha256": digest(path)})
    config = tmp_path / "benchmark-config.json"
    config.write_text(json.dumps(original))
    (tmp_path / "usable_ids.json").write_text(json.dumps({"excluded": []}))
    cohorts = development_cohorts({"benchmark_config": str(config)})
    assert cohorts["val_7"] == val_ids
    assert cohorts["train_1236"] == [s for s in train_ids if int(s[-6:]) % 2 == 1]
    files["samples"]["sample_ids"][0] = "dataset_8/keyframe_1/frame_data000000"
    path = Path(original["samples"])
    path.write_text(json.dumps(files["samples"]))
    original["pinned_files"][0]["sha256"] = digest(path)
    config.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="exact 56"):
        development_cohorts({"benchmark_config": str(config)})


def test_left_source_only_renders_with_the_left_grid_only():
    b = batch()
    encoder, decoder = Encoder(False), Decoder()
    config = replace(PosedGaussianConfig(0.01, 1.0, 300001), left_source_only=True)
    predictor = PosedGaussianPredictor(encoder, decoder, config)
    sample = ScaredA00Sample("dataset_7/keyframe_1/frame_data000000", "final_train", b, {})
    both = PosedGaussianPredictor(Encoder(False), Decoder(), PosedGaussianConfig(0.01, 1.0, 300001))
    out = predictor(sample)
    count = 4 * 8
    assert decoder.seen.means.shape == (1, count, 3) and decoder.seen.opacities.shape == (1, count)
    assert decoder.seen.harmonics.shape[1] == count and decoder.seen.covariances.shape[1] == count
    assert out.field.means3d.shape == (1, count, 3) and out.field.scales.shape == (1, count, 3)
    assert torch.equal(out.source_index, torch.arange(count)[None])  # no right-grid entries
    # geometry and the native two-camera render call are unchanged
    torch.testing.assert_close(out.disparity, both(sample).disparity)
    assert decoder.poses.shape[1] == 2
    # the left Gaussians are exactly the first half of the native ones
    assert torch.equal(decoder.seen.means, encoder.gaussians.means[:, :count])
