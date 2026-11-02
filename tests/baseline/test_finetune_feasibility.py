"""CPU probes of train isolation, true stereo depth and differentiable RGB."""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image
from torch import nn

from reliable_endo_gs.baseline.finetune_feasibility import (
    TrainingDepthStatistics,
    both_view_mse,
    posed_training_rgb,
    splatt3r_training_rgb,
    splatt3r_views,
    training_depth_sample,
)
from reliable_endo_gs.baseline.posed_gaussians import PosedGaussianConfig, PosedGaussianPredictor
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera
from reliable_endo_gs.data.scared_a00 import (
    A01Manifest,
    A01ManifestEntry,
    ScaredA00Adapter,
    ScaredA00Sample,
    ScaredSupervision,
)
from reliable_endo_gs.teachers.roles import DEFAULT_A01_SCARED_ROLES


def sample() -> ScaredA00Sample:
    """Synthetic [1,3,2,4], fx=100, B=0.02 m and cx_L-cx_R=2 px."""
    k = torch.tensor([[[100.0, 0, 2.0], [0, 100.0, 1.0], [0, 0, 1.0]]])
    right_k = k.clone()
    right_k[:, 0, 2] = 0
    t = torch.eye(4)[None]
    t[:, 0, 3] = -0.02
    batch = DeploymentStereoBatch(
        torch.zeros(1, 3, 2, 4),
        torch.ones(1, 3, 2, 4),
        RectifiedStereoCamera(k, right_k, t, True),
        torch.ones(1, 1, 2, 4, dtype=torch.bool),
    )
    truth = ScaredSupervision(torch.full((1, 1, 2, 4), 12.0), batch.support.clone())
    return ScaredA00Sample("dataset_1/keyframe_1/frame_data000000", "final_train", batch, {}, truth)


def test_training_statistics_use_optical_z_and_principal_offset():
    stats = TrainingDepthStatistics(3)
    stats.update(sample())
    got = stats.result()
    assert got["frames"] == 1 and got["sampled_pixels"] == 3
    assert got["q005_m"] == pytest.approx(0.2)
    assert got["near_m"] == pytest.approx(0.18)
    assert got["far_m"] == pytest.approx(0.22)
    with pytest.raises(ValueError, match="duplicate"):
        stats.update(sample())


@pytest.mark.parametrize("dataset", [4, 5, 7, 8, 9])
def test_nontraining_identifiers_are_rejected(dataset: int):
    s = replace(sample(), sample_id=f"dataset_{dataset}/keyframe_1/frame_data000000")
    with pytest.raises(ValueError, match="train_1236"):
        TrainingDepthStatistics().update(s)


def test_depth_population_follows_valid_support_not_rgb():
    s = sample()
    assert s.supervision is not None
    s.supervision.disparity[0, 0, 0, 0] = 2.0  # zero effective disparity
    s.supervision.valid_mask[0, 0, 0, 1] = False
    s.batch.support[0, 0, 0, 2] = False
    stats = TrainingDepthStatistics()
    stats.update(s)
    assert stats.result()["valid_pixels"] == 5
    with pytest.raises(ValueError, match="separate supervision"):
        stats.update(
            replace(s, sample_id="dataset_2/keyframe_1/frame_data000000", supervision=None)
        )


def test_rgb_loss_includes_both_cameras_and_preserves_gradients():
    s = sample()
    rgb = torch.zeros(1, 2, 3, 2, 4, requires_grad=True)
    loss = both_view_mse(rgb, s.batch)
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert rgb.grad is not None and rgb.grad[:, 1].abs().sum() > 0
    assert rgb.grad[:, 0].abs().sum() == 0
    with pytest.raises(ValueError, match="full input grids"):
        both_view_mse(rgb[..., :3], s.batch)


@dataclass
class Gaussians:
    means: torch.Tensor
    covariances: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, context: dict, step: int, deterministic: bool):
        assert not deterministic and step == 300001
        return Gaussians(*(self.weight[None] for _ in range(4)))


class Decoder(nn.Module):
    def forward(self, gaussians, extrinsics, intrinsics, near, far, hw):
        return SimpleNamespace(color=gaussians.means.expand(1, 2, 3, *hw))


def test_native_training_boundary_keeps_gaussian_gradient():
    encoder = Encoder()
    predictor = PosedGaussianPredictor(encoder, Decoder(), PosedGaussianConfig(0.1, 1.0, 300001))
    rgb = posed_training_rgb(predictor, sample().batch)
    rgb.sum().backward()
    assert encoder.weight.grad == 48


def test_splatt3r_preserves_full_rectangular_grid():
    s = sample()
    batch = replace(
        s.batch,
        left=torch.zeros(1, 3, 16, 32),
        right=torch.ones(1, 3, 16, 32),
        support=torch.ones(1, 1, 16, 32, dtype=torch.bool),
    )
    left, right = splatt3r_views(batch)
    assert left["img"].shape == (1, 3, 16, 32)
    assert left["true_shape"].tolist() == [[16, 32]]
    assert left["instance"] == ["0"] and right["instance"] == ["1"]
    assert left["img"].min() == -1 and right["img"].max() == 1
    assert np.prod(left["true_shape"].numpy()) == 512
    with pytest.raises(ValueError, match="no silent crop"):
        splatt3r_views(s.batch)


def test_statistics_loader_matches_full_a00_camera_and_supervision(tmp_path: Path):
    s = sample()
    data = tmp_path / "dataset_1/keyframe_1/data"
    for folder in ("left_finalpass", "right_finalpass", "frame_data", "newpram_data", "disparity"):
        (data / folder).mkdir(parents=True)
    for side in ("left", "right"):
        Image.fromarray(np.full((2, 4, 3), 128, np.uint8)).save(
            data / f"{side}_finalpass/frame_data000000.png"
        )
    (data / "frame_data/frame_data000000.json").write_text("{}")
    camera = s.batch.camera
    (data / "newpram_data/frame_data000000.json").write_text(
        json.dumps(
            {
                "intr0": camera.K_L[0].tolist(),
                "intr1": camera.K_R[0].tolist(),
                "T_R_from_L": camera.T_R_from_L[0].tolist(),
            }
        )
    )
    (data / "a00_protocol.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_range": "uint8_0_255",
                "rectified": True,
                "distortion_removed": True,
                "pixel_center": "integer",
                "disparity": {"units": "pixels", "sign": "u_left_minus_right", "reference": "left"},
                "extrinsics": {
                    "source": "newpram.T_R_from_L",
                    "direction": "right_from_left",
                    "units": "m",
                },
                "preprocessing": {"images_grid": "processed"},
            }
        )
    )
    tifffile.imwrite(data / "disparity/frame_data000000.tiff", np.full((2, 4), 12.0, np.float32))
    manifest = A01Manifest(
        DEFAULT_A01_SCARED_ROLES.role_hash, (A01ManifestEntry(s.sample_id, "final_train"),)
    )
    adapter = ScaredA00Adapter(tmp_path, manifest)
    fast = training_depth_sample(adapter, s.sample_id, (4, 8))
    full = adapter.load(
        s.sample_id, data_role="final_train", include_supervision=True, output_hw=(4, 8)
    )
    torch.testing.assert_close(fast.batch.camera.K_L, full.batch.camera.K_L)
    torch.testing.assert_close(fast.batch.camera.K_R, full.batch.camera.K_R)
    torch.testing.assert_close(fast.batch.camera.T_R_from_L, full.batch.camera.T_R_from_L)
    assert fast.supervision is not None and full.supervision is not None
    torch.testing.assert_close(fast.supervision.disparity, full.supervision.disparity)
    assert torch.equal(fast.supervision.valid_mask, full.supervision.valid_mask)
    with pytest.raises(ValueError, match="train_1236"):
        training_depth_sample(adapter, "dataset_8/keyframe_1/frame_data000000", (4, 8))


def test_splatt_render_boundary_preserves_rig_bounds_and_gradients(monkeypatch):
    s = sample()
    b = replace(
        s.batch,
        left=torch.zeros(1, 3, 16, 32),
        right=torch.ones(1, 3, 16, 32),
        support=torch.ones(1, 1, 16, 32, dtype=torch.bool),
    )
    value = nn.Parameter(torch.tensor(0.2))

    def encoder(*views):
        return tuple(
            {
                "means": value.expand(1, 16, 32, 3),
                "scales": torch.ones(1, 16, 32, 3),
                "rotations": torch.ones(1, 16, 32, 4),
                "sh": torch.zeros(1, 16, 32, 3, 1),
                "opacities": torch.ones(1, 16, 32, 1),
            }
            for _ in views
        )

    def render(poses, k, near, far, hw, background, means, cov, sh, opacity):
        assert hw == (16, 32) and means.shape == (2, 1024, 3)
        torch.testing.assert_close(near, torch.full((2,), 0.01))
        torch.testing.assert_close(far, torch.full((2,), 1.0))
        assert poses[1, 0, 3] == pytest.approx(0.02)
        # SH's DC colour residual includes both source RGB views.
        assert sh[:, :512].sum() == 0 and sh[:, 512:].sum() > 0
        return means.mean().expand(2, 3, *hw)

    modules = {
        "utils.geometry": SimpleNamespace(
            build_covariance=lambda scale, rot: torch.eye(3).expand(1, 16, 32, 3, 3)
        ),
        "utils.sh_utils": SimpleNamespace(RGB2SH=lambda rgb: rgb),
        "src.pixelsplat_src.cuda_splatting": SimpleNamespace(render_cuda=render),
    }
    monkeypatch.setattr(
        "reliable_endo_gs.baseline.finetune_feasibility.importlib.import_module",
        modules.__getitem__,
    )
    rgb = splatt3r_training_rgb(encoder, b, PosedGaussianConfig(0.01, 1.0, 0))
    assert rgb.shape == (1, 2, 3, 16, 32)
    rgb.sum().backward()
    assert value.grad is not None and value.grad > 0
