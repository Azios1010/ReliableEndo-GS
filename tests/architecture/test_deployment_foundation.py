"""A00 analytic CPU acceptance; no datasets, checkpoints or GPU required."""

import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.config.deployment import DeploymentRuntimeProfile, load_deployment_profile
from reliable_endo_gs.contracts.cameras import CameraBatch
from reliable_endo_gs.contracts.deployment import (
    DeploymentStereoBatch,
    RectifiedStereoCamera,
    deployment_from_stereo,
)
from reliable_endo_gs.contracts.gaussians import GaussianField
from reliable_endo_gs.contracts.samples import StereoBatch
from reliable_endo_gs.geometry.rectified import (
    crop_resize_camera,
    reconstruct_rectified,
    resized_raw_disparity,
)
from reliable_endo_gs.rendering.cross_view import ReferenceRenderer
from reliable_endo_gs.rendering.field import ExistingRendererBridge, FieldRenderRequest


def camera(batch: int = 1) -> RectifiedStereoCamera:
    k = torch.tensor([[100.0, 0, 2], [0, 80, 1], [0, 0, 1]], dtype=torch.float64)
    left = k.repeat(batch, 1, 1)
    right = left.clone()
    right[:, 0, 2] -= 3
    t = torch.eye(4, dtype=k.dtype).repeat(batch, 1, 1)
    t[:, 0, 3] = -0.1
    return RectifiedStereoCamera(left, right, t, True)


def reconstruct(d: torch.Tensor, c: RectifiedStereoCamera):
    return reconstruct_rectified(
        d, c, torch.ones_like(d, dtype=torch.bool), min_effective_disparity=1e-6
    )


def test_deployment_adapter_excludes_all_privileged_fields():
    c = camera()
    left_pose = torch.eye(4, dtype=c.K_L.dtype)[None]
    right_pose = torch.linalg.inv(c.T_R_from_L)
    rgb = torch.zeros(1, 3, 3, 5, dtype=c.K_L.dtype)
    gt = torch.full((1, 1, 3, 5), 999.0, dtype=rgb.dtype)
    sample = StereoBatch(
        rgb,
        rgb.clone(),
        CameraBatch(c.K_L, left_pose),
        CameraBatch(c.K_R, right_pose),
        ["s"],
        ["seq"],
        gt_depth=gt,
        masks={"gt_valid": torch.zeros_like(gt, dtype=torch.bool)},
        metadata={"future": gt},
    )
    deployed = deployment_from_stereo(sample, rectified=True)
    assert {f.name for f in fields(deployed)} == {"left", "right", "camera", "support"}
    assert deployed.support.all()
    torch.testing.assert_close(deployed.camera.T_R_from_L, c.T_R_from_L)
    changed = replace(sample, gt_depth=gt * 3, masks={})
    again = deployment_from_stereo(changed, rectified=True)
    torch.testing.assert_close(again.support, deployed.support)
    assert not hasattr(deployed, "gt_depth")
    assert not hasattr(deployed, "metadata")


@pytest.mark.parametrize(
    "fault", ["rotation", "baseline", "focal", "scanline", "skew", "nan", "uncertified"]
)
def test_bad_camera_rejected(fault):
    c = camera()
    left_k, r, t = c.K_L.clone(), c.K_R.clone(), c.T_R_from_L.clone()
    if fault == "rotation":
        t[:, 0, 0] = 0.9
    elif fault == "baseline":
        t[:, 0, 3] = 0.1
    elif fault == "focal":
        r[:, 0, 0] += 1
    elif fault == "scanline":
        r[:, 1, 2] += 1
    elif fault == "skew":
        left_k[:, 0, 1] = 0.1
    elif fault == "nan":
        left_k[:, 0, 2] = float("nan")
    with pytest.raises(ValueError):
        RectifiedStereoCamera(left_k, r, t, fault != "uncertified")


@pytest.mark.parametrize("slope", [0.0, 0.3])
def test_metric_plane_projection_and_gradients(slope):
    c = camera()
    u = torch.arange(5, dtype=torch.float64).view(1, 1, 1, 5).expand(1, 1, 3, 5)
    # Plane z = 2 + slope*x; x = (u-cx)*z/fx.
    expected_z = 2 / (1 - slope * (u - 2) / 100)
    d = (10 / expected_z + 3).detach().requires_grad_()
    result = reconstruct(d, c)
    torch.testing.assert_close(result.depth.depth, expected_z)
    xyz = result.points.centers
    torch.testing.assert_close(xyz[:, 2:3], 2 + slope * xyz[:, :1])
    u_right = 100 * (xyz[:, :1] - 0.1) / xyz[:, 2:3] - 1
    torch.testing.assert_close(u - u_right, d)
    result.points.centers.sum().backward()
    assert torch.isfinite(d.grad).all() and (d.grad != 0).all()


def test_crop_resize_preserves_metric_depth_and_projected_coordinates():
    c = camera()
    resized = crop_resize_camera(
        c, left_crop_xy=(2, 1), right_crop_xy=(1, 1), crop_hw=(4, 6), output_hw=(8, 9)
    )
    d = resized_raw_disparity(
        torch.full((1, 1, 8, 9), 8.0, dtype=torch.float64),
        left_crop_x=2,
        right_crop_x=1,
        crop_width=6,
        output_width=9,
    )
    result = reconstruct(d, resized)
    torch.testing.assert_close(result.depth.depth, torch.full_like(d, 2))
    assert resized.K_L[0, 0, 2] == 0.25
    assert resized.K_L[0, 1, 2] == 0.5
    with pytest.raises(ValueError, match="scanline"):
        crop_resize_camera(
            c, left_crop_xy=(0, 1), right_crop_xy=(0, 0), crop_hw=(4, 6), output_hw=(8, 9)
        )


def test_invalid_disparity_and_support_are_not_clamped_to_geometry():
    c = camera()
    d = torch.tensor([[[[3.0, 2.0, float("nan"), float("inf"), 8.0, 8.0]]]], dtype=torch.float64)
    support = torch.ones_like(d, dtype=torch.bool)
    support[..., -1] = False
    result = reconstruct_rectified(d, c, support, min_effective_disparity=1e-6)
    assert result.depth.valid_mask.flatten().tolist() == [False, False, False, False, True, False]
    assert result.depth.depth.flatten()[4] == 2
    assert torch.isnan(result.depth.depth[~result.depth.valid_mask]).all()


def test_batch_isolation_and_fp32_geometry_island():
    c = camera(2)
    t = c.T_R_from_L.clone()
    t[1, 0, 3] = -0.2
    c = replace(c, T_R_from_L=t)
    d = torch.tensor([8.0, 13.0], dtype=torch.float64).view(2, 1, 1, 1)
    both = reconstruct(d, c)
    for i in range(2):
        single = RectifiedStereoCamera(
            c.K_L[i : i + 1], c.K_R[i : i + 1], c.T_R_from_L[i : i + 1], True
        )
        torch.testing.assert_close(
            both.depth.depth[i : i + 1], reconstruct(d[i : i + 1], single).depth.depth
        )
    c32 = RectifiedStereoCamera(c.K_L.float(), c.K_R.float(), c.T_R_from_L.float(), True)
    assert reconstruct(d.to(torch.bfloat16), c32).depth.depth.dtype == torch.float32


def field() -> GaussianField:
    return GaussianField(
        torch.tensor([[[0.0, 0, 2], [float("nan"), 0, 0]]], dtype=torch.float64),
        torch.ones(1, 2, 3, dtype=torch.float64),
        torch.tensor([[[1.0, 0, 0, 0], [0.0, 0, 0, 0]]], dtype=torch.float64),
        torch.ones(1, 2, 3, dtype=torch.float64) * 0.01,
        torch.ones(1, 2, 1, dtype=torch.float64) * 0.5,
        torch.tensor([[True, False]]),
    )


@pytest.mark.parametrize("view", ["left", "right"])
def test_independent_field_reference_render(view):
    f = field()
    output = ExistingRendererBridge(ReferenceRenderer()).render(
        FieldRenderRequest(f, camera(), (3, 5), view)
    )
    assert output.image.shape == (1, 3, 3, 5)
    assert torch.isfinite(output.image).all()
    assert torch.isnan(f.means3d[0, 1, 0])  # caller field not mutated
    torch.testing.assert_close(output.depth, torch.full_like(output.depth, 2))


def test_field_rejects_bad_scales_and_ambiguous_covariance():
    f = field()
    with pytest.raises(ValueError, match="positive"):
        FieldRenderRequest(replace(f, scales=-f.scales), camera(), (3, 5))
    with pytest.raises(ValueError, match="ambiguous"):
        FieldRenderRequest(
            replace(f, cov_surface=torch.eye(3).double().repeat(1, 2, 1, 1)), camera(), (3, 5)
        )


def test_runtime_profile_has_bounded_server_defaults_and_is_configurable():
    root = Path(__file__).resolve().parents[2]
    p = load_deployment_profile(root / "configs/reconstruction/runtime.yaml")
    assert p.batch_size == 1 and p.memory_gib == 16
    assert p.resolution_hw == (512, 640)
    assert p.deadline_ms == 100.0
    p.require_benchmark_ready()
    replace(
        p, hardware="test CPU", device="cpu", precision="bf16", resolution_hw=(3, 5), deadline_ms=30
    ).require_benchmark_ready()
    with pytest.raises(ValueError):
        DeploymentRuntimeProfile(precision="fp64")


def test_import_and_execution_with_forbidden_modules_blocked():
    # A fresh process catches transitive imports concealed by pytest's module cache.
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lib" or fullname.startswith(("lib.", "reliable_endo_gs.baseline", "third_party")):
            raise AssertionError("forbidden import: " + fullname)
sys.meta_path.insert(0, Block())
import runpy
t = runpy.run_path("tests/architecture/test_deployment_foundation.py")
t["test_metric_plane_projection_and_gradients"](0.3)
t["test_independent_field_reference_render"]("right")
t["test_deployment_adapter_excludes_all_privileged_fields"]()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )


def test_deployment_rejects_non_rgb_range():
    rgb = torch.full((1, 3, 3, 5), 255.0)
    with pytest.raises(ValueError, match="RGB"):
        DeploymentStereoBatch(rgb, rgb, camera(), torch.ones(1, 1, 3, 5, dtype=torch.bool))


def test_relative_camera_independent_of_world_frame():
    c = camera()
    world_left = torch.tensor(
        [[[0.0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]]],
        dtype=torch.float64,
    )
    world_right = world_left @ torch.linalg.inv(c.T_R_from_L)
    relative = RectifiedStereoCamera.from_cameras(
        CameraBatch(c.K_L, world_left), CameraBatch(c.K_R, world_right), rectified=True
    )
    torch.testing.assert_close(relative.T_R_from_L, c.T_R_from_L)


def test_all_invalid_field_returns_finite_background():
    f = field()
    f = replace(f, valid_mask=torch.zeros_like(f.valid_mask))
    result = ExistingRendererBridge(ReferenceRenderer()).render(
        FieldRenderRequest(f, camera(), (3, 5))
    )
    assert torch.isfinite(result.image).all()
    assert (result.image == 0).all()
