"""Focused A03 CPU contracts and analytic fixtures."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.contracts.deployment import RectifiedStereoCamera
from reliable_endo_gs.gaussians import (
    GeometryPriorPolicy,
    SamplingPolicy,
    build_geometry_prior,
    build_surface_covariance,
    matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    sample_grid,
)
from reliable_endo_gs.geometry.surface import SurfaceFramePolicy, estimate_surface_frame


def _camera(dtype: torch.dtype = torch.float64) -> RectifiedStereoCamera:
    left = torch.tensor(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]], dtype=dtype
    ).unsqueeze(0)
    right = left.clone()
    right[:, 0, 2] -= 3.0
    transform = torch.eye(4, dtype=dtype).unsqueeze(0)
    transform[:, 0, 3] = -0.1
    return RectifiedStereoCamera(left, right, transform, True)


def _plane(height: int = 5, width: int = 5, *, slope: float = 0.0):
    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    z = 2.0 / (1.0 - slope * (u - 2.0) / 100.0)
    points = torch.stack(((u - 2.0) * z / 100.0, (v - 2.0) * z / 100.0, z), dim=-1)
    return points.unsqueeze(0)


def test_frontoparallel_and_slanted_planes_have_oriented_right_handed_frames():
    for slope in (0.0, 0.25):
        points = _plane(slope=slope)
        result = estimate_surface_frame(points, torch.ones(1, 5, 5, dtype=torch.bool))
        frame = result.frame[0, 2, 2]
        assert result.valid_mask.all()
        assert not result.fallback_mask[0, 2, 2]
        assert torch.linalg.det(frame) > 0
        assert torch.dot(frame[:, 2], points[0, 2, 2]) < 0
        torch.testing.assert_close(torch.linalg.cross(frame[:, 0], frame[:, 1]), frame[:, 2])


def test_surface_policy_requires_positive_bounded_scales():
    with pytest.raises(ValueError, match="min_tangent_scale_m"):
        SurfaceFramePolicy(min_tangent_scale_m=0.0)
    with pytest.raises(ValueError, match="max_tangent_scale_m"):
        SurfaceFramePolicy(max_tangent_scale_m=0.0)
    with pytest.raises(ValueError, match="max_normal_thickness_m"):
        SurfaceFramePolicy(max_normal_thickness_m=0.0)
    with pytest.raises(ValueError, match="<= max_normal_thickness_m"):
        SurfaceFramePolicy(normal_thickness_m=0.3, max_normal_thickness_m=0.2)

    policy = SurfaceFramePolicy(normal_thickness_m=0.2, max_normal_thickness_m=0.2)
    result = estimate_surface_frame(_plane(), torch.ones(1, 5, 5, dtype=torch.bool), policy=policy)
    assert torch.all(result.scales > 0)
    assert bool((result.scales[..., 2] <= policy.max_normal_thickness_m).all())

    narrow_policy = SurfaceFramePolicy(
        normal_thickness_m=1.0e-4, max_normal_thickness_m=2.0e-4, fallback_scale_m=0.1
    )
    fallback = estimate_surface_frame(
        _plane(1, 1), torch.ones(1, 1, 1, dtype=torch.bool), policy=narrow_policy
    )
    assert torch.all(fallback.scales > 0)
    assert bool((fallback.scales <= narrow_policy.max_normal_thickness_m).all())


def test_ambiguous_layout_requires_explicit_layout_and_nonpositive_depth_is_invalid():
    points = torch.zeros(1, 3, 3, 3, dtype=torch.float64)
    points[..., 2] = 2.0
    points[:, 2, :, :] = 2.0
    valid = torch.ones(1, 3, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="ambiguous"):
        estimate_surface_frame(points, valid)
    for layout in ("channel_first", "channel_last"):
        result = estimate_surface_frame(points, valid, layout=layout)
        assert result.valid_mask.all()

    points[..., 2] = -1.0
    result = estimate_surface_frame(points, valid, layout="channel_last")
    assert not result.valid_mask.any()
    assert torch.isfinite(result.scales).all()


def test_boundary_invalid_and_discontinuous_neighbourhoods_use_flagged_fallback():
    points = _plane()
    valid = torch.ones(1, 5, 5, dtype=torch.bool)
    valid[:, 2, 2] = False
    result = estimate_surface_frame(points, valid)
    assert not result.valid_mask[0, 2, 2]
    assert result.selected_pair[0, 2, 2].tolist() == [-1, -1]
    corner = estimate_surface_frame(points[:, :1, :1], torch.ones(1, 1, 1, dtype=torch.bool))
    assert corner.fallback_mask[0, 0, 0]

    discontinuous = points.clone()
    discontinuous[:, 1:4, 1:4, 2] += 1.0
    discontinuous[:, 2, 2, 2] = points[:, 2, 2, 2]
    result = estimate_surface_frame(
        discontinuous,
        torch.ones_like(valid),
        policy=SurfaceFramePolicy(depth_discontinuity_abs_m=0.001, depth_discontinuity_rel=0.0),
    )
    assert result.discontinuity_mask[0, 2, 2]
    assert result.fallback_mask[0, 2, 2]
    assert torch.isfinite(result.scales).all()


def test_curved_surface_and_metric_footprint_are_supported():
    v, u = torch.meshgrid(
        torch.arange(7, dtype=torch.float64),
        torch.arange(7, dtype=torch.float64),
        indexing="ij",
    )
    z = 2.0 + 0.002 * ((u - 3.0).square() + (v - 3.0).square())
    curved = torch.stack(((u - 3.0) * z / 100.0, (v - 3.0) * z / 100.0, z), dim=-1)
    curved_result = estimate_surface_frame(
        curved.unsqueeze(0), torch.ones(1, 7, 7, dtype=torch.bool)
    )
    assert curved_result.normal_valid_mask[0, 3, 3]
    assert torch.isfinite(curved_result.scales).all()
    assert torch.all(curved_result.scales > 0)

    def metric_plane(focal_length: float) -> torch.Tensor:
        z = torch.full((5, 5), 2.0, dtype=torch.float64)
        v, u = torch.meshgrid(
            torch.arange(5, dtype=torch.float64),
            torch.arange(5, dtype=torch.float64),
            indexing="ij",
        )
        return torch.stack(
            ((u - 2.0) * z / focal_length, (v - 2.0) * z / focal_length, z), dim=-1
        ).unsqueeze(0)

    footprint_policy = SurfaceFramePolicy(min_tangent_scale_m=1.0e-6, max_tangent_scale_m=1.0)
    low_focal = estimate_surface_frame(
        metric_plane(100.0), torch.ones(1, 5, 5, dtype=torch.bool), policy=footprint_policy
    )
    high_focal = estimate_surface_frame(
        metric_plane(200.0), torch.ones(1, 5, 5, dtype=torch.bool), policy=footprint_policy
    )
    torch.testing.assert_close(low_focal.scales[0, 2, 2, :2], 2.0 * high_focal.scales[0, 2, 2, :2])


def test_least_aligned_axis_gauge_and_covariance_factor_equality():
    # A plane normal parallel to camera x makes the camera-x tangent branch
    # degenerate.  The fixed least-aligned-axis fallback remains orthonormal.
    normal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    points = torch.zeros(1, 4, 4, 3, dtype=torch.float64)
    points[..., 0] = 2.0
    points[..., 1] = torch.arange(4, dtype=torch.float64).view(1, 4, 1)
    points[..., 2] = 2.0 + 0.01 * torch.arange(4, dtype=torch.float64).view(1, 1, 4)
    result = estimate_surface_frame(points, torch.ones(1, 4, 4, dtype=torch.bool))
    assert result.tangent_fallback_mask[0, 2, 2]
    torch.testing.assert_close(
        result.frame[0, 2, 2].T @ result.frame[0, 2, 2], torch.eye(3, dtype=torch.float64)
    )
    torch.testing.assert_close(result.frame[0, 2, 2][:, 2], normal.neg())

    q = matrix_to_quaternion(result.frame[0, 2, 2])
    torch.testing.assert_close(torch.linalg.vector_norm(q), torch.tensor(1.0, dtype=q.dtype))
    torch.testing.assert_close(quaternion_to_rotation_matrix(q), result.frame[0, 2, 2])
    scales = result.scales[0, 2, 2].view(1, 1, 3)
    covariance = build_surface_covariance(q.view(1, 1, 4), scales).cov_surface
    expected = result.frame[0, 2, 2] @ torch.diag(scales[0, 0].square()) @ result.frame[0, 2, 2].T
    torch.testing.assert_close(covariance[0, 0], expected)


def test_sampling_modes_are_deterministic_capped_and_corresponding():
    valid = torch.ones(2, 4, 5, dtype=torch.bool)
    valid[0, 1, 1] = False
    for mode in ("dense", "fixed", "sparse"):
        first = sample_grid(valid, policy=SamplingPolicy(mode=mode, stride=2, cap=3))
        second = sample_grid(valid, policy=SamplingPolicy(mode=mode, stride=2, cap=3))
        assert torch.equal(first.source_index, second.source_index)
        assert bool((first.counts <= 3).all())
        assert torch.equal(first.coverage.sum(dim=(-1, -2)).flatten(), first.counts)
        if first.source_index.shape[1]:
            u = first.source_index.remainder(5)
            v = first.source_index // 5
            torch.testing.assert_close(first.source_xy[..., 0].to(torch.long), u)
            torch.testing.assert_close(first.source_xy[..., 1].to(torch.long), v)
    dense = sample_grid(
        torch.ones(1, 3, 4, dtype=torch.bool), policy=SamplingPolicy(mode="dense", stride=2)
    )
    assert dense.counts.tolist() == [12]


def test_geometry_prior_keeps_rgb_correspondence_alpha_and_gradients():
    camera = _camera()
    disparity = torch.full((1, 1, 5, 5), 8.0, dtype=torch.float64, requires_grad=True)
    rgb = torch.arange(75, dtype=torch.float64).reshape(1, 3, 5, 5) / 75.0
    features = torch.arange(50, dtype=torch.float64).reshape(1, 2, 5, 5)
    policy = GeometryPriorPolicy(sampling=SamplingPolicy(mode="fixed", stride=2, cap=7))
    prior = build_geometry_prior(
        disparity,
        camera,
        torch.ones_like(disparity, dtype=torch.bool),
        rgb,
        policy=policy,
        source_features=features,
    )
    assert prior.mu.shape == (1, 7, 3)
    assert prior.valid_mask.all()
    assert torch.equal(prior.alpha0, prior.coverage)
    assert torch.all((prior.rgb >= 0) & (prior.rgb <= 1))
    assert prior.source_features is not None
    for index in range(prior.mu.shape[1]):
        linear = int(prior.source_index[0, index])
        v, u = divmod(linear, 5)
        torch.testing.assert_close(prior.rgb[0, index], rgb[0, :, v, u])
        torch.testing.assert_close(prior.source_features[0, index], features[0, :, v, u])
    loss = prior.mu.square().sum() + prior.S0.sum() + prior.cov_surface.sum()
    loss.backward()
    assert disparity.grad is not None and torch.isfinite(disparity.grad).all()
    assert bool((disparity.grad != 0).any())


def test_invalid_mask_boundary_stops_geometry_gradient():
    camera = _camera()
    disparity = torch.full((1, 1, 5, 5), 8.0, dtype=torch.float64, requires_grad=True)
    support = torch.ones_like(disparity, dtype=torch.bool)
    support[:, :, 2, 2] = False
    prior = build_geometry_prior(disparity, camera, support)
    loss = prior.mu.square().sum() + prior.S0.sum() + prior.cov_surface.sum()
    loss.backward()
    assert disparity.grad is not None
    assert disparity.grad[0, 0, 2, 2] == 0
    assert bool((disparity.grad != 0).any())


def test_invalid_geometry_is_padded_finite_and_omitted():
    camera = _camera()
    disparity = torch.full((1, 1, 3, 4), 8.0, dtype=torch.float64)
    support = torch.zeros_like(disparity, dtype=torch.bool)
    prior = build_geometry_prior(disparity, camera, support)
    assert prior.valid_mask.shape == (1, 0)
    assert prior.mu.shape == (1, 0, 3)
    assert torch.isfinite(prior.mu).all()
    assert prior.diagnostics.valid_count == 0


def test_a03_package_does_not_import_historical_modules():
    root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc, sys
blocked = ("reliable_endo_gs.baseline", "reliable_endo_gs.probabilistic_gs", "lib", "third_party")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("forbidden import: " + fullname)
        return None
sys.meta_path.insert(0, Block())
from reliable_endo_gs.gaussians import GeometryPriorPolicy, SamplingPolicy
from reliable_endo_gs.geometry.surface import SurfaceFramePolicy
assert GeometryPriorPolicy(sampling=SamplingPolicy(cap=2), surface=SurfaceFramePolicy())
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=root, check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize("bad", [0, -1])
def test_sampling_rejects_nonpositive_caps(bad):
    with pytest.raises(ValueError):
        SamplingPolicy(cap=bad)
