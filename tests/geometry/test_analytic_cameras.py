"""End-to-end float64 analytic camera fixture for the geometry package."""

import torch

from reliable_endo_gs.geometry import (
    GeometryConvention,
    backproject_depth,
    center_covariance_from_disparity,
    disparity_to_depth,
    focal_length_x_from_intrinsics,
    pixel_rays,
)


def test_batched_camera_pipeline_preserves_shapes_frames_units_and_masks() -> None:
    convention = GeometryConvention()
    intrinsics = torch.tensor(
        [
            [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]],
            [[200.0, 0.0, 1.0], [0.0, 200.0, 1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    disparity = torch.tensor([[[[2.0]]], [[[4.0]]]], dtype=torch.float64)
    sigma_d = torch.tensor([[[[0.1]]], [[[0.2]]]], dtype=torch.float64)
    baseline = torch.tensor([0.01, 0.02], dtype=torch.float64)
    valid_mask = torch.ones_like(disparity, dtype=torch.bool)
    focal_length_x = focal_length_x_from_intrinsics(intrinsics)

    depth = disparity_to_depth(
        disparity,
        focal_length_x,
        baseline,
        valid_mask,
        min_abs_disparity=0.001,
        convention=convention,
    )
    rays = pixel_rays(intrinsics, height=1, width=1, convention=convention)
    centers = backproject_depth(depth.depth, intrinsics, depth.valid_mask, convention=convention)
    covariance = center_covariance_from_disparity(
        disparity,
        sigma_d,
        focal_length_x,
        baseline,
        rays.rays,
        valid_mask,
        min_abs_disparity=0.001,
        epsilon=0.0,
        max_eigenvalue=None,
        rank_tolerance=1.0e-14,
        convention=convention,
    )

    assert depth.depth.shape == (2, 1, 1, 1)
    assert centers.centers.shape == (2, 3, 1, 1)
    assert covariance.cov_center.shape == (2, 1, 1, 3, 3)
    assert centers.convention.frame == "left_camera"
    assert covariance.convention.length_unit == "m"
    torch.testing.assert_close(depth.depth.flatten(), torch.tensor([0.5, 1.0], dtype=torch.float64))
    torch.testing.assert_close(
        centers.centers[:, 2, 0, 0], torch.tensor([0.5, 1.0], dtype=torch.float64)
    )
    assert covariance.valid_mask.all()
