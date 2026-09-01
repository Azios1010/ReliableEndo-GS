"""Analytic camera-ray and backprojection tests."""

import pytest
import torch

from reliable_endo_gs.geometry import GeometryConvention, backproject_depth, pixel_grid, pixel_rays


def _intrinsics() -> torch.Tensor:
    return torch.tensor(
        [[[100.0, 0.0, 2.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )


def test_pixel_grid_and_rays_follow_integer_pixel_centres() -> None:
    convention = GeometryConvention()
    grid = pixel_grid(
        batch_size=1,
        height=4,
        width=3,
        device=torch.device("cpu"),
        dtype=torch.float64,
        convention=convention,
    )
    rays = pixel_rays(_intrinsics(), height=4, width=3, convention=convention)

    torch.testing.assert_close(grid[0, 0, 0], torch.tensor([0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(grid[0, 3, 2], torch.tensor([2.0, 3.0], dtype=torch.float64))
    torch.testing.assert_close(
        rays.rays[0, :, 3, 2], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        rays.rays[0, :, 0, 0], torch.tensor([-0.02, -0.03, 1.0], dtype=torch.float64)
    )


def test_backprojection_uses_optical_axis_depth_and_propagates_masks() -> None:
    convention = GeometryConvention()
    depth = torch.tensor([[[[2.0, torch.nan], [-1.0, 3.0]]]], dtype=torch.float64)
    valid_mask = torch.tensor([[[[True, True], [True, False]]]])

    result = backproject_depth(depth, _intrinsics(), valid_mask, convention=convention)

    assert result.centers.shape == (1, 3, 2, 2)
    assert result.valid_mask.tolist() == [[[[True, False], [False, False]]]]
    torch.testing.assert_close(
        result.centers[0, :, 0, 0], torch.tensor([-0.04, -0.06, 2.0], dtype=torch.float64)
    )
    assert torch.isnan(result.centers[:, :, 0, 1]).all()
    assert result.diagnostics.nonfinite_depth_count == 1
    assert result.diagnostics.nonpositive_depth_count == 1
    assert result.diagnostics.input_masked_count == 1
    assert result.diagnostics.nonfinite_center_count == 0


def test_off_axis_backprojection_uses_fx_and_fy_with_manual_coordinates() -> None:
    convention = GeometryConvention()
    intrinsics = torch.tensor(
        [[[200.0, 0.0, 1.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    depth = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    valid_mask = torch.ones_like(depth, dtype=torch.bool)

    result = backproject_depth(depth, intrinsics, valid_mask, convention=convention)

    # K^-1 [u, v, 1] at integer pixel centre (u, v)=(0, 0), then scaled by Z.
    torch.testing.assert_close(
        result.centers[0, :, 0, 0],
        torch.tensor([-0.01, -0.04, 2.0], dtype=torch.float64),
    )


def test_backprojection_rejects_shape_dtype_and_device_contract_violations() -> None:
    convention = GeometryConvention()
    depth = torch.ones((1, 1, 1, 1), dtype=torch.float64)
    valid_mask = torch.ones_like(depth, dtype=torch.bool)
    intrinsics = _intrinsics()

    with pytest.raises(ValueError, match="valid_mask must have shape"):
        backproject_depth(
            depth,
            intrinsics,
            torch.ones((1, 1, 1, 2), dtype=torch.bool),
            convention=convention,
        )
    with pytest.raises(TypeError, match="same dtype"):
        backproject_depth(depth.float(), intrinsics, valid_mask, convention=convention)
    with pytest.raises(TypeError, match="dtype torch.bool"):
        backproject_depth(depth, intrinsics, valid_mask.to(torch.float64), convention=convention)
