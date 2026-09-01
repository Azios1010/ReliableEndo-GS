"""Float64 analytic tests for canonical disparity/depth ownership."""

import pytest
import torch

from reliable_endo_gs.geometry import (
    GeometryConvention,
    depth_jacobian_wrt_disparity,
    depth_to_disparity,
    depth_uncertainty_from_disparity,
    disparity_to_depth,
)


def _convention() -> GeometryConvention:
    return GeometryConvention()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    disparity = torch.tensor([[[[2.0, 4.0]]]], dtype=torch.float64)
    focal_length_x = torch.tensor([100.0], dtype=torch.float64)
    baseline = torch.tensor([0.01], dtype=torch.float64)
    valid_mask = torch.ones_like(disparity, dtype=torch.bool)
    return disparity, focal_length_x, baseline, valid_mask


def test_known_disparity_depth_pairs_and_inverse_are_float64() -> None:
    disparity, focal_length_x, baseline, valid_mask = _inputs()

    depth = disparity_to_depth(
        disparity,
        focal_length_x,
        baseline,
        valid_mask,
        min_abs_disparity=0.01,
        convention=_convention(),
    )
    inverse = depth_to_disparity(
        depth.depth,
        focal_length_x,
        baseline,
        depth.valid_mask,
        min_depth=0.001,
        convention=_convention(),
    )

    torch.testing.assert_close(depth.depth, torch.tensor([[[[0.5, 0.25]]]], dtype=torch.float64))
    torch.testing.assert_close(inverse.disparity, disparity)
    assert depth.valid_mask.all()
    assert inverse.valid_mask.all()
    assert depth.diagnostics.valid_count == 2


def test_invalid_and_near_zero_disparity_are_masked_before_division() -> None:
    disparity = torch.tensor(
        [[[[2.0, 0.05, -1.0, torch.nan, torch.inf, 5.0]]]], dtype=torch.float64
    )
    focal_length_x = torch.tensor([100.0], dtype=torch.float64)
    baseline = torch.tensor([0.01], dtype=torch.float64)
    valid_mask = torch.tensor([[[[True, True, True, True, True, False]]]])

    result = disparity_to_depth(
        disparity,
        focal_length_x,
        baseline,
        valid_mask,
        min_abs_disparity=0.1,
        convention=_convention(),
    )

    assert result.valid_mask.tolist() == [[[[True, False, False, False, False, False]]]]
    assert torch.isfinite(result.depth[0, 0, 0, 0])
    assert torch.isnan(result.depth[~result.valid_mask]).all()
    assert result.diagnostics.input_masked_count == 1
    assert result.diagnostics.below_threshold_count == 1
    assert result.diagnostics.wrong_sign_count == 1
    assert result.diagnostics.nonfinite_input_count == 2
    assert result.diagnostics.nonfinite_output_count == 0


def test_analytic_depth_jacobian_and_uncertainty_scaling() -> None:
    disparity, focal_length_x, baseline, valid_mask = _inputs()
    sigma_d = torch.tensor([[[[0.1, 0.2]]]], dtype=torch.float64)

    jacobian = depth_jacobian_wrt_disparity(
        disparity,
        focal_length_x,
        baseline,
        valid_mask,
        min_abs_disparity=0.01,
        convention=_convention(),
    )
    uncertainty = depth_uncertainty_from_disparity(
        disparity,
        sigma_d,
        focal_length_x,
        baseline,
        valid_mask,
        min_abs_disparity=0.01,
        convention=_convention(),
    )

    torch.testing.assert_close(
        jacobian.jacobian, torch.tensor([[[[-0.25, -0.0625]]]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        uncertainty.sigma_depth, torch.tensor([[[[0.025, 0.0125]]]], dtype=torch.float64)
    )
    assert uncertainty.diagnostics.valid_count == 2


@pytest.mark.parametrize(
    ("sigma_value", "diagnostic"),
    [
        (0.0, "zero_sigma_count"),
        (-0.1, "negative_sigma_count"),
        (float("nan"), "nonfinite_sigma_count"),
        (float("inf"), "nonfinite_sigma_count"),
    ],
)
def test_sigma_d_requires_strictly_positive_finite_values_where_valid(
    sigma_value: float, diagnostic: str
) -> None:
    disparity, focal_length_x, baseline, valid_mask = _inputs()
    sigma_d = torch.full_like(disparity, sigma_value)

    result = depth_uncertainty_from_disparity(
        disparity,
        sigma_d,
        focal_length_x,
        baseline,
        valid_mask,
        min_abs_disparity=0.01,
        convention=_convention(),
    )

    assert not result.valid_mask.any()
    assert torch.isnan(result.sigma_depth).all()
    assert getattr(result.diagnostics, diagnostic) == 2


def test_invalid_sigma_is_masked_without_diagnostics_for_previously_masked_pixels() -> None:
    disparity, focal_length_x, baseline, _ = _inputs()
    valid_mask = torch.tensor([[[[True, False]]]])
    sigma_d = torch.tensor([[[[0.1, float("nan")]]]], dtype=torch.float64)

    result = depth_uncertainty_from_disparity(
        disparity,
        sigma_d,
        focal_length_x,
        baseline,
        valid_mask,
        min_abs_disparity=0.01,
        convention=_convention(),
    )

    assert result.valid_mask.tolist() == [[[[True, False]]]]
    assert torch.isnan(result.sigma_depth[~result.valid_mask]).all()
    assert result.diagnostics.nonfinite_sigma_count == 0


def test_float32_overflow_is_masked_and_reported() -> None:
    disparity = torch.tensor([[[[1.0e-20]]]], dtype=torch.float32)
    focal_length_x = torch.tensor([1.0], dtype=torch.float32)
    baseline = torch.tensor([1.0], dtype=torch.float32)
    valid_mask = torch.ones_like(disparity, dtype=torch.bool)

    result = depth_jacobian_wrt_disparity(
        disparity,
        focal_length_x,
        baseline,
        valid_mask,
        min_abs_disparity=0.0,
        convention=_convention(),
    )

    assert not result.valid_mask.any()
    assert torch.isnan(result.jacobian).all()
    assert result.diagnostics.nonfinite_output_count == 1


def test_geometry_convention_rejects_implicit_conversions() -> None:
    with pytest.raises(ValueError, match="left_camera"):
        GeometryConvention(frame="world")
    with pytest.raises(ValueError, match="unit conversion"):
        GeometryConvention(length_unit="mm")
    with pytest.raises(ValueError, match="integer_pixel_centers"):
        GeometryConvention(pixel_coordinate_convention="half_pixel")
