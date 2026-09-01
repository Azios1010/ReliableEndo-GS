"""Float64 Jacobian, PSD, rank, and covariance-policy tests."""

import torch

from reliable_endo_gs.geometry import (
    GeometryConvention,
    backproject_depth,
    center_covariance_from_calibrated_sigma,
    center_covariance_from_disparity,
    center_jacobian_wrt_disparity,
    disparity_to_depth,
    pixel_rays,
)
from reliable_endo_gs.uncertainty.records import CalibratedDisparitySigma


def _intrinsics() -> torch.Tensor:
    return torch.tensor(
        [[[300.0, 0.0, 0.5], [0.0, 250.0, 0.5], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )


def _geometry_inputs() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    disparity = torch.tensor([[[[8.0, 12.0], [16.0, 20.0]]]], dtype=torch.float64)
    sigma_d = torch.tensor([[[[0.2, 0.3], [0.4, 0.5]]]], dtype=torch.float64)
    focal_length_x = torch.tensor([300.0], dtype=torch.float64)
    baseline = torch.tensor([0.005], dtype=torch.float64)
    valid_mask = torch.ones_like(disparity, dtype=torch.bool)
    return disparity, sigma_d, focal_length_x, baseline, valid_mask


def _centers_from_disparity(
    disparity: torch.Tensor,
    focal_length_x: torch.Tensor,
    baseline: torch.Tensor,
    valid_mask: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    convention = GeometryConvention()
    depth = disparity_to_depth(
        disparity,
        focal_length_x,
        baseline,
        valid_mask,
        min_abs_disparity=0.01,
        convention=convention,
    )
    return backproject_depth(
        depth.depth, intrinsics, depth.valid_mask, convention=convention
    ).centers


def test_analytic_center_jacobian_matches_central_finite_difference() -> None:
    disparity, _, focal_length_x, baseline, valid_mask = _geometry_inputs()
    convention = GeometryConvention()
    intrinsics = torch.tensor(
        [[[300.0, 0.0, 1.0], [0.0, 250.0, 1.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    rays = pixel_rays(intrinsics, height=2, width=2, convention=convention)
    analytic = center_jacobian_wrt_disparity(
        disparity,
        focal_length_x,
        baseline,
        rays.rays,
        valid_mask,
        min_abs_disparity=0.01,
        convention=convention,
    )
    step = 1.0e-5
    numerical = (
        _centers_from_disparity(disparity + step, focal_length_x, baseline, valid_mask, intrinsics)
        - _centers_from_disparity(
            disparity - step, focal_length_x, baseline, valid_mask, intrinsics
        )
    ) / (2.0 * step)

    torch.testing.assert_close(analytic.jacobian, numerical, atol=1.0e-11, rtol=1.0e-9)


def test_disparity_only_covariance_is_symmetric_psd_rank_one_and_ray_aligned() -> None:
    disparity, sigma_d, focal_length_x, baseline, valid_mask = _geometry_inputs()
    convention = GeometryConvention()
    rays = pixel_rays(_intrinsics(), height=2, width=2, convention=convention)
    result = center_covariance_from_disparity(
        disparity,
        sigma_d,
        focal_length_x,
        baseline,
        rays.rays,
        valid_mask,
        min_abs_disparity=0.01,
        epsilon=0.0,
        max_eigenvalue=None,
        rank_tolerance=1.0e-14,
        convention=convention,
    )

    covariance = result.cov_center
    torch.testing.assert_close(covariance, covariance.transpose(-1, -2), atol=1.0e-14, rtol=0.0)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.reshape(-1, 3, 3))
    assert torch.all(eigenvalues >= -1.0e-14)
    assert torch.all((eigenvalues > 1.0e-14).sum(dim=-1) == 1)
    ray_vectors = rays.rays.permute(0, 2, 3, 1).reshape(-1, 3)
    principal_vectors = eigenvectors[:, :, -1]
    alignment = torch.abs(
        (
            principal_vectors
            * ray_vectors
            / torch.linalg.vector_norm(ray_vectors, dim=-1, keepdim=True)
        ).sum(dim=-1)
    )
    torch.testing.assert_close(alignment, torch.ones_like(alignment), atol=1.0e-12, rtol=0.0)
    assert result.diagnostics.psd_violation_count == 0
    assert result.diagnostics.maximum_effective_rank == 1
    assert result.diagnostics.minimum_principal_ray_alignment is not None


def test_covariance_scales_quadratically_and_reports_stabilization_and_cap() -> None:
    disparity, sigma_d, focal_length_x, baseline, valid_mask = _geometry_inputs()
    convention = GeometryConvention()
    rays = pixel_rays(_intrinsics(), height=2, width=2, convention=convention)
    base = center_covariance_from_disparity(
        disparity,
        sigma_d,
        focal_length_x,
        baseline,
        rays.rays,
        valid_mask,
        min_abs_disparity=0.01,
        epsilon=0.0,
        max_eigenvalue=None,
        rank_tolerance=1.0e-14,
        convention=convention,
    )
    scaled = center_covariance_from_disparity(
        disparity,
        2.0 * sigma_d,
        focal_length_x,
        baseline,
        rays.rays,
        valid_mask,
        min_abs_disparity=0.01,
        epsilon=0.0,
        max_eigenvalue=None,
        rank_tolerance=1.0e-14,
        convention=convention,
    )
    stabilized = center_covariance_from_disparity(
        disparity,
        sigma_d,
        focal_length_x,
        baseline,
        rays.rays,
        valid_mask,
        min_abs_disparity=0.01,
        epsilon=1.0e-8,
        max_eigenvalue=2.0e-8,
        rank_tolerance=1.0e-14,
        convention=convention,
    )

    torch.testing.assert_close(scaled.cov_center, 4.0 * base.cov_center, atol=1.0e-14, rtol=1.0e-12)
    stabilized_eigenvalues = torch.linalg.eigvalsh(stabilized.cov_center.reshape(-1, 3, 3))
    assert torch.all(stabilized_eigenvalues <= 2.0e-8 + 1.0e-18)
    assert stabilized.diagnostics.stabilization_count == 4
    assert stabilized.diagnostics.capped_count == 4


def test_covariance_baseline_x2_and_disparity_x2_have_analytic_scaling() -> None:
    disparity, sigma_d, focal_length_x, baseline, valid_mask = _geometry_inputs()
    convention = GeometryConvention()
    rays = pixel_rays(_intrinsics(), height=2, width=2, convention=convention)
    reference = center_covariance_from_disparity(
        disparity,
        sigma_d,
        focal_length_x,
        baseline,
        rays.rays,
        valid_mask,
        min_abs_disparity=0.01,
        epsilon=0.0,
        max_eigenvalue=None,
        rank_tolerance=1.0e-14,
        convention=convention,
    )
    baseline_x2 = center_covariance_from_disparity(
        disparity,
        sigma_d,
        focal_length_x,
        2.0 * baseline,
        rays.rays,
        valid_mask,
        min_abs_disparity=0.01,
        epsilon=0.0,
        max_eigenvalue=None,
        rank_tolerance=1.0e-14,
        convention=convention,
    )
    disparity_x2 = center_covariance_from_disparity(
        2.0 * disparity,
        sigma_d,
        focal_length_x,
        baseline,
        rays.rays,
        valid_mask,
        min_abs_disparity=0.01,
        epsilon=0.0,
        max_eigenvalue=None,
        rank_tolerance=1.0e-14,
        convention=convention,
    )

    torch.testing.assert_close(baseline_x2.cov_center, 4.0 * reference.cov_center)
    torch.testing.assert_close(disparity_x2.cov_center, reference.cov_center / 16.0)


def test_zero_valid_mask_returns_empty_finite_free_covariance_result() -> None:
    disparity, sigma_d, focal_length_x, baseline, _ = _geometry_inputs()
    convention = GeometryConvention()
    rays = pixel_rays(_intrinsics(), height=2, width=2, convention=convention)
    result = center_covariance_from_disparity(
        disparity,
        sigma_d,
        focal_length_x,
        baseline,
        rays.rays,
        torch.zeros_like(disparity, dtype=torch.bool),
        min_abs_disparity=0.01,
        epsilon=0.0,
        max_eigenvalue=None,
        rank_tolerance=1.0e-14,
        convention=convention,
    )

    assert result.valid_mask.shape == disparity.shape
    assert not result.valid_mask.any()
    assert torch.isnan(result.cov_center).all()
    assert result.diagnostics.valid_count == 0


def test_zero_sigma_is_invalid_for_center_covariance() -> None:
    disparity, _, focal_length_x, baseline, valid_mask = _geometry_inputs()
    sigma_d = torch.zeros_like(disparity)
    convention = GeometryConvention()
    rays = pixel_rays(_intrinsics(), height=2, width=2, convention=convention)

    result = center_covariance_from_disparity(
        disparity,
        sigma_d,
        focal_length_x,
        baseline,
        rays.rays,
        valid_mask,
        min_abs_disparity=0.01,
        epsilon=0.0,
        max_eigenvalue=None,
        rank_tolerance=1.0e-14,
        convention=convention,
    )

    assert not result.valid_mask.any()
    assert torch.isnan(result.cov_center).all()
    assert result.diagnostics.zero_sigma_count == disparity.numel()


def test_calibrated_sigma_handoff_is_provider_neutral_and_preserves_provenance() -> None:
    disparity, sigma_d, focal_length_x, baseline, valid_mask = _geometry_inputs()
    convention = GeometryConvention()
    rays = pixel_rays(_intrinsics(), height=2, width=2, convention=convention)
    calibrated = CalibratedDisparitySigma(
        sigma_d=sigma_d,
        valid_mask=valid_mask,
        provider_id="provider.proxy-or-learned",
        calibration_id="calibration.val-v1",
    )

    result = center_covariance_from_calibrated_sigma(
        disparity,
        calibrated,
        focal_length_x,
        baseline,
        rays.rays,
        valid_mask,
        min_abs_disparity=0.01,
        epsilon=0.0,
        max_eigenvalue=None,
        rank_tolerance=1.0e-14,
        convention=convention,
    )

    assert result.provenance is not None
    assert result.provenance.provider_id == "provider.proxy-or-learned"
    assert result.provenance.calibration_id == "calibration.val-v1"
    assert result.provenance.geometry_schema_version == "geometry_center_covariance.v1"
    assert result.provenance.convention == convention


def test_cpu_autograd_has_finite_nonzero_disparity_and_sigma_gradients() -> None:
    disparity = torch.tensor([[[[8.0]]]], dtype=torch.float64, requires_grad=True)
    sigma_d = torch.tensor([[[[0.2]]]], dtype=torch.float64, requires_grad=True)
    focal_length_x = torch.tensor([300.0], dtype=torch.float64)
    baseline = torch.tensor([0.005], dtype=torch.float64)
    valid_mask = torch.ones_like(disparity, dtype=torch.bool)
    convention = GeometryConvention()
    rays = pixel_rays(_intrinsics(), height=1, width=1, convention=convention)
    result = center_covariance_from_disparity(
        disparity,
        sigma_d,
        focal_length_x,
        baseline,
        rays.rays,
        valid_mask,
        min_abs_disparity=0.01,
        epsilon=0.0,
        max_eigenvalue=None,
        rank_tolerance=1.0e-14,
        convention=convention,
    )
    result.cov_center.sum().backward()

    assert disparity.grad is not None and torch.isfinite(disparity.grad).all()
    assert sigma_d.grad is not None and torch.isfinite(sigma_d.grad).all()
    assert disparity.grad.abs().sum() > 0
    assert sigma_d.grad.abs().sum() > 0


def test_invalid_disparity_and_sigma_are_masked_without_nonfinite_valid_covariances() -> None:
    disparity, sigma_d, focal_length_x, baseline, valid_mask = _geometry_inputs()
    disparity[0, 0, 0, 1] = 0.0
    sigma_d[0, 0, 1, 0] = torch.nan
    sigma_d[0, 0, 1, 1] = -0.1
    convention = GeometryConvention()
    rays = pixel_rays(_intrinsics(), height=2, width=2, convention=convention)
    result = center_covariance_from_disparity(
        disparity,
        sigma_d,
        focal_length_x,
        baseline,
        rays.rays,
        valid_mask,
        min_abs_disparity=0.01,
        epsilon=0.0,
        max_eigenvalue=None,
        rank_tolerance=1.0e-14,
        convention=convention,
    )

    assert result.valid_mask.tolist() == [[[[True, False], [False, False]]]]
    assert torch.isfinite(result.cov_center[0, 0, 0]).all()
    assert torch.isnan(result.cov_center[0, 0, 1]).all()
    assert torch.isnan(result.cov_center[0, 1, 0]).all()
    assert torch.isnan(result.cov_center[0, 1, 1]).all()
    assert result.diagnostics.nonfinite_sigma_count == 1
    assert result.diagnostics.negative_sigma_count == 1
    assert result.diagnostics.zero_sigma_count == 0
    assert result.diagnostics.nonfinite_covariance_count == 0


def test_finite_jacobian_but_overflowing_outer_product_is_masked_and_reported() -> None:
    disparity, _, focal_length_x, baseline, valid_mask = _geometry_inputs()
    sigma_d = torch.full_like(disparity, 1.0e200)
    convention = GeometryConvention()
    rays = pixel_rays(_intrinsics(), height=2, width=2, convention=convention)

    result = center_covariance_from_disparity(
        disparity,
        sigma_d,
        focal_length_x,
        baseline,
        rays.rays,
        valid_mask,
        min_abs_disparity=0.01,
        epsilon=0.0,
        max_eigenvalue=None,
        rank_tolerance=1.0e-14,
        convention=convention,
    )

    assert not result.valid_mask.any()
    assert torch.isnan(result.cov_center).all()
    assert result.diagnostics.nonfinite_covariance_count == 4
