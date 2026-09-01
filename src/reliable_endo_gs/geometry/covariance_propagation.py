"""Canonical first-order disparity-to-camera-center covariance propagation."""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import torch

from reliable_endo_gs.geometry.conventions import (
    GEOMETRY_SCHEMA_VERSION,
    GeometryConvention,
    GeometryProvenance,
)
from reliable_endo_gs.geometry.disparity import (
    ScalarValidityDiagnostics,
    _count,
    _nan_where_invalid,
    _validate_sigma,
    depth_jacobian_wrt_disparity,
)


@dataclass(frozen=True, slots=True)
class CenterJacobianResult:
    """``d mu / dd`` shaped ``[B, 3, H, W]`` with its input rays and mask."""

    jacobian: torch.Tensor
    rays: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: ScalarValidityDiagnostics
    convention: GeometryConvention


@dataclass(frozen=True, slots=True)
class CenterCovarianceDiagnostics:
    """Stability, spectrum, and ray-alignment diagnostics for ``cov_center``."""

    valid_count: int
    nonfinite_sigma_count: int
    negative_sigma_count: int
    zero_sigma_count: int
    stabilization_count: int
    capped_count: int
    nonfinite_covariance_count: int
    psd_violation_count: int
    maximum_symmetry_error: float | None
    minimum_eigenvalue: float | None
    maximum_eigenvalue: float | None
    mean_effective_rank: float | None
    maximum_effective_rank: int | None
    minimum_principal_ray_alignment: float | None


@dataclass(frozen=True, slots=True)
class CenterCovarianceResult:
    """Camera-frame covariance ``[B, H, W, 3, 3]`` and a ``[B, 1, H, W]`` mask."""

    cov_center: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: CenterCovarianceDiagnostics
    convention: GeometryConvention
    provenance: GeometryProvenance | None = None


class CalibratedSigmaInput(Protocol):
    """Provider-neutral structural boundary for finalized calibrated sigma."""

    sigma_d: torch.Tensor
    valid_mask: torch.Tensor
    provider_id: str
    calibration_id: str


def center_jacobian_wrt_disparity(
    disparity: torch.Tensor,
    focal_length_x: torch.Tensor,
    baseline: torch.Tensor,
    rays: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    min_abs_disparity: float,
    convention: GeometryConvention,
) -> CenterJacobianResult:
    """Compute canonical analytic ``J_d = -(f_x B / d^2) r``.

    ``rays`` must be finite, camera-frame ``[B, 3, H, W]`` rays from
    :func:`reliable_endo_gs.geometry.backprojection.pixel_rays`.
    """

    depth_jacobian = depth_jacobian_wrt_disparity(
        disparity,
        focal_length_x,
        baseline,
        valid_mask,
        min_abs_disparity=min_abs_disparity,
        convention=convention,
    )
    _validate_rays(rays, reference=disparity)
    safe_depth_jacobian = torch.where(
        depth_jacobian.valid_mask,
        depth_jacobian.jacobian,
        torch.zeros_like(depth_jacobian.jacobian),
    )
    jacobian = safe_depth_jacobian * rays
    jacobian = _nan_where_invalid(
        jacobian,
        depth_jacobian.valid_mask.expand(-1, 3, -1, -1),
    )
    return CenterJacobianResult(
        jacobian,
        rays,
        depth_jacobian.valid_mask,
        depth_jacobian.diagnostics,
        convention,
    )


def propagate_disparity_covariance(
    sigma_d: torch.Tensor,
    center_jacobian: CenterJacobianResult,
    *,
    epsilon: float,
    max_eigenvalue: float | None,
    rank_tolerance: float,
) -> CenterCovarianceResult:
    """Propagate independent scalar disparity variance as ``J sigma_d^2 J^T``.

    The default policy is ``epsilon=0`` and produces a rank-1 covariance for
    every valid nonzero uncertainty. A nonzero ``epsilon`` adds ``epsilon I``
    in squared metric units. An optional cap scales only the rank-1 component
    and is counted in diagnostics; it never clips disparity itself.
    """

    _validate_sigma(sigma_d, reference=center_jacobian.jacobian[:, :1])
    _validate_stabilization(epsilon, max_eigenvalue, rank_tolerance)
    jacobian = center_jacobian.jacobian
    if sigma_d.device != jacobian.device or sigma_d.dtype != jacobian.dtype:
        raise TypeError("sigma_d must have the same device and dtype as center_jacobian")

    sigma_finite = torch.isfinite(sigma_d)
    sigma_positive = sigma_d > 0
    valid = center_jacobian.valid_mask & sigma_finite & sigma_positive
    safe_sigma = torch.where(valid, sigma_d, torch.zeros_like(sigma_d))
    safe_jacobian = torch.where(valid.expand(-1, 3, -1, -1), jacobian, torch.zeros_like(jacobian))
    scaled_jacobian = safe_jacobian * safe_sigma
    scaled_finite = torch.isfinite(scaled_jacobian).all(dim=1, keepdim=True)
    valid = valid & scaled_finite
    scaled_jacobian = torch.where(
        valid.expand(-1, 3, -1, -1), scaled_jacobian, torch.zeros_like(scaled_jacobian)
    )

    rank_one_eigenvalue = scaled_jacobian.square().sum(dim=1, keepdim=True)
    capped_mask = torch.zeros_like(valid)
    if max_eigenvalue is not None:
        available_rank_one_eigenvalue = max_eigenvalue - epsilon
        capped_mask = valid & (rank_one_eigenvalue > available_rank_one_eigenvalue)
        safe_eigenvalue = torch.where(
            rank_one_eigenvalue > 0,
            rank_one_eigenvalue,
            torch.ones_like(rank_one_eigenvalue),
        )
        scale = torch.minimum(
            torch.ones_like(rank_one_eigenvalue),
            torch.full_like(rank_one_eigenvalue, available_rank_one_eigenvalue) / safe_eigenvalue,
        )
        scaled_jacobian = scaled_jacobian * torch.sqrt(scale)

    vectors = scaled_jacobian.permute(0, 2, 3, 1)
    covariance = vectors.unsqueeze(-1) * vectors.unsqueeze(-2)
    if epsilon > 0:
        covariance = covariance + epsilon * torch.eye(
            3, dtype=covariance.dtype, device=covariance.device
        )
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    finite_covariance = torch.isfinite(covariance).all(dim=-1).all(dim=-1).unsqueeze(1)
    covariance_input_mask = valid
    valid = valid & finite_covariance
    covariance = _nan_where_invalid(covariance, valid.squeeze(1).unsqueeze(-1).unsqueeze(-1))
    diagnostics = _covariance_diagnostics(
        covariance,
        valid,
        center_jacobian.rays,
        input_valid_mask=center_jacobian.valid_mask,
        covariance_input_mask=covariance_input_mask,
        finite_covariance=finite_covariance,
        sigma_finite=sigma_finite,
        sigma_d=sigma_d,
        stabilization_count=_count(valid) if epsilon > 0 else 0,
        capped_count=_count(capped_mask),
        rank_tolerance=rank_tolerance,
    )
    return CenterCovarianceResult(covariance, valid, diagnostics, center_jacobian.convention)


def center_covariance_from_disparity(
    disparity: torch.Tensor,
    sigma_d: torch.Tensor,
    focal_length_x: torch.Tensor,
    baseline: torch.Tensor,
    rays: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    min_abs_disparity: float,
    epsilon: float,
    max_eigenvalue: float | None,
    rank_tolerance: float,
    convention: GeometryConvention,
) -> CenterCovarianceResult:
    """Build the canonical camera-frame center covariance from disparity inputs."""

    center_jacobian = center_jacobian_wrt_disparity(
        disparity,
        focal_length_x,
        baseline,
        rays,
        valid_mask,
        min_abs_disparity=min_abs_disparity,
        convention=convention,
    )
    return propagate_disparity_covariance(
        sigma_d,
        center_jacobian,
        epsilon=epsilon,
        max_eigenvalue=max_eigenvalue,
        rank_tolerance=rank_tolerance,
    )


def center_covariance_from_calibrated_sigma(
    disparity: torch.Tensor,
    calibrated_sigma: CalibratedSigmaInput,
    focal_length_x: torch.Tensor,
    baseline: torch.Tensor,
    rays: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    min_abs_disparity: float,
    epsilon: float,
    max_eigenvalue: float | None,
    rank_tolerance: float,
    convention: GeometryConvention,
) -> CenterCovarianceResult:
    """Consume the finalized calibrated-sigma boundary without provider branching.

    Only ``calibrated_sigma.sigma_d`` is propagated; raw proxy scores and
    Laplace scales are intentionally not accepted. The provider and
    calibration identities remain attached to the geometry result.
    """

    _validate_calibrated_sigma(calibrated_sigma, reference=disparity)
    combined_mask = valid_mask & calibrated_sigma.valid_mask
    result = center_covariance_from_disparity(
        disparity,
        calibrated_sigma.sigma_d,
        focal_length_x,
        baseline,
        rays,
        combined_mask,
        min_abs_disparity=min_abs_disparity,
        epsilon=epsilon,
        max_eigenvalue=max_eigenvalue,
        rank_tolerance=rank_tolerance,
        convention=convention,
    )
    provenance = GeometryProvenance(
        provider_id=calibrated_sigma.provider_id,
        calibration_id=calibrated_sigma.calibration_id,
        convention=convention,
        geometry_schema_version=GEOMETRY_SCHEMA_VERSION,
    )
    return CenterCovarianceResult(
        result.cov_center,
        result.valid_mask,
        result.diagnostics,
        result.convention,
        provenance,
    )


def _validate_rays(rays: torch.Tensor, *, reference: torch.Tensor) -> None:
    if not isinstance(rays, torch.Tensor):
        raise TypeError("rays must be a torch.Tensor")
    expected = (reference.shape[0], 3, reference.shape[2], reference.shape[3])
    if tuple(rays.shape) != expected:
        raise ValueError(f"rays must have shape {expected}; got {tuple(rays.shape)}")
    if not torch.is_floating_point(rays):
        raise TypeError(f"rays must be floating-point; got {rays.dtype}")
    if rays.device != reference.device:
        raise ValueError("rays must be on the same device as disparity")
    if rays.dtype != reference.dtype:
        raise TypeError("rays must have the same dtype as disparity")
    if not bool(torch.all(torch.isfinite(rays))):
        raise ValueError("rays must be finite")


def _validate_stabilization(
    epsilon: float,
    max_eigenvalue: float | None,
    rank_tolerance: float,
) -> None:
    for name, value in (("epsilon", epsilon), ("rank_tolerance", rank_tolerance)):
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a non-negative finite float")
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite float; got {value}")
    if max_eigenvalue is not None:
        if not isinstance(max_eigenvalue, (float, int)) or isinstance(max_eigenvalue, bool):
            raise TypeError("max_eigenvalue must be a non-negative finite float or None")
        if not isfinite(max_eigenvalue) or max_eigenvalue < epsilon:
            raise ValueError("max_eigenvalue must be finite and greater than or equal to epsilon")


def _covariance_diagnostics(
    covariance: torch.Tensor,
    valid_mask: torch.Tensor,
    rays: torch.Tensor,
    *,
    input_valid_mask: torch.Tensor,
    covariance_input_mask: torch.Tensor,
    finite_covariance: torch.Tensor,
    sigma_finite: torch.Tensor,
    sigma_d: torch.Tensor,
    stabilization_count: int,
    capped_count: int,
    rank_tolerance: float,
) -> CenterCovarianceDiagnostics:
    valid_flat = valid_mask.squeeze(1)
    sigma_negative = sigma_d < 0
    sigma_zero = sigma_d == 0
    valid_covariance = covariance[valid_flat]
    valid_rays = rays.permute(0, 2, 3, 1)[valid_flat]
    valid_count = int(valid_covariance.shape[0])
    nonfinite_count = _count(covariance_input_mask & ~finite_covariance)
    if valid_count == 0:
        return CenterCovarianceDiagnostics(
            valid_count=0,
            nonfinite_sigma_count=_count(input_valid_mask & ~sigma_finite),
            negative_sigma_count=_count(input_valid_mask & sigma_finite & sigma_negative),
            zero_sigma_count=_count(input_valid_mask & sigma_finite & sigma_zero),
            stabilization_count=stabilization_count,
            capped_count=capped_count,
            nonfinite_covariance_count=nonfinite_count,
            psd_violation_count=0,
            maximum_symmetry_error=None,
            minimum_eigenvalue=None,
            maximum_eigenvalue=None,
            mean_effective_rank=None,
            maximum_effective_rank=None,
            minimum_principal_ray_alignment=None,
        )

    eigenvalues, eigenvectors = torch.linalg.eigh(valid_covariance)
    symmetry_error = (valid_covariance - valid_covariance.transpose(-1, -2)).abs().amax()
    effective_rank = (eigenvalues > rank_tolerance).sum(dim=-1)
    principal_values = eigenvalues[:, -1]
    principal_vectors = eigenvectors[:, :, -1]
    ray_norm = torch.linalg.vector_norm(valid_rays, dim=-1)
    has_principal_direction = (principal_values > rank_tolerance) & (ray_norm > 0)
    if bool(torch.any(has_principal_direction)):
        unit_rays = valid_rays[has_principal_direction] / ray_norm[has_principal_direction, None]
        alignment = torch.abs((principal_vectors[has_principal_direction] * unit_rays).sum(dim=-1))
        minimum_alignment: float | None = float(alignment.amin().item())
    else:
        minimum_alignment = None
    return CenterCovarianceDiagnostics(
        valid_count=valid_count,
        nonfinite_sigma_count=_count(input_valid_mask & ~sigma_finite),
        negative_sigma_count=_count(input_valid_mask & sigma_finite & sigma_negative),
        zero_sigma_count=_count(input_valid_mask & sigma_finite & sigma_zero),
        stabilization_count=stabilization_count,
        capped_count=capped_count,
        nonfinite_covariance_count=nonfinite_count,
        psd_violation_count=int((eigenvalues < -rank_tolerance).any(dim=-1).sum().item()),
        maximum_symmetry_error=float(symmetry_error.item()),
        minimum_eigenvalue=float(eigenvalues.amin().item()),
        maximum_eigenvalue=float(eigenvalues.amax().item()),
        mean_effective_rank=float(effective_rank.to(torch.float64).mean().item()),
        maximum_effective_rank=int(effective_rank.max().item()),
        minimum_principal_ray_alignment=minimum_alignment,
    )


def _validate_calibrated_sigma(
    calibrated_sigma: CalibratedSigmaInput,
    *,
    reference: torch.Tensor,
) -> None:
    """Validate the structural calibrated-sigma boundary without importing providers."""

    for name in ("sigma_d", "valid_mask", "provider_id", "calibration_id"):
        if not hasattr(calibrated_sigma, name):
            raise TypeError(f"calibrated_sigma must provide {name!r}")
    _validate_sigma(calibrated_sigma.sigma_d, reference=reference)
    if calibrated_sigma.valid_mask.dtype != torch.bool:
        raise TypeError("calibrated_sigma.valid_mask must have dtype torch.bool")
    if tuple(calibrated_sigma.valid_mask.shape) != tuple(reference.shape):
        raise ValueError("calibrated_sigma.valid_mask must match disparity shape")
    if calibrated_sigma.valid_mask.device != reference.device:
        raise ValueError("calibrated_sigma.valid_mask must share disparity device")
    for name in ("provider_id", "calibration_id"):
        value = getattr(calibrated_sigma, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"calibrated_sigma.{name} must be a non-empty string")
