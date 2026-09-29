"""Canonical bounded residual composition for A04 Gaussian fields.

This module owns the only residual-to-attribute transformation used by the
reference pass.  Residuals are local-frame axis-angle, log-scale and alpha
logit values.  Invalid primitive masking is kept separate from opacity and
all mixed-precision-sensitive operations use an explicit FP32 island.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from reliable_endo_gs.contracts.gaussians import GaussianField
from reliable_endo_gs.gaussians.priors import GeometryPrior, rotation_matrix_to_quaternion
from reliable_endo_gs.gaussians.residual_head import AttributeResiduals, ResidualBounds
from reliable_endo_gs.gaussians.support_covariance import build_surface_covariance


@dataclass(frozen=True, slots=True)
class CompositionDiagnostics:
    """Finite diagnostics for one residual composition."""

    valid_count: int
    invalid_count: int
    rotation_saturated_count: int
    log_scale_saturated_count: int
    alpha_saturated_count: int
    fp32_stabilized: bool

    def to_dict(self) -> dict[str, int | bool]:
        """Return a serializable diagnostics mapping."""

        return {
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "rotation_saturated_count": self.rotation_saturated_count,
            "log_scale_saturated_count": self.log_scale_saturated_count,
            "alpha_saturated_count": self.alpha_saturated_count,
            "fp32_stabilized": self.fp32_stabilized,
        }


@dataclass(frozen=True, slots=True)
class ComposedGaussianAttributes:
    """Composed R/S/alpha values before packaging as ``GaussianField``."""

    rotations: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    valid_mask: torch.Tensor
    cov_surface: torch.Tensor
    diagnostics: CompositionDiagnostics


def _validate_residuals(prior: GeometryPrior, residuals: AttributeResiduals) -> None:
    expected = (prior.mu.shape[0], prior.mu.shape[1])
    if residuals.delta_rotation.shape[:2] != expected:
        raise ValueError("residual batch/primitive dimensions must match GeometryPrior")
    if residuals.delta_rotation.device != prior.mu.device:
        raise ValueError("residuals must share the prior device")
    if residuals.delta_rotation.dtype != prior.mu.dtype:
        raise TypeError("residuals must share the prior dtype")


def _skew(vector: torch.Tensor) -> torch.Tensor:
    """Return the cross-product matrix for ``[...,3]`` vectors."""

    x, y, z = vector.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def stable_axis_angle_exp(delta_rotation: torch.Tensor) -> torch.Tensor:
    """Compute ``Exp([delta_rotation]x)`` with stable small-angle series.

    The calculation promotes FP16/BF16 inputs to FP32, while FP64 inputs stay
    FP64.  The returned matrix uses the promoted dtype; callers may cast once
    after the full composition to avoid half-precision cancellation.
    """

    if not isinstance(delta_rotation, torch.Tensor):
        raise TypeError("delta_rotation must be a tensor")
    if delta_rotation.ndim < 1 or delta_rotation.shape[-1] != 3:
        raise ValueError("delta_rotation must have trailing shape [3]")
    if not torch.is_floating_point(delta_rotation):
        raise TypeError("delta_rotation must be floating point")
    work_dtype = (
        torch.float32
        if delta_rotation.dtype in (torch.float16, torch.bfloat16)
        else delta_rotation.dtype
    )
    vector = delta_rotation.to(dtype=work_dtype)
    theta2 = vector.square().sum(dim=-1, keepdim=True)
    # Clamp before the square root.  ``sqrt(theta2).clamp_min(eps)`` still
    # differentiates through sqrt(0), whose derivative is NaN; the clamped
    # argument gives the direct branch a finite, zero derivative at the origin.
    # The Taylor branch supplies the exact value selected for small angles.
    theta2_safe = theta2.clamp_min(1.0e-8)
    theta_safe = torch.sqrt(theta2_safe)
    a_direct = torch.sin(theta_safe) / theta_safe
    b_direct = (1.0 - torch.cos(theta_safe)) / theta2_safe
    theta4 = theta2.square()
    a_series = 1.0 - theta2 / 6.0 + theta4 / 120.0
    b_series = 0.5 - theta2 / 24.0 + theta4 / 720.0
    small = theta2 < 1.0e-6
    coefficient_a = torch.where(small, a_series, a_direct)
    coefficient_b = torch.where(small, b_series, b_direct)
    cross = _skew(vector)
    identity = torch.eye(3, dtype=work_dtype, device=vector.device)
    identity = identity.expand(*vector.shape[:-1], 3, 3)
    return (
        identity
        + coefficient_a.unsqueeze(-1) * cross
        + coefficient_b.unsqueeze(-1) * (cross @ cross)
    )


def _safe_prior_frame(prior: GeometryPrior, valid: torch.Tensor) -> torch.Tensor:
    frame = prior.R0
    if not bool(torch.isfinite(frame[valid]).all()):
        raise ValueError("valid GeometryPrior frames must be finite")
    identity = torch.eye(3, dtype=frame.dtype, device=frame.device)
    identity = identity.expand(frame.shape[0], frame.shape[1], 3, 3)
    return torch.where(valid.unsqueeze(-1).unsqueeze(-1), frame, identity)


def _saturation_count(value: torch.Tensor, bound: float, valid: torch.Tensor) -> int:
    return int(((value.abs() >= bound - 1.0e-6) & valid.unsqueeze(-1)).sum().item())


def compose_gaussian_attributes(
    prior: GeometryPrior,
    residuals: AttributeResiduals,
    *,
    bounds: ResidualBounds | None = None,
) -> ComposedGaussianAttributes:
    """Compose bounded local-frame residuals with an A03 geometry prior.

    ``R = R0 @ Exp([delta_r]x)`` uses intentional right multiplication;
    ``S = clamp(S0 * exp(delta_s))`` is in positive metric units; and
    ``alpha = sigmoid(logit(clamp(alpha0)) + delta_a)`` clips only the valid
    opacity prior.  Invalid primitives are represented by finite placeholders
    and a separate false mask, never by opacity alone.
    """

    if not isinstance(prior, GeometryPrior):
        raise TypeError("prior must be a GeometryPrior")
    if not isinstance(residuals, AttributeResiduals):
        raise TypeError("residuals must be AttributeResiduals")
    _validate_residuals(prior, residuals)
    resolved = bounds or ResidualBounds()
    valid = prior.valid_mask
    if valid.dtype != torch.bool:
        raise TypeError("prior.valid_mask must be bool")

    delta_rotation = residuals.delta_rotation.clamp(
        -resolved.max_rotation_rad, resolved.max_rotation_rad
    )
    delta_log_scale = residuals.delta_log_scale.clamp(
        -resolved.max_log_scale, resolved.max_log_scale
    )
    delta_alpha = residuals.delta_alpha_logit.clamp(
        -resolved.max_alpha_logit, resolved.max_alpha_logit
    )

    work_dtype = (
        torch.float32 if prior.mu.dtype in (torch.float16, torch.bfloat16) else prior.mu.dtype
    )
    frame = _safe_prior_frame(prior, valid).to(dtype=work_dtype)
    local_rotation = stable_axis_angle_exp(delta_rotation.to(dtype=work_dtype))
    composed_frame = frame @ local_rotation
    composed_frame = torch.where(valid.unsqueeze(-1).unsqueeze(-1), composed_frame, frame)

    scales0 = torch.where(
        valid.unsqueeze(-1),
        prior.S0,
        prior.S0.new_full(prior.S0.shape, resolved.min_scale_m),
    ).to(dtype=work_dtype)
    scales = scales0 * torch.exp(delta_log_scale.to(dtype=work_dtype))
    scales = scales.clamp(min=resolved.min_scale_m, max=resolved.max_scale_m)
    scales = torch.where(
        valid.unsqueeze(-1), scales, scales.new_full(scales.shape, resolved.min_scale_m)
    )

    alpha0 = prior.alpha0.to(dtype=work_dtype).clamp(
        min=resolved.alpha_logit_epsilon, max=1.0 - resolved.alpha_logit_epsilon
    )
    alpha = torch.sigmoid(torch.logit(alpha0) + delta_alpha.to(dtype=work_dtype)).clamp(
        min=0.0, max=1.0
    )
    alpha = torch.where(valid.unsqueeze(-1), alpha, torch.zeros_like(alpha))

    rotations = rotation_matrix_to_quaternion(composed_frame)
    rotations = rotations / torch.linalg.vector_norm(rotations, dim=-1, keepdim=True).clamp_min(
        1.0e-12
    )
    covariance_result = build_surface_covariance(rotations, scales, valid_mask=valid)
    covariance = torch.where(
        valid.unsqueeze(-1).unsqueeze(-1),
        covariance_result.cov_surface,
        torch.zeros_like(covariance_result.cov_surface),
    )

    output_dtype = prior.mu.dtype
    return ComposedGaussianAttributes(
        rotations=rotations.to(dtype=output_dtype),
        scales=scales.to(dtype=output_dtype),
        opacities=alpha.to(dtype=output_dtype),
        valid_mask=valid,
        cov_surface=covariance.to(dtype=output_dtype),
        diagnostics=CompositionDiagnostics(
            valid_count=int(valid.sum().item()),
            invalid_count=int((~valid).sum().item()),
            rotation_saturated_count=_saturation_count(
                delta_rotation, resolved.max_rotation_rad, valid
            ),
            log_scale_saturated_count=_saturation_count(
                delta_log_scale, resolved.max_log_scale, valid
            ),
            alpha_saturated_count=_saturation_count(delta_alpha, resolved.max_alpha_logit, valid),
            fp32_stabilized=work_dtype is torch.float32
            and prior.mu.dtype in (torch.float16, torch.bfloat16),
        ),
    )


def compose_gaussian_field(
    prior: GeometryPrior,
    residuals: AttributeResiduals,
    *,
    bounds: ResidualBounds | None = None,
) -> GaussianField:
    """Return a canonical first-party ``GaussianField`` from A03 + residuals."""

    composed = compose_gaussian_attributes(prior, residuals, bounds=bounds)
    colors = torch.where(
        composed.valid_mask.unsqueeze(-1),
        prior.rgb.clamp(0.0, 1.0),
        torch.zeros_like(prior.rgb),
    )
    means = torch.where(composed.valid_mask.unsqueeze(-1), prior.mu, torch.zeros_like(prior.mu))
    return GaussianField(
        means3d=means,
        colors=colors,
        rotations=composed.rotations,
        scales=composed.scales,
        opacities=composed.opacities,
        valid_mask=composed.valid_mask,
        cov_surface=composed.cov_surface,
    )


def compose_residuals(
    prior: GeometryPrior,
    residuals: AttributeResiduals,
    *,
    bounds: ResidualBounds | None = None,
) -> GaussianField:
    """Compatibility alias for :func:`compose_gaussian_field`."""

    return compose_gaussian_field(prior, residuals, bounds=bounds)


compose_reference_attributes = compose_gaussian_attributes


__all__ = [
    "ComposedGaussianAttributes",
    "CompositionDiagnostics",
    "compose_gaussian_attributes",
    "compose_gaussian_field",
    "compose_reference_attributes",
    "compose_residuals",
    "stable_axis_angle_exp",
]
