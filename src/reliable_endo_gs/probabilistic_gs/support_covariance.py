"""Canonical intrinsic Gaussian support covariance construction.

The pinned Endo-E2E-GS parameter head emits normalized ``[w, x, y, z]``
quaternions, positive physical scales, and direct opacity values.  This module
keeps that parameterization at the ReliableEndo-GS boundary and owns the only
implementation of ``R diag(s**2) R.T``.
"""

from dataclasses import dataclass

import torch

from reliable_endo_gs.contracts.common import (
    require_bool,
    require_floating,
    require_same_device,
    require_same_dtype,
    require_shape,
    require_tensor,
)
from reliable_endo_gs.probabilistic_gs.stability import (
    DEFAULT_COVARIANCE_STABILITY_POLICY,
    CovarianceStabilityPolicy,
)

SURFACE_COVARIANCE_SCHEMA_VERSION = "surface_covariance.v1"


@dataclass(frozen=True, slots=True)
class SurfaceCovarianceDiagnostics:
    """Validation counts for one surface-covariance construction."""

    valid_count: int
    invalid_count: int
    nonfinite_count: int
    nonpositive_scale_count: int
    invalid_rotation_count: int


@dataclass(frozen=True, slots=True)
class SurfaceCovarianceResult:
    """Surface covariance plus explicit primitive validity."""

    cov_surface: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: SurfaceCovarianceDiagnostics


def _validate_parameter_shapes(rotations: torch.Tensor, scales: torch.Tensor) -> None:
    require_tensor("rotations", rotations)
    require_tensor("scales", scales)
    if rotations.ndim < 2 or rotations.shape[-1] != 4:
        raise ValueError(f"rotations must have shape [..., 4]; got {tuple(rotations.shape)}")
    if scales.shape != rotations.shape[:-1] + (3,):
        raise ValueError(
            "scales must have shape rotations.shape[:-1] + (3,); "
            f"got {tuple(scales.shape)} for rotations {tuple(rotations.shape)}"
        )
    require_floating("rotations", rotations)
    require_floating("scales", scales)
    require_same_device("scales", scales, "rotations", rotations)
    require_same_dtype("scales", scales, "rotations", rotations)


def quaternion_to_rotation_matrix(
    rotations: torch.Tensor,
    *,
    norm_epsilon: float = 1.0e-12,
) -> torch.Tensor:
    """Convert normalized-or-normalizable ``[..., 4]`` ``[w,x,y,z]`` quaternions.

    The function is strict: non-finite or near-zero quaternions raise instead
    of being silently converted to an identity rotation.  Non-unit but finite
    quaternions are normalized once at this canonical owner.
    """

    require_tensor("rotations", rotations)
    if rotations.ndim < 1 or rotations.shape[-1] != 4:
        raise ValueError(f"rotations must have shape [..., 4]; got {tuple(rotations.shape)}")
    require_floating("rotations", rotations)
    if not bool(torch.isfinite(rotations).all()):
        raise ValueError("rotations must contain only finite values")
    norm = torch.linalg.vector_norm(rotations, dim=-1)
    if not bool(torch.isfinite(norm).all()) or bool((norm <= norm_epsilon).any()):
        raise ValueError("rotations must have a finite norm greater than norm_epsilon")
    w, x, y, z = (rotations / norm.unsqueeze(-1)).unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(rotations.shape[:-1] + (3, 3))


def build_surface_covariance(
    rotations: torch.Tensor,
    scales: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    policy: CovarianceStabilityPolicy = DEFAULT_COVARIANCE_STABILITY_POLICY,
) -> SurfaceCovarianceResult:
    """Build ``Sigma_surf`` for ``rotations/scales`` with explicit validity.

    Parameters use ``[..., 4]`` rotations and ``[..., 3]`` physical scales;
    the output has shape ``[..., 3, 3]``.  Invalid primitives are returned as
    ``NaN`` covariance entries and marked false, never replaced by identity or
    jitter.  ``valid_mask`` has shape ``rotations.shape[:-1]``.
    """

    _validate_parameter_shapes(rotations, scales)
    shape = rotations.shape[:-1]
    if valid_mask is None:
        input_valid = torch.ones(shape, dtype=torch.bool, device=rotations.device)
    else:
        require_tensor("valid_mask", valid_mask)
        require_shape("valid_mask", valid_mask, tuple(shape))
        require_bool("valid_mask", valid_mask)
        require_same_device("valid_mask", valid_mask, "rotations", rotations)
        input_valid = valid_mask

    finite = torch.isfinite(rotations).all(dim=-1) & torch.isfinite(scales).all(dim=-1)
    positive_scales = (scales > 0).all(dim=-1)
    safe_rotations = torch.where(torch.isfinite(rotations), rotations, torch.zeros_like(rotations))
    rotation_norm = torch.linalg.vector_norm(safe_rotations, dim=-1)
    valid_rotation = torch.isfinite(rotation_norm) & (
        rotation_norm > policy.quaternion_norm_epsilon
    )
    valid = input_valid & finite & positive_scales & valid_rotation

    safe_norm = torch.where(valid_rotation, rotation_norm, torch.ones_like(rotation_norm))
    normalized = safe_rotations / safe_norm.unsqueeze(-1)
    w, x, y, z = normalized.unbind(dim=-1)
    rotation = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(shape + (3, 3))
    diagonal = torch.diag_embed(
        torch.where(torch.isfinite(scales), scales, torch.zeros_like(scales)).square()
    )
    covariance = rotation @ diagonal @ rotation.transpose(-1, -2)
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    nan_covariance = torch.full_like(covariance, float("nan"))
    covariance = torch.where(valid.unsqueeze(-1).unsqueeze(-1), covariance, nan_covariance)

    diagnostics = SurfaceCovarianceDiagnostics(
        valid_count=int(valid.sum().item()),
        invalid_count=int((~valid).sum().item()),
        nonfinite_count=int((~finite).sum().item()),
        nonpositive_scale_count=int((~positive_scales).sum().item()),
        invalid_rotation_count=int((~valid_rotation).sum().item()),
    )
    return SurfaceCovarianceResult(covariance, valid, diagnostics)


def surface_covariance(
    rotations: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Return strict ``Sigma_surf = R diag(s**2) R.T``.

    This convenience API raises if any supplied primitive is invalid.  Call
    :func:`build_surface_covariance` when per-primitive masks/diagnostics are
    required.
    """

    result = build_surface_covariance(rotations, scales)
    if not bool(result.valid_mask.all()):
        raise ValueError("rotations/scales contain invalid Gaussian primitives")
    return result.cov_surface


__all__ = [
    "SURFACE_COVARIANCE_SCHEMA_VERSION",
    "SurfaceCovarianceDiagnostics",
    "SurfaceCovarianceResult",
    "build_surface_covariance",
    "quaternion_to_rotation_matrix",
    "surface_covariance",
]
