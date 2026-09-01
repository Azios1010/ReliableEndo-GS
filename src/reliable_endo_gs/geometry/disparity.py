"""Canonical disparity/depth transformations and scalar uncertainty propagation."""

from dataclasses import dataclass
from math import isfinite

import torch

from reliable_endo_gs.geometry.conventions import GeometryConvention


@dataclass(frozen=True, slots=True)
class ScalarValidityDiagnostics:
    """Counts for a masked scalar transformation over an image batch."""

    total_count: int
    input_masked_count: int
    nonfinite_input_count: int
    wrong_sign_count: int
    below_threshold_count: int
    valid_count: int
    nonfinite_output_count: int


@dataclass(frozen=True, slots=True)
class DepthResult:
    """Metric depth and its validity for tensors shaped ``[B, 1, H, W]``."""

    depth: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: ScalarValidityDiagnostics
    convention: GeometryConvention


@dataclass(frozen=True, slots=True)
class DisparityResult:
    """Disparity and its validity for tensors shaped ``[B, 1, H, W]``."""

    disparity: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: ScalarValidityDiagnostics
    convention: GeometryConvention


@dataclass(frozen=True, slots=True)
class DepthJacobianResult:
    """``dD/dd`` and validity for tensors shaped ``[B, 1, H, W]``."""

    jacobian: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: ScalarValidityDiagnostics
    convention: GeometryConvention


@dataclass(frozen=True, slots=True)
class DepthUncertaintyDiagnostics:
    """Diagnostics for propagation from disparity standard deviation to depth."""

    scalar: ScalarValidityDiagnostics
    nonfinite_sigma_count: int
    negative_sigma_count: int
    zero_sigma_count: int
    valid_count: int
    nonfinite_output_count: int


@dataclass(frozen=True, slots=True)
class DepthUncertaintyResult:
    """Depth standard deviation ``sigma_D`` shaped ``[B, 1, H, W]``."""

    sigma_depth: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: DepthUncertaintyDiagnostics
    convention: GeometryConvention


def focal_length_x_from_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    """Extract ``f_x`` as ``[B]`` from finite ``[B, 3, 3]`` intrinsics."""

    if not isinstance(intrinsics, torch.Tensor):
        raise TypeError("intrinsics must be a torch.Tensor")
    if intrinsics.ndim != 3 or tuple(intrinsics.shape[-2:]) != (3, 3):
        raise ValueError(f"intrinsics must have shape [B, 3, 3]; got {tuple(intrinsics.shape)}")
    if not torch.is_floating_point(intrinsics):
        raise TypeError(f"intrinsics must be floating-point; got {intrinsics.dtype}")
    if not bool(torch.all(torch.isfinite(intrinsics))):
        raise ValueError("intrinsics must be finite")
    focal_length_x = intrinsics[:, 0, 0]
    if not bool(torch.all(focal_length_x > 0)):
        raise ValueError("intrinsics[:, 0, 0] (focal_length_x) must be positive")
    return focal_length_x


def disparity_to_depth(
    disparity: torch.Tensor,
    focal_length_x: torch.Tensor,
    baseline: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    min_abs_disparity: float,
    convention: GeometryConvention,
) -> DepthResult:
    """Compute ``D = f_x B / d`` with explicit invalid-disparity masking.

    All tensor inputs use ``[B, 1, H, W]`` except ``focal_length_x`` and
    ``baseline``, which are batch vectors ``[B]``. Invalid output locations
    contain ``NaN`` and are identified by ``valid_mask``; no disparity is
    clipped before the physical transformation.
    """

    _validate_minimum(min_abs_disparity, name="min_abs_disparity")
    _validate_image_inputs(disparity, valid_mask, value_name="disparity")
    _validate_camera_scalars(focal_length_x, baseline, reference=disparity)
    valid, diagnostics = _positive_scalar_validity(
        disparity,
        valid_mask,
        min_abs_value=min_abs_disparity,
    )
    safe_disparity = torch.where(valid, disparity, torch.ones_like(disparity))
    factor = (focal_length_x * baseline).view(-1, 1, 1, 1)
    depth = factor / safe_disparity
    depth, valid, diagnostics = _finalize_scalar_output(depth, valid, diagnostics)
    return DepthResult(depth, valid, diagnostics, convention)


def depth_to_disparity(
    depth: torch.Tensor,
    focal_length_x: torch.Tensor,
    baseline: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    min_depth: float,
    convention: GeometryConvention,
) -> DisparityResult:
    """Compute inverse disparity ``d = f_x B / D`` with explicit depth masking."""

    _validate_minimum(min_depth, name="min_depth")
    _validate_image_inputs(depth, valid_mask, value_name="depth")
    _validate_camera_scalars(focal_length_x, baseline, reference=depth)
    valid, diagnostics = _positive_scalar_validity(
        depth,
        valid_mask,
        min_abs_value=min_depth,
    )
    safe_depth = torch.where(valid, depth, torch.ones_like(depth))
    factor = (focal_length_x * baseline).view(-1, 1, 1, 1)
    disparity = factor / safe_depth
    disparity, valid, diagnostics = _finalize_scalar_output(disparity, valid, diagnostics)
    return DisparityResult(disparity, valid, diagnostics, convention)


def depth_jacobian_wrt_disparity(
    disparity: torch.Tensor,
    focal_length_x: torch.Tensor,
    baseline: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    min_abs_disparity: float,
    convention: GeometryConvention,
) -> DepthJacobianResult:
    """Compute canonical analytic ``dD/dd = -f_x B / d^2``.

    The output and mask use ``[B, 1, H, W]``. The division is only applied
    after substituting a safe denominator for every invalid input location.
    """

    _validate_minimum(min_abs_disparity, name="min_abs_disparity")
    _validate_image_inputs(disparity, valid_mask, value_name="disparity")
    _validate_camera_scalars(focal_length_x, baseline, reference=disparity)
    valid, diagnostics = _positive_scalar_validity(
        disparity,
        valid_mask,
        min_abs_value=min_abs_disparity,
    )
    safe_disparity = torch.where(valid, disparity, torch.ones_like(disparity))
    factor = (focal_length_x * baseline).view(-1, 1, 1, 1)
    jacobian = -factor / safe_disparity.square()
    jacobian, valid, diagnostics = _finalize_scalar_output(jacobian, valid, diagnostics)
    return DepthJacobianResult(jacobian, valid, diagnostics, convention)


def depth_uncertainty_from_disparity(
    disparity: torch.Tensor,
    sigma_d: torch.Tensor,
    focal_length_x: torch.Tensor,
    baseline: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    min_abs_disparity: float,
    convention: GeometryConvention,
) -> DepthUncertaintyResult:
    """Propagate disparity standard deviation using ``sigma_D = |dD/dd| sigma_d``."""

    jacobian_result = depth_jacobian_wrt_disparity(
        disparity,
        focal_length_x,
        baseline,
        valid_mask,
        min_abs_disparity=min_abs_disparity,
        convention=convention,
    )
    _validate_sigma(sigma_d, reference=disparity)
    sigma_finite = torch.isfinite(sigma_d)
    sigma_positive = sigma_d > 0
    valid = jacobian_result.valid_mask & sigma_finite & sigma_positive
    safe_sigma = torch.where(valid, sigma_d, torch.zeros_like(sigma_d))
    safe_jacobian = torch.where(valid, jacobian_result.jacobian, torch.zeros_like(sigma_d))
    sigma_depth = torch.abs(safe_jacobian) * safe_sigma
    finite_output = torch.isfinite(sigma_depth)
    valid = valid & finite_output
    sigma_depth = _nan_where_invalid(sigma_depth, valid)
    diagnostics = DepthUncertaintyDiagnostics(
        scalar=jacobian_result.diagnostics,
        nonfinite_sigma_count=_count(jacobian_result.valid_mask & ~sigma_finite),
        negative_sigma_count=_count(jacobian_result.valid_mask & sigma_finite & (sigma_d < 0)),
        zero_sigma_count=_count(jacobian_result.valid_mask & sigma_finite & (sigma_d == 0)),
        valid_count=_count(valid),
        nonfinite_output_count=_count(
            jacobian_result.valid_mask & sigma_finite & sigma_positive & ~finite_output
        ),
    )
    return DepthUncertaintyResult(sigma_depth, valid, diagnostics, convention)


def _validate_image_inputs(
    value: torch.Tensor, valid_mask: torch.Tensor, *, value_name: str
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{value_name} must be a torch.Tensor")
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{value_name} must have shape [B, 1, H, W]; got {tuple(value.shape)}")
    if tuple(valid_mask.shape) != tuple(value.shape):
        raise ValueError(
            f"valid_mask must have shape {tuple(value.shape)}; got {tuple(valid_mask.shape)}"
        )
    if not torch.is_floating_point(value):
        raise TypeError(f"{value_name} must be floating-point; got {value.dtype}")
    if valid_mask.dtype != torch.bool:
        raise TypeError(f"valid_mask must have dtype torch.bool; got {valid_mask.dtype}")
    if valid_mask.device != value.device:
        raise ValueError(
            f"valid_mask must be on the same device as {value_name} ({value.device}); "
            f"got {valid_mask.device}"
        )


def _validate_camera_scalars(
    focal_length_x: torch.Tensor,
    baseline: torch.Tensor,
    *,
    reference: torch.Tensor,
) -> None:
    for name, scalar in (("focal_length_x", focal_length_x), ("baseline", baseline)):
        if not isinstance(scalar, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if scalar.ndim != 1 or scalar.shape[0] != reference.shape[0]:
            raise ValueError(
                f"{name} must have shape [{reference.shape[0]}]; got {tuple(scalar.shape)}"
            )
        if not torch.is_floating_point(scalar):
            raise TypeError(f"{name} must be floating-point; got {scalar.dtype}")
        if scalar.device != reference.device:
            raise ValueError(
                f"{name} must be on the same device as the image tensor ({reference.device}); "
                f"got {scalar.device}"
            )
        if scalar.dtype != reference.dtype:
            raise TypeError(
                f"{name} must have the same dtype as the image tensor ({reference.dtype}); "
                f"got {scalar.dtype}"
            )
        if not bool(torch.all(torch.isfinite(scalar))):
            raise ValueError(f"{name} must be finite")
        if not bool(torch.all(scalar > 0)):
            raise ValueError(f"{name} must be positive")


def _validate_sigma(sigma_d: torch.Tensor, *, reference: torch.Tensor) -> None:
    if not isinstance(sigma_d, torch.Tensor):
        raise TypeError("sigma_d must be a torch.Tensor")
    if tuple(sigma_d.shape) != tuple(reference.shape):
        raise ValueError(
            f"sigma_d must have shape {tuple(reference.shape)}; got {tuple(sigma_d.shape)}"
        )
    if not torch.is_floating_point(sigma_d):
        raise TypeError(f"sigma_d must be floating-point; got {sigma_d.dtype}")
    if sigma_d.device != reference.device:
        raise ValueError(
            f"sigma_d must be on the same device as disparity ({reference.device}); "
            f"got {sigma_d.device}"
        )
    if sigma_d.dtype != reference.dtype:
        raise TypeError(
            f"sigma_d must have the same dtype as disparity ({reference.dtype}); "
            f"got {sigma_d.dtype}"
        )


def _positive_scalar_validity(
    value: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    min_abs_value: float,
) -> tuple[torch.Tensor, ScalarValidityDiagnostics]:
    finite = torch.isfinite(value)
    positive = value > 0
    above_threshold = value >= min_abs_value
    valid = valid_mask & finite & positive & above_threshold
    diagnostics = ScalarValidityDiagnostics(
        total_count=value.numel(),
        input_masked_count=_count(~valid_mask),
        nonfinite_input_count=_count(valid_mask & ~finite),
        wrong_sign_count=_count(valid_mask & finite & ~positive),
        below_threshold_count=_count(valid_mask & finite & positive & ~above_threshold),
        valid_count=_count(valid),
        nonfinite_output_count=0,
    )
    return valid, diagnostics


def _finalize_scalar_output(
    output: torch.Tensor,
    valid_mask: torch.Tensor,
    diagnostics: ScalarValidityDiagnostics,
) -> tuple[torch.Tensor, torch.Tensor, ScalarValidityDiagnostics]:
    finite_output = torch.isfinite(output)
    final_valid_mask = valid_mask & finite_output
    finalized_output = _nan_where_invalid(output, final_valid_mask)
    finalized_diagnostics = ScalarValidityDiagnostics(
        total_count=diagnostics.total_count,
        input_masked_count=diagnostics.input_masked_count,
        nonfinite_input_count=diagnostics.nonfinite_input_count,
        wrong_sign_count=diagnostics.wrong_sign_count,
        below_threshold_count=diagnostics.below_threshold_count,
        valid_count=_count(final_valid_mask),
        nonfinite_output_count=_count(valid_mask & ~finite_output),
    )
    return finalized_output, final_valid_mask, finalized_diagnostics


def _nan_where_invalid(value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return torch.where(valid_mask, value, torch.full_like(value, torch.nan))


def _validate_minimum(value: float, *, name: str) -> None:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative float")
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite float; got {value}")


def _count(mask: torch.Tensor) -> int:
    return int(mask.sum().item())
