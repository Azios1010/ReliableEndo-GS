"""P1b reliability-signal rescue diagnostics.

P1b is a development-only, inference-time diagnostic study.  The functions in
this module deliberately do not produce a calibrated disparity standard
deviation, geometry covariance, Gaussian parameters, or renderer inputs.

All maps use ``[B, 1, H, W]`` unless documented otherwise.  A score is valid
only where its companion mask is true.  Invalid or unsupported pixels are
represented by ``False`` in the mask and zero in the storage tensor; callers
must use the mask and must not interpret the storage zero as a low-risk value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional

from reliable_endo_gs.evaluation.p1 import evaluate_p1_proxy
from reliable_endo_gs.uncertainty.proxies import (
    left_right_consistency_score,
    photometric_residual_score,
)
from reliable_endo_gs.uncertainty.records import RawUncertaintyScore

P1B_FIT_SEQUENCES: tuple[str, ...] = (
    "dataset_1/keyframe_1",
    "dataset_2/keyframe_1",
    "dataset_3/keyframe_1",
)
P1B_HOLDOUT_SEQUENCES: tuple[str, ...] = (
    "dataset_7/keyframe_2",
    "dataset_4/keyframe_4",
)
P1B_DEVELOPMENT_SEQUENCES: tuple[str, ...] = P1B_FIT_SEQUENCES + P1B_HOLDOUT_SEQUENCES
P1B_FINAL_SEQUENCES: frozenset[str] = frozenset({"dataset_5/keyframe_1", "dataset_8/keyframe_2"})
P1B_COVERAGE_LEVELS: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.5)
P1B_GRID: tuple[int, int] = (4, 4)
P1B_EPS: float = 1.0e-6

RiskDirection = Literal["high", "low", "unknown"]


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """Preregistered candidate metadata.

    ``direction`` describes only the orientation used for risk-coverage
    reporting.  ``unknown`` is intentional for the raw decay ratio: P1b does
    not assume that either increasing or decreasing decay is reliable.
    """

    name: str
    family: str
    direction: RiskDirection
    description: str


@dataclass(frozen=True, slots=True)
class TrajectoryFeatures:
    """Ordered first-three-iteration dynamics and convergence maps."""

    signals: Mapping[str, RawUncertaintyScore]
    v1: torch.Tensor
    v2: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class CorrespondenceSupport:
    """GT-free positive-magnitude LR correspondence support."""

    sample_x: torch.Tensor
    sampled_target_disparity: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class GeometryFeatures:
    """Historical U3 control plus strict and visibility-aware variants."""

    signals: Mapping[str, RawUncertaintyScore]
    support: CorrespondenceSupport
    visibility_mask: torch.Tensor
    occlusion_or_unsupported_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class PhotometricFeatures:
    """Photometric candidates and support diagnostics."""

    signals: Mapping[str, RawUncertaintyScore]
    left_valid_mask: torch.Tensor
    right_valid_mask: torch.Tensor
    bidirectional_valid_mask: torch.Tensor
    specular_exclusion_mask: torch.Tensor
    left_residual: torch.Tensor
    right_residual: torch.Tensor


@dataclass(frozen=True, slots=True)
class FrameSignalRecord:
    """One frame's oracle target and candidate maps for metric aggregation."""

    sequence: str
    sample_id: str
    absolute_error: torch.Tensor
    target_valid_mask: torch.Tensor
    signals: Mapping[str, RawUncertaintyScore]


def _require_map(name: str, value: torch.Tensor, *, dtype: torch.dtype | None = None) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B, 1, H, W]; got {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must use dtype {dtype}; got {value.dtype}")


def _require_mask(name: str, value: torch.Tensor, expected: torch.Size) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype is not torch.bool:
        raise TypeError(f"{name} must be a torch.bool tensor")
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {tuple(expected)}; got {tuple(value.shape)}")


def _finite_mask(*values: torch.Tensor) -> torch.Tensor:
    result = torch.ones_like(values[0], dtype=torch.bool)
    for value in values:
        result &= torch.isfinite(value)
    return result


def _score(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    estimator_id: str,
    *,
    version: str = "1",
) -> RawUncertaintyScore:
    clean = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    clean = torch.where(valid_mask, clean, torch.zeros_like(clean))
    return RawUncertaintyScore(clean, valid_mask, estimator_id, version)


def _validate_iterations(iterations: torch.Tensor) -> None:
    if not isinstance(iterations, torch.Tensor):
        raise TypeError("disparity_iterations must be a torch.Tensor")
    if iterations.ndim != 4 or iterations.shape[1] < 3:
        raise ValueError("disparity_iterations must have shape [B, K>=3, H, W]")
    if not torch.is_floating_point(iterations):
        raise TypeError("disparity_iterations must be floating point")


def validate_p1b_sequences(sequences: Sequence[str]) -> tuple[str, ...]:
    """Validate the exact development-only sequence contract."""

    values = tuple(str(sequence) for sequence in sequences)
    forbidden = P1B_FINAL_SEQUENCES.intersection(values)
    if forbidden:
        raise ValueError(f"P1b refuses final sequence(s): {sorted(forbidden)}")
    if values != P1B_DEVELOPMENT_SEQUENCES:
        raise ValueError(
            f"P1b requires ordered development sequences {P1B_DEVELOPMENT_SEQUENCES}; got {values}"
        )
    return values


def compute_trajectory_features(
    disparity_iterations: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = P1B_EPS,
) -> TrajectoryFeatures:
    """Compute preregistered ordered U1b/U2b features from ``d1,d2,d3``.

    The first three returned iterations are used exactly as ``d1,d2,d3``;
    later shadow iterations are not silently substituted.  ``u1b_decay`` is
    retained as a raw ratio with unknown monotonic direction.  The fixed
    ``u1b_decay_deviation`` diagnostic measures departure from a unit ratio and
    is not a learned transformation.
    """

    _validate_iterations(disparity_iterations)
    _require_mask("valid_mask", valid_mask, disparity_iterations[:, :1].shape)
    if valid_mask.device != disparity_iterations.device:
        raise ValueError("valid_mask and disparity_iterations must share a device")
    if valid_mask.dtype is not torch.bool:
        raise TypeError("valid_mask must be boolean")
    if not torch.isfinite(torch.tensor(float(eps))):
        raise ValueError("eps must be finite")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    d1, d2, d3 = (disparity_iterations[:, index : index + 1] for index in range(3))
    v1 = d2 - d1
    v2 = d3 - d2
    valid = valid_mask & _finite_mask(d1, d2, d3)
    abs_v1 = v1.abs()
    abs_v2 = v2.abs()
    path = abs_v1 + abs_v2
    decay = abs_v2 / (abs_v1 + eps)
    decay_deviation = torch.log((abs_v2 + eps) / (abs_v1 + eps)).abs()
    acceleration = (v2 - v1).abs()
    directional_disagreement = acceleration / (abs_v1 + abs_v2 + eps)
    directional_agreement = torch.clamp(1.0 - directional_disagreement, min=0.0, max=1.0)
    dispersion = disparity_iterations[:, :3].std(dim=1, unbiased=False).unsqueeze(1)
    normalized_dispersion = dispersion / (d3.abs() + eps)
    net = (d3 - d1).abs()
    efficiency = torch.clamp(net / (path + eps), min=0.0, max=1.0)
    oscillation = (v1 * v2 < 0.0).to(disparity_iterations.dtype)

    signals = {
        "u1_historical": _score(abs_v2, valid, "u1_historical"),
        "u1b_path": _score(path, valid, "u1b_path"),
        "u1b_decay": _score(decay, valid, "u1b_decay"),
        "u1b_decay_deviation": _score(decay_deviation, valid, "u1b_decay_deviation"),
        "u1b_acceleration": _score(acceleration, valid, "u1b_acceleration"),
        "u1b_directional_disagreement": _score(
            directional_disagreement, valid, "u1b_directional_disagreement"
        ),
        "u1b_directional_agreement": _score(
            directional_agreement, valid, "u1b_directional_agreement"
        ),
        "u2_historical": _score(dispersion, valid, "u2_historical"),
        "u2b_normalized_dispersion": _score(
            normalized_dispersion, valid, "u2b_normalized_dispersion"
        ),
        "u2b_path_efficiency": _score(efficiency, valid, "u2b_path_efficiency"),
        "u2b_oscillation": _score(oscillation, valid, "u2b_oscillation"),
    }
    return TrajectoryFeatures(signals=signals, v1=v1, v2=v2, valid_mask=valid)


def candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return the complete preregistered candidate table in stable order."""

    return (
        CandidateSpec("u1_historical", "dynamics", "high", "historical |d3-d2| control"),
        CandidateSpec("u1b_path", "dynamics", "high", "|v1|+|v2|"),
        CandidateSpec("u1b_decay", "dynamics", "unknown", "|v2|/(|v1|+eps), raw orientation"),
        CandidateSpec(
            "u1b_decay_deviation",
            "dynamics",
            "high",
            "absolute log-deviation of the update-decay ratio from one",
        ),
        CandidateSpec("u1b_acceleration", "dynamics", "high", "|v2-v1|"),
        CandidateSpec(
            "u1b_directional_disagreement",
            "dynamics",
            "high",
            "normalized disagreement of successive update directions",
        ),
        CandidateSpec(
            "u1b_directional_agreement",
            "dynamics",
            "low",
            "one minus normalized disagreement of successive update directions",
        ),
        CandidateSpec("u2_historical", "dynamics", "high", "population std(d1,d2,d3) control"),
        CandidateSpec(
            "u2b_normalized_dispersion",
            "dynamics",
            "high",
            "population dispersion divided by |d3|+eps",
        ),
        CandidateSpec(
            "u2b_path_efficiency",
            "dynamics",
            "low",
            "|d3-d1|/(|v1|+|v2|+eps), high means direct trajectory",
        ),
        CandidateSpec("u2b_oscillation", "dynamics", "high", "1[v1*v2<0]"),
        CandidateSpec("u3_corrected_raw", "geometry", "high", "historical corrected LR residual"),
        CandidateSpec(
            "u3b_valid_absolute", "geometry", "high", "strict-support absolute LR residual"
        ),
        CandidateSpec("u3b_relative", "geometry", "high", "strict-support relative LR residual"),
        CandidateSpec(
            "u3b_visibility_aware",
            "geometry",
            "high",
            "relative LR residual on collision/order visibility support",
        ),
        CandidateSpec("u4_historical", "observation", "high", "historical one-way RGB L1 control"),
        CandidateSpec(
            "u4b_bidirectional_mean",
            "observation",
            "high",
            "mean left/right photometric residual on intersection support",
        ),
        CandidateSpec(
            "u4b_bidirectional_max",
            "observation",
            "high",
            "max left/right photometric residual on intersection support",
        ),
        CandidateSpec(
            "u4b_bidirectional_disagreement",
            "observation",
            "high",
            "absolute left/right photometric residual difference on intersection support",
        ),
        CandidateSpec(
            "u4b_left_right_asymmetry",
            "observation",
            "high",
            "absolute left/right residual disagreement normalized by their sum",
        ),
        CandidateSpec(
            "u4b_normalized_photo",
            "observation",
            "high",
            "fixed 5x5 local per-channel normalized-intensity residual",
        ),
        CandidateSpec(
            "u4b_gradient",
            "observation",
            "high",
            "fixed central finite-difference gradient residual",
        ),
        CandidateSpec(
            "u4b_ssim_like",
            "observation",
            "high",
            "fixed 5x5 local SSIM-like structural residual",
        ),
        CandidateSpec(
            "u4b_visibility_aware",
            "observation",
            "high",
            "raw photometric residual on the same GT-free visibility support as U3b",
        ),
        CandidateSpec(
            "u4b_specularity_aware",
            "observation",
            "high",
            "raw photometric residual excluding fixed probable-specularity support",
        ),
    )


def risk_oriented_score(score: RawUncertaintyScore, spec: CandidateSpec) -> RawUncertaintyScore:
    """Orient a candidate for ``higher score = higher risk`` evaluation.

    The raw map and its validity mask remain untouched.  For low-risk-is-high
    diagnostics, the fixed map transform is ``1-score`` because both path
    efficiency and the normalized agreement family are bounded in [0, 1].
    Unknown-direction candidates retain their raw orientation and are marked
    as such in metric metadata.
    """

    if spec.direction == "high" or spec.direction == "unknown":
        return score
    values = torch.clamp(1.0 - score.score, min=0.0)
    return _score(values, score.valid_mask, f"{score.estimator_id}:risk_oriented")


def _sampling_grid(sample_x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Build an ``align_corners=True`` grid for a pixel-coordinate map."""

    batch = sample_x.shape[0]
    dtype = sample_x.dtype
    device = sample_x.device
    x = torch.arange(width, dtype=dtype, device=device)
    y = torch.arange(height, dtype=dtype, device=device)
    grid_y, _ = torch.meshgrid(y, x, indexing="ij")
    sample_y = grid_y.expand(batch, -1, -1)
    normalizer_x = max(width - 1, 1)
    normalizer_y = max(height - 1, 1)
    return torch.stack(
        (
            2.0 * sample_x / normalizer_x - 1.0,
            2.0 * sample_y / normalizer_y - 1.0,
        ),
        dim=-1,
    )


def _sample_tensor(
    tensor: torch.Tensor,
    sample_x: torch.Tensor,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Sample a tensor at ``sample_x`` with zero padding and aligned corners."""

    _, _, height, width = tensor.shape
    grid = _sampling_grid(sample_x, height, width)
    clean = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
    return functional.grid_sample(
        clean,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )


def _strict_target_support(
    target_valid_mask: torch.Tensor,
    sample_x: torch.Tensor,
    *,
    threshold: float = 1.0 - 1.0e-6,
) -> torch.Tensor:
    """Require in-view coordinates and all non-zero bilinear mask support."""

    _, _, height, width = target_valid_mask.shape
    target_support = _sample_tensor(
        target_valid_mask.to(dtype=sample_x.dtype), sample_x, mode="bilinear"
    )
    in_view = torch.isfinite(sample_x) & (sample_x >= 0.0) & (sample_x <= float(width - 1))
    return (target_support >= threshold) & in_view.unsqueeze(1)


def _make_correspondence_support(
    source_disparity: torch.Tensor,
    target_disparity: torch.Tensor,
    source_valid_mask: torch.Tensor,
    target_valid_mask: torch.Tensor,
    *,
    direction: Literal[-1, 1],
) -> CorrespondenceSupport:
    """Make strict GT-free support for positive-magnitude disparity warps.

    ``direction=-1`` implements the canonical left-to-right sample
    ``x_target=x_source-d_source``.  ``direction=+1`` implements the swapped
    right-to-left sample ``x_target=x_source+d_source``.
    """

    _require_map("source_disparity", source_disparity)
    _require_map("target_disparity", target_disparity, dtype=source_disparity.dtype)
    _require_mask("source_valid_mask", source_valid_mask, source_disparity.shape)
    _require_mask("target_valid_mask", target_valid_mask, target_disparity.shape)
    if source_disparity.device != target_disparity.device:
        raise ValueError("source and target disparities must share a device")
    if source_valid_mask.device != source_disparity.device:
        raise ValueError("source_valid_mask must share the disparity device")
    if target_valid_mask.device != source_disparity.device:
        raise ValueError("target_valid_mask must share the disparity device")
    if source_disparity.shape != target_disparity.shape:
        raise ValueError("source and target disparity maps must share shape")

    _, _, height, width = source_disparity.shape
    x = torch.arange(width, dtype=source_disparity.dtype, device=source_disparity.device)
    y = torch.arange(height, dtype=source_disparity.dtype, device=source_disparity.device)
    _, grid_x = torch.meshgrid(y, x, indexing="ij")
    sample_x = grid_x.unsqueeze(0) + float(direction) * source_disparity[:, 0]
    target_clean = torch.where(
        torch.isfinite(target_disparity), target_disparity, torch.zeros_like(target_disparity)
    )
    sampled_target = _sample_tensor(target_clean, sample_x, mode="bilinear")
    support = _strict_target_support(target_valid_mask, sample_x)
    valid = source_valid_mask & torch.isfinite(source_disparity) & support
    sampled_target = torch.where(valid, sampled_target, torch.zeros_like(sampled_target))
    return CorrespondenceSupport(sample_x, sampled_target, valid)


def _visibility_from_support(
    sample_x: torch.Tensor,
    strict_mask: torch.Tensor,
) -> torch.Tensor:
    """Construct a conservative GT-free visibility mask.

    A left pixel is retained when its rounded target column has exactly one
    source claimant in the same row and the local target-coordinate mapping is
    monotone.  This fixed one-pixel collision/order rule is descriptive and is
    deliberately not fitted to development holdouts.
    """

    if strict_mask.ndim != 4 or strict_mask.shape[1] != 1:
        raise ValueError("strict_mask must have shape [B, 1, H, W]")
    batch, _, height, width = strict_mask.shape
    safe_x = torch.nan_to_num(sample_x, nan=0.0, posinf=0.0, neginf=0.0)
    rounded_x = torch.floor(safe_x + 0.5).to(torch.long).clamp(0, width - 1)
    batch_ids = torch.arange(batch, device=sample_x.device, dtype=torch.long)[:, None, None]
    row_ids = torch.arange(height, device=sample_x.device, dtype=torch.long)[None, :, None]
    linear = ((batch_ids * height + row_ids) * width + rounded_x).reshape(-1)
    valid_flat = strict_mask[:, 0].reshape(-1)
    counts = torch.zeros(batch * height * width, dtype=torch.long, device=sample_x.device)
    counts.scatter_add_(0, linear[valid_flat], torch.ones_like(linear[valid_flat]))
    unique = counts[linear].reshape(batch, height, width) == 1

    ordered = torch.ones_like(strict_mask[:, 0], dtype=torch.bool)
    x_left = safe_x[..., :-1]
    x_right = safe_x[..., 1:]
    pair = strict_mask[:, 0, :, :-1] & strict_mask[:, 0, :, 1:]
    ordered[..., :-1] &= (~pair) | (x_left <= x_right)
    ordered[..., 1:] &= (~pair) | (x_left <= x_right)
    return strict_mask & unique.unsqueeze(1) & ordered.unsqueeze(1)


def compute_geometry_features(
    disparity_left: torch.Tensor,
    disparity_right_magnitude: torch.Tensor,
    left_valid_mask: torch.Tensor,
    right_valid_mask: torch.Tensor,
    *,
    eps: float = P1B_EPS,
) -> GeometryFeatures:
    """Compute corrected U3 controls and strict/relative/visibility variants.

    The right map is the positive-magnitude output from the swapped forward.
    No signed ``+`` LR residual is introduced.  The historical control calls
    the existing corrected implementation, while all rescued variants use the
    stricter bilinear-support mask defined here.
    """

    _require_map("disparity_left", disparity_left)
    _require_map("disparity_right_magnitude", disparity_right_magnitude, dtype=disparity_left.dtype)
    _require_mask("left_valid_mask", left_valid_mask, disparity_left.shape)
    _require_mask("right_valid_mask", right_valid_mask, disparity_right_magnitude.shape)
    if disparity_left.shape != disparity_right_magnitude.shape:
        raise ValueError("left and right disparity maps must share shape")
    if eps <= 0.0 or not torch.isfinite(torch.tensor(float(eps))):
        raise ValueError("eps must be finite and positive")

    historical = left_right_consistency_score(
        disparity_left,
        disparity_right_magnitude,
        left_valid_mask,
        right_valid_mask,
    )
    support = _make_correspondence_support(
        disparity_left,
        disparity_right_magnitude,
        left_valid_mask,
        right_valid_mask,
        direction=-1,
    )
    residual = (disparity_left - support.sampled_target_disparity).abs()
    relative = residual / (disparity_left.abs() + support.sampled_target_disparity.abs() + eps)
    visibility = _visibility_from_support(support.sample_x, support.valid_mask)
    signals = {
        "u3_corrected_raw": historical,
        "u3b_valid_absolute": _score(residual, support.valid_mask, "u3b_valid_absolute"),
        "u3b_relative": _score(relative, support.valid_mask, "u3b_relative"),
        "u3b_visibility_aware": _score(relative, visibility, "u3b_visibility_aware"),
    }
    return GeometryFeatures(
        signals=signals,
        support=support,
        visibility_mask=visibility,
        occlusion_or_unsupported_mask=support.valid_mask & ~visibility,
    )


def _image_support(
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    sample_x: torch.Tensor,
    disparity_support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add finite source/target-image support to disparity correspondence support."""

    if source_image.ndim != 4 or source_image.shape[1] != 3:
        raise ValueError("images must have shape [B, 3, H, W]")
    if target_image.shape != source_image.shape:
        raise ValueError("source and target images must share shape")
    source_finite = torch.isfinite(source_image).all(dim=1, keepdim=True)
    target_finite = torch.isfinite(target_image).all(dim=1, keepdim=True)
    target_finite_support = _strict_target_support(target_finite, sample_x)
    valid = disparity_support & source_finite & target_finite_support
    warped = _sample_tensor(target_image, sample_x, mode="bilinear")
    return warped, valid


def _local_normalize(image: torch.Tensor, *, window: int = 5, eps: float = 1.0e-4) -> torch.Tensor:
    """Apply fixed per-channel local mean/standard-deviation normalization."""

    if window < 1 or window % 2 == 0:
        raise ValueError("local normalization window must be a positive odd integer")
    padding = window // 2
    mean = functional.avg_pool2d(image, window, stride=1, padding=padding, count_include_pad=False)
    second = functional.avg_pool2d(
        image.square(), window, stride=1, padding=padding, count_include_pad=False
    )
    variance = (second - mean.square()).clamp_min(0.0)
    return (image - mean) / torch.sqrt(variance + eps)


def _central_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed central finite differences with replicated borders."""

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape [B, 3, H, W]")
    kernel_x = image.new_tensor([[-0.5, 0.0, 0.5]]).view(1, 1, 1, 3)
    kernel_y = image.new_tensor([[-0.5], [0.0], [0.5]]).view(1, 1, 3, 1)
    channels = image.shape[1]
    kernel_x = kernel_x.expand(channels, 1, 1, 3)
    kernel_y = kernel_y.expand(channels, 1, 3, 1)
    padded_x = functional.pad(image, (1, 1, 0, 0), mode="replicate")
    padded_y = functional.pad(image, (0, 0, 1, 1), mode="replicate")
    grad_x = functional.conv2d(padded_x, kernel_x, groups=channels)
    grad_y = functional.conv2d(padded_y, kernel_y, groups=channels)
    return grad_x, grad_y


def _ssim_like_residual(
    left: torch.Tensor,
    warped_right: torch.Tensor,
    *,
    window: int = 5,
    data_range: float = 2.0,
) -> torch.Tensor:
    """Return a fixed local SSIM-like residual averaged over channels."""

    if window < 1 or window % 2 == 0:
        raise ValueError("SSIM window must be a positive odd integer")
    padding = window // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mean_left = functional.avg_pool2d(
        left, window, stride=1, padding=padding, count_include_pad=False
    )
    mean_right = functional.avg_pool2d(
        warped_right, window, stride=1, padding=padding, count_include_pad=False
    )
    second_left = functional.avg_pool2d(
        left.square(), window, stride=1, padding=padding, count_include_pad=False
    )
    second_right = functional.avg_pool2d(
        warped_right.square(), window, stride=1, padding=padding, count_include_pad=False
    )
    cross = functional.avg_pool2d(
        left * warped_right, window, stride=1, padding=padding, count_include_pad=False
    )
    variance_left = (second_left - mean_left.square()).clamp_min(0.0)
    variance_right = (second_right - mean_right.square()).clamp_min(0.0)
    covariance = cross - mean_left * mean_right
    numerator = (2.0 * mean_left * mean_right + c1) * (2.0 * covariance + c2)
    denominator = (mean_left.square() + mean_right.square() + c1) * (
        variance_left + variance_right + c2
    )
    ssim = numerator / denominator.clamp_min(torch.finfo(left.dtype).eps)
    return ((1.0 - ssim).clamp(min=0.0, max=2.0)).mean(dim=1, keepdim=True) / 2.0


def _probable_specular(
    image: torch.Tensor,
    *,
    intensity_threshold: float = 0.95,
    saturation_threshold: float = 0.20,
) -> torch.Tensor:
    """Fixed high-intensity/low-saturation heuristic in normalized RGB."""

    image01 = ((image + 1.0) / 2.0).clamp(0.0, 1.0)
    intensity = image01.mean(dim=1, keepdim=True)
    saturation = image01.amax(dim=1, keepdim=True) - image01.amin(dim=1, keepdim=True)
    return (intensity >= intensity_threshold) & (saturation <= saturation_threshold)


def compute_photometric_features(
    left_image: torch.Tensor,
    right_image: torch.Tensor,
    disparity_left: torch.Tensor,
    disparity_right_magnitude: torch.Tensor,
    geometry: GeometryFeatures,
    left_valid_mask: torch.Tensor,
    right_valid_mask: torch.Tensor,
    *,
    eps: float = P1B_EPS,
    local_window: int = 5,
) -> PhotometricFeatures:
    """Compute historical U4 plus bidirectional and robust U4b diagnostics.

    All robust variants use fixed operations and constants.  The bidirectional
    candidates use the intersection of left/right image support, while the
    visibility-aware and specularity-aware candidates report their reduced
    support explicitly instead of assigning a large score to excluded pixels.
    """

    for name, image in (("left_image", left_image), ("right_image", right_image)):
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"{name} must have shape [B, 3, H, W]")
        if not torch.is_floating_point(image):
            raise TypeError(f"{name} must be floating point")
    _require_map("disparity_left", disparity_left)
    _require_map("disparity_right_magnitude", disparity_right_magnitude, dtype=disparity_left.dtype)
    _require_mask("left_valid_mask", left_valid_mask, disparity_left.shape)
    _require_mask("right_valid_mask", right_valid_mask, disparity_right_magnitude.shape)
    if left_image.shape != right_image.shape:
        raise ValueError("left_image and right_image must share shape")
    if (
        left_image.shape[0] != disparity_left.shape[0]
        or left_image.shape[2:] != disparity_left.shape[2:]
    ):
        raise ValueError("image and disparity spatial shapes must match")
    if disparity_left.shape != disparity_right_magnitude.shape:
        raise ValueError("left and right disparity maps must share shape")
    if left_image.dtype != right_image.dtype:
        raise TypeError("left_image and right_image must share dtype")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    _, _, height, width = disparity_left.shape
    del height, width
    left_sample_x = geometry.support.sample_x
    right_support = _make_correspondence_support(
        disparity_right_magnitude,
        disparity_left,
        right_valid_mask,
        left_valid_mask,
        direction=1,
    )
    warped_right, left_image_valid = _image_support(
        left_image,
        right_image,
        left_sample_x,
        geometry.support.valid_mask,
    )
    warped_left, right_image_valid = _image_support(
        right_image,
        left_image,
        right_support.sample_x,
        right_support.valid_mask,
    )
    left_residual = (left_image - warped_right).abs().mean(dim=1, keepdim=True)
    right_residual = (right_image - warped_left).abs().mean(dim=1, keepdim=True)
    bidirectional_valid = left_image_valid & right_image_valid
    bidirectional_mean = (left_residual + right_residual) / 2.0
    bidirectional_max = torch.maximum(left_residual, right_residual)
    bidirectional_disagreement = (left_residual - right_residual).abs()
    asymmetry = (left_residual - right_residual).abs() / (left_residual + right_residual + eps)

    normalized_left = _local_normalize(left_image, window=local_window)
    normalized_right = _local_normalize(right_image, window=local_window)
    warped_normalized_right = _sample_tensor(normalized_right, left_sample_x, mode="bilinear")
    normalized_photo = (normalized_left - warped_normalized_right).abs().mean(dim=1, keepdim=True)

    left_grad_x, left_grad_y = _central_gradients(left_image)
    right_grad_x, right_grad_y = _central_gradients(right_image)
    warped_right_grad_x = _sample_tensor(right_grad_x, left_sample_x, mode="bilinear")
    warped_right_grad_y = _sample_tensor(right_grad_y, left_sample_x, mode="bilinear")
    gradient = (
        (left_grad_x - warped_right_grad_x).abs() + (left_grad_y - warped_right_grad_y).abs()
    ).mean(dim=1, keepdim=True) / 2.0
    ssim_like = _ssim_like_residual(left_image, warped_right, window=local_window)

    visibility_valid = geometry.visibility_mask & left_image_valid
    spec_left = _probable_specular(left_image)
    spec_right = _probable_specular(right_image)
    sampled_spec_right = (
        _sample_tensor(spec_right.to(dtype=left_image.dtype), left_sample_x, mode="nearest") >= 0.5
    )
    specular_exclusion = left_image_valid & ~spec_left & ~sampled_spec_right

    historical = photometric_residual_score(
        left_image,
        right_image,
        disparity_left,
        valid_mask=left_valid_mask,
    )
    signals = {
        "u4_historical": historical,
        "u4b_bidirectional_mean": _score(
            bidirectional_mean, bidirectional_valid, "u4b_bidirectional_mean"
        ),
        "u4b_bidirectional_max": _score(
            bidirectional_max, bidirectional_valid, "u4b_bidirectional_max"
        ),
        "u4b_bidirectional_disagreement": _score(
            bidirectional_disagreement,
            bidirectional_valid,
            "u4b_bidirectional_disagreement",
        ),
        "u4b_left_right_asymmetry": _score(
            asymmetry, bidirectional_valid, "u4b_left_right_asymmetry"
        ),
        "u4b_normalized_photo": _score(normalized_photo, left_image_valid, "u4b_normalized_photo"),
        "u4b_gradient": _score(gradient, left_image_valid, "u4b_gradient"),
        "u4b_ssim_like": _score(ssim_like, left_image_valid, "u4b_ssim_like"),
        "u4b_visibility_aware": _score(left_residual, visibility_valid, "u4b_visibility_aware"),
        "u4b_specularity_aware": _score(left_residual, specular_exclusion, "u4b_specularity_aware"),
    }
    return PhotometricFeatures(
        signals=signals,
        left_valid_mask=left_image_valid,
        right_valid_mask=right_image_valid,
        bidirectional_valid_mask=bidirectional_valid,
        specular_exclusion_mask=specular_exclusion,
        left_residual=left_residual,
        right_residual=right_residual,
    )


def build_p1b_signals(
    disparity_iterations: torch.Tensor,
    left_image: torch.Tensor,
    right_image: torch.Tensor,
    *,
    right_disparity_iterations: torch.Tensor,
    left_valid_mask: torch.Tensor | None = None,
    right_valid_mask: torch.Tensor | None = None,
    eps: float = P1B_EPS,
) -> tuple[Mapping[str, RawUncertaintyScore], GeometryFeatures, PhotometricFeatures]:
    """Build all independent P1b candidates without fusing them."""

    _validate_iterations(disparity_iterations)
    _validate_iterations(right_disparity_iterations)
    if disparity_iterations.shape != right_disparity_iterations.shape:
        raise ValueError("left and right iteration stacks must share shape")
    left_disparity = disparity_iterations[:, -1:]
    right_disparity = right_disparity_iterations[:, -1:]
    left_valid = (
        left_valid_mask
        if left_valid_mask is not None
        else torch.isfinite(left_disparity) & (left_disparity > 0.0)
    )
    right_valid = (
        right_valid_mask
        if right_valid_mask is not None
        else torch.isfinite(right_disparity) & (right_disparity > 0.0)
    )
    _require_mask("left_valid_mask", left_valid, left_disparity.shape)
    _require_mask("right_valid_mask", right_valid, right_disparity.shape)
    trajectory = compute_trajectory_features(disparity_iterations, left_valid, eps=eps)
    geometry = compute_geometry_features(
        left_disparity,
        right_disparity,
        left_valid,
        right_valid,
        eps=eps,
    )
    photometric = compute_photometric_features(
        left_image,
        right_image,
        left_disparity,
        right_disparity,
        geometry,
        left_valid,
        right_valid,
        eps=eps,
    )
    signals: dict[str, RawUncertaintyScore] = {}
    signals.update(trajectory.signals)
    signals.update(geometry.signals)
    signals.update(photometric.signals)
    return signals, geometry, photometric


def shadow_diagnostics(
    disparity_iterations: torch.Tensor,
    *,
    baseline_index: int = 2,
    eps: float = P1B_EPS,
) -> Mapping[str, torch.Tensor]:
    """Return bounded extra-iteration diagnostics without replacing ``d3``.

    The caller is responsible for measuring the extra forward cost and for
    proving that the upstream model can safely return the requested stack.
    ``baseline_index=2`` means the first three predictions remain the Stage-2
    baseline path and all returned diagnostics are shadow-only.
    """

    _validate_iterations(disparity_iterations)
    if baseline_index < 0 or baseline_index >= disparity_iterations.shape[1]:
        raise ValueError("baseline_index is outside the iteration stack")
    if disparity_iterations.shape[1] <= baseline_index + 1:
        raise ValueError("shadow diagnostics require at least one iteration after baseline_index")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    updates = disparity_iterations[:, 1:] - disparity_iterations[:, :-1]
    late_updates = updates[:, baseline_index:]
    d3 = disparity_iterations[:, baseline_index : baseline_index + 1]
    d_last = disparity_iterations[:, -1:]
    sign_changes = (updates[:, 1:] * updates[:, :-1] < 0.0).to(disparity_iterations.dtype)
    return {
        "remaining_trajectory_length": late_updates.abs().sum(dim=1, keepdim=True),
        "late_update_magnitude": late_updates[:, -1:].abs(),
        "late_decay_ratio": late_updates[:, -1:].abs() / (late_updates[:, -2:-1].abs() + eps)
        if late_updates.shape[1] >= 2
        else torch.zeros_like(d_last),
        "oscillation_count": sign_changes[:, baseline_index - 1 :].sum(dim=1, keepdim=True),
        "trajectory_variance": disparity_iterations[:, baseline_index:]
        .var(dim=1, unbiased=False)
        .unsqueeze(1),
        "distance_baseline_to_last": (d_last - d3).abs(),
    }


def _region_values(
    values: torch.Tensor,
    absolute_error: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    grid: tuple[int, int] = P1B_GRID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate each map to fixed-region score/error/coverage vectors."""

    _require_map("values", values)
    _require_map("absolute_error", absolute_error, dtype=values.dtype)
    _require_mask("valid_mask", valid_mask, values.shape)
    rows, columns = grid
    if rows < 1 or columns < 1:
        raise ValueError("region grid dimensions must be positive")
    batch, _, height, width = values.shape
    batch_scores: list[torch.Tensor] = []
    batch_errors: list[torch.Tensor] = []
    batch_coverages: list[torch.Tensor] = []
    for batch_index in range(batch):
        scores: list[torch.Tensor] = []
        errors: list[torch.Tensor] = []
        coverages: list[torch.Tensor] = []
        for row in range(rows):
            y0, y1 = height * row // rows, height * (row + 1) // rows
            for column in range(columns):
                x0, x1 = width * column // columns, width * (column + 1) // columns
                region_values = values[batch_index : batch_index + 1, :, y0:y1, x0:x1]
                region_errors = absolute_error[batch_index : batch_index + 1, :, y0:y1, x0:x1]
                region_mask = valid_mask[batch_index : batch_index + 1, :, y0:y1, x0:x1]
                finite = region_mask & torch.isfinite(region_values) & torch.isfinite(region_errors)
                score_region = region_values[finite]
                error_region = region_errors[finite]
                if score_region.numel():
                    scores.append(score_region.mean())
                    errors.append(error_region.mean())
                    coverages.append(finite.to(values.dtype).mean())
                else:
                    scores.append(values.new_tensor(float("nan")))
                    errors.append(values.new_tensor(float("nan")))
                    coverages.append(values.new_tensor(0.0))
        batch_scores.append(torch.stack(scores))
        batch_errors.append(torch.stack(errors))
        batch_coverages.append(torch.stack(coverages))
    return torch.stack(batch_scores), torch.stack(batch_errors), torch.stack(batch_coverages)


def evaluate_frame_signal(
    signal: RawUncertaintyScore,
    absolute_error: torch.Tensor,
    target_valid_mask: torch.Tensor,
    spec: CandidateSpec,
    *,
    high_error_threshold: float = 1.0,
    coverage_levels: Sequence[float] = P1B_COVERAGE_LEVELS,
    grid: tuple[int, int] = P1B_GRID,
) -> Mapping[str, object]:
    """Evaluate one candidate at pixel, frame, and fixed-region levels."""

    _require_map("absolute_error", absolute_error)
    _require_mask("target_valid_mask", target_valid_mask, absolute_error.shape)
    if signal.score.shape != absolute_error.shape:
        raise ValueError("signal and absolute_error must share shape")
    combined_valid = signal.valid_mask & target_valid_mask & torch.isfinite(absolute_error)
    risk = risk_oriented_score(signal, spec)
    pixel = evaluate_p1_proxy(
        spec.name,
        absolute_error,
        risk.score,
        combined_valid,
        high_error_threshold=high_error_threshold,
        coverage_levels=coverage_levels,
    )
    region_score, region_error, region_coverage = _region_values(
        risk.score, absolute_error, combined_valid, grid=grid
    )
    if region_score.shape[0] != 1:
        raise ValueError("evaluate_frame_signal currently requires a single frame batch")
    region_valid = torch.isfinite(region_score) & torch.isfinite(region_error)
    region_error_map = region_error.reshape(1, 1, 1, -1)
    region_score_map = region_score.reshape(1, 1, 1, -1)
    region_mask_map = region_valid.reshape(1, 1, 1, -1)
    region = evaluate_p1_proxy(
        f"{spec.name}:region",
        region_error_map,
        region_score_map,
        region_mask_map,
        high_error_threshold=high_error_threshold,
        coverage_levels=coverage_levels,
    )
    return {
        "orientation": spec.direction,
        "pixel": pixel.to_dict(),
        "region": region.to_dict(),
        "region_values": {
            "score": region_score.detach().cpu().reshape(-1).tolist(),
            "error": region_error.detach().cpu().reshape(-1).tolist(),
            "coverage": region_coverage.detach().cpu().reshape(-1).tolist(),
        },
        "valid_fraction": float(combined_valid.float().mean().item()),
    }


def _mean_optional(values: Sequence[object]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _bootstrap_mean(
    values: Sequence[float], *, repeats: int, seed: int
) -> Mapping[str, float | None]:
    if not values:
        return {"estimate": None, "lower": None, "upper": None, "count": 0.0}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    samples = torch.tensor(values, dtype=torch.float64)
    indices = torch.randint(
        low=0,
        high=samples.numel(),
        size=(repeats, samples.numel()),
        generator=generator,
    )
    means = samples[indices].mean(dim=1).sort().values
    lower = int(max(0, min(repeats - 1, round(0.025 * (repeats - 1)))))
    upper = int(max(0, min(repeats - 1, round(0.975 * (repeats - 1)))))
    return {
        "estimate": float(samples.mean().item()),
        "lower": float(means[lower].item()),
        "upper": float(means[upper].item()),
        "count": float(samples.numel()),
    }


def bootstrap_mean(
    values: Sequence[float], *, repeats: int = 1000, seed: int = 1314
) -> Mapping[str, float | None]:
    """Return deterministic percentile-bootstrap support for a cluster mean."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    return _bootstrap_mean(values, repeats=repeats, seed=seed)


def _coverage_lookup(curve: Sequence[Mapping[str, object]], coverage: float) -> float | None:
    if not curve:
        return None
    item = min(curve, key=lambda value: abs(float(value["coverage"]) - coverage))
    risk = item.get("risk")
    return None if risk is None else float(risk)


def summarize_candidate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_repeats: int = 1000,
    bootstrap_seed: int = 1314,
) -> Mapping[str, object]:
    """Aggregate per-frame candidate metrics without pooling raw pixels.

    Pixel metrics are frame-macro metrics, matching historical P1's
    per-frame evaluation.  Region metrics are pooled over the fixed 4x4
    summaries and are the primary selection-level summaries.
    """

    if not rows:
        return {
            "frames": 0,
            "pixel": {},
            "frame": {},
            "region": {},
            "coverage_mean": None,
            "bootstrap": {"frame_cluster": {}, "region_frame_rho": {}},
        }
    pixel_rows = [row["pixel"] for row in rows]
    frame_rhos = [
        float(row["region"]["spearman"])
        for row in rows
        if row["region"].get("spearman") is not None
    ]
    pixel: dict[str, object] = {
        "valid_count_mean": _mean_optional([row["valid_count"] for row in pixel_rows]),
        "total_count_mean": _mean_optional([row["total_count"] for row in pixel_rows]),
        "spearman_mean": _mean_optional([row.get("spearman") for row in pixel_rows]),
        "auroc_mean": _mean_optional([row.get("auroc") for row in pixel_rows]),
        "auprc_mean": _mean_optional([row.get("auprc") for row in pixel_rows]),
        "risk_coverage": [],
    }
    pixel_curves = [row.get("risk_coverage", ()) for row in pixel_rows]
    for index, requested in enumerate(P1B_COVERAGE_LEVELS):
        risks = [
            float(curve[index]["risk"])
            for curve in pixel_curves
            if len(curve) > index and curve[index].get("risk") is not None
        ]
        pixel["risk_coverage"].append(
            {
                "coverage": requested,
                "risk_mean": _mean_optional(risks),
                "frame_count": len(risks),
            }
        )

    region_scores: list[float] = []
    region_errors: list[float] = []
    region_coverages: list[float] = []
    for row in rows:
        region_values = row["region_values"]
        for score, error, coverage in zip(
            region_values["score"],
            region_values["error"],
            region_values["coverage"],
            strict=True,
        ):
            if (
                score is not None
                and error is not None
                and coverage
                and torch.isfinite(torch.tensor([float(score), float(error)])).all()
            ):
                region_scores.append(float(score))
                region_errors.append(float(error))
                region_coverages.append(float(coverage))
    if region_scores:
        region_score_tensor = torch.tensor(region_scores, dtype=torch.float32).reshape(1, 1, 1, -1)
        region_error_tensor = torch.tensor(region_errors, dtype=torch.float32).reshape(1, 1, 1, -1)
        region_mask = torch.ones_like(region_score_tensor, dtype=torch.bool)
        pooled_region = evaluate_p1_proxy(
            "region",
            region_error_tensor,
            region_score_tensor,
            region_mask,
            high_error_threshold=1.0,
            coverage_levels=P1B_COVERAGE_LEVELS,
        ).to_dict()
    else:
        pooled_region = {
            "spearman": None,
            "auroc": None,
            "auprc": None,
            "risk_coverage": [],
            "valid_count": 0,
            "total_count": 0,
        }
    frame = {
        "rho_mean": _mean_optional(frame_rhos),
        "rho_median": _median(frame_rhos),
        "rho_count": len(frame_rhos),
        "fraction_rho_gt_0": (
            sum(value > 0.0 for value in frame_rhos) / len(frame_rhos) if frame_rhos else None
        ),
        "fraction_rho_gt_0_2": (
            sum(value > 0.2 for value in frame_rhos) / len(frame_rhos) if frame_rhos else None
        ),
        "fraction_rho_gt_0_3": (
            sum(value > 0.3 for value in frame_rhos) / len(frame_rhos) if frame_rhos else None
        ),
        "fraction_rho_gt_0_5": (
            sum(value > 0.5 for value in frame_rhos) / len(frame_rhos) if frame_rhos else None
        ),
        "rho_distribution": frame_rhos,
    }
    return {
        "frames": len(rows),
        "pixel": pixel,
        "frame": frame,
        "region": pooled_region,
        "coverage_mean": _mean_optional(region_coverages),
        "bootstrap": {
            "frame_cluster": _bootstrap_mean(
                frame_rhos, repeats=bootstrap_repeats, seed=bootstrap_seed
            ),
            "region_frame_rho": _bootstrap_mean(
                frame_rhos, repeats=bootstrap_repeats, seed=bootstrap_seed + 1
            ),
        },
    }


def classify_candidate(
    per_sequence: Mapping[str, Mapping[str, object]],
    *,
    holdouts: Sequence[str] = P1B_HOLDOUT_SEQUENCES,
) -> str:
    """Apply the preregistered qualitative PASS/MARGINAL/FAIL rule.

    A PASS requires positive region-level association and an improvement at
    50% retained coverage over 100% on both holdouts.  MARGINAL records any
    partial holdout evidence without calling it transferable rescue.
    """

    evidence: list[tuple[bool, bool]] = []
    for sequence in holdouts:
        summary = per_sequence.get(sequence)
        if summary is None:
            return "FAIL"
        region = summary.get("region", {})
        rho = region.get("spearman")
        curve = region.get("risk_coverage", ())
        risk_50 = _coverage_lookup(curve, 0.5)
        risk_100 = _coverage_lookup(curve, 1.0)
        positive = rho is not None and float(rho) > 0.0
        useful = risk_50 is not None and risk_100 is not None and risk_50 < risk_100
        evidence.append((positive, useful))
    if all(positive and useful for positive, useful in evidence):
        return "PASS"
    if any(positive or useful for positive, useful in evidence):
        return "MARGINAL"
    return "FAIL"
