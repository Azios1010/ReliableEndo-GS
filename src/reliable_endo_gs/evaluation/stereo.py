"""Development-only stereo metrics on explicit tensor contracts.

The functions in this module do not convert units, rectify images, or infer a
dataset convention.  Callers provide disparity targets and an explicit valid
mask; invalid or non-finite values are excluded from the denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from reliable_endo_gs.contracts import StereoPrediction


class MetricInputError(ValueError):
    """Raised when metric inputs do not satisfy the tensor contract."""


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One scalar metric and the denominator used to compute it."""

    name: str
    value: float | None
    valid_count: int
    total_count: int
    definition: str

    @property
    def available(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "valid_count": self.valid_count,
            "total_count": self.total_count,
            "definition": self.definition,
        }


def _check_pair(predicted: torch.Tensor, target: torch.Tensor, name: str) -> None:
    if not isinstance(predicted, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise MetricInputError(f"{name} inputs must be torch.Tensor values")
    if tuple(predicted.shape) != tuple(target.shape):
        raise MetricInputError(f"{name} inputs must have identical shapes")
    if predicted.ndim != 4 or predicted.shape[1] != 1:
        raise MetricInputError(f"{name} tensors must have shape [B, 1, H, W]")
    if not torch.is_floating_point(predicted) or not torch.is_floating_point(target):
        raise MetricInputError(f"{name} tensors must be floating-point")
    if predicted.device != target.device:
        raise MetricInputError(f"{name} tensors must share a device")


def _valid_values(
    predicted: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None
) -> tuple[torch.Tensor, int, int]:
    _check_pair(predicted, target, "stereo")
    total = int(predicted.numel())
    if valid_mask is None:
        mask = torch.ones_like(predicted, dtype=torch.bool)
    else:
        if not isinstance(valid_mask, torch.Tensor) or tuple(valid_mask.shape) != tuple(
            predicted.shape
        ):
            raise MetricInputError("valid_mask must have shape [B, 1, H, W]")
        if valid_mask.dtype != torch.bool or valid_mask.device != predicted.device:
            raise MetricInputError("valid_mask must be boolean and share the prediction device")
        mask = valid_mask
    mask = mask & torch.isfinite(predicted) & torch.isfinite(target)
    values = (predicted - target).abs()[mask]
    return values, int(mask.sum().item()), total


def endpoint_error(
    predicted: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> MetricResult:
    """Return mean absolute disparity error (EPE) over valid pixels."""

    values, count, total = _valid_values(predicted, target, valid_mask)
    value = float(values.mean().item()) if count else None
    return MetricResult(
        "epe", value, count, total, "mean(abs(predicted_disparity-target_disparity))"
    )


def bad_pixel_rate(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    absolute_threshold: float = 3.0,
    relative_threshold: float = 0.05,
) -> MetricResult:
    """Return the standard D1/bad-pixel rate.

    A pixel is bad when absolute disparity error exceeds the absolute
    threshold *and* relative error exceeds the relative threshold.
    """

    if absolute_threshold < 0 or relative_threshold < 0:
        raise MetricInputError("bad-pixel thresholds must be non-negative")
    _check_pair(predicted, target, "stereo")
    total = int(predicted.numel())
    if valid_mask is None:
        mask = torch.ones_like(predicted, dtype=torch.bool)
    else:
        if (
            not isinstance(valid_mask, torch.Tensor)
            or tuple(valid_mask.shape) != tuple(predicted.shape)
            or valid_mask.dtype != torch.bool
        ):
            raise MetricInputError("valid_mask must be boolean with shape [B, 1, H, W]")
        if valid_mask.device != predicted.device:
            raise MetricInputError("valid_mask must share the prediction device")
        mask = valid_mask
    finite = torch.isfinite(predicted) & torch.isfinite(target)
    mask = mask & finite
    difference = (predicted - target).abs()
    relative = difference / target.abs().clamp_min(torch.finfo(predicted.dtype).eps)
    bad = (difference > absolute_threshold) & (relative > relative_threshold)
    count = int(mask.sum().item())
    value = float(bad[mask].float().mean().item()) if count else None
    return MetricResult(
        "d1",
        value,
        count,
        total,
        f"fraction(abs_error>{absolute_threshold:g} and relative_error>{relative_threshold:g})",
    )


def evaluate_stereo(
    predicted: torch.Tensor | StereoPrediction,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    absolute_threshold: float = 3.0,
    relative_threshold: float = 0.05,
) -> dict[str, MetricResult]:
    """Compute contract-level EPE and D1 without mutating ``predicted``."""

    if isinstance(predicted, StereoPrediction):
        if valid_mask is None:
            valid_mask = predicted.valid_mask
        predicted = predicted.disparity
    return {
        "epe": endpoint_error(predicted, target, valid_mask),
        "d1": bad_pixel_rate(
            predicted,
            target,
            valid_mask,
            absolute_threshold=absolute_threshold,
            relative_threshold=relative_threshold,
        ),
    }


compute_epe = endpoint_error
compute_d1 = bad_pixel_rate
compute_stereo_metrics = evaluate_stereo
epe = endpoint_error
d1 = bad_pixel_rate
