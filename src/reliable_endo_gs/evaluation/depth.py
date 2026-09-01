"""Development-only depth metrics with explicit validity masks."""

from __future__ import annotations

import torch

from reliable_endo_gs.evaluation.stereo import MetricInputError, MetricResult


def _values(
    predicted: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None
) -> tuple[torch.Tensor, int, int]:
    if not isinstance(predicted, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise MetricInputError("depth inputs must be torch.Tensor values")
    if (
        tuple(predicted.shape) != tuple(target.shape)
        or predicted.ndim != 4
        or predicted.shape[1] != 1
    ):
        raise MetricInputError("depth tensors must have identical shape [B, 1, H, W]")
    if not torch.is_floating_point(predicted) or not torch.is_floating_point(target):
        raise MetricInputError("depth tensors must be floating-point")
    if predicted.device != target.device:
        raise MetricInputError("depth tensors must share a device")
    if valid_mask is None:
        mask = torch.ones_like(predicted, dtype=torch.bool)
    elif (
        not isinstance(valid_mask, torch.Tensor)
        or tuple(valid_mask.shape) != tuple(predicted.shape)
        or valid_mask.dtype != torch.bool
        or valid_mask.device != predicted.device
    ):
        raise MetricInputError("valid_mask must be boolean, on-device, and shape [B, 1, H, W]")
    else:
        mask = valid_mask
    mask = mask & torch.isfinite(predicted) & torch.isfinite(target)
    return (predicted - target).abs()[mask], int(mask.sum().item()), int(predicted.numel())


def depth_mae(
    predicted: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> MetricResult:
    """Return mean absolute depth error in the caller-declared depth units."""

    errors, count, total = _values(predicted, target, valid_mask)
    return MetricResult(
        "depth_mae",
        float(errors.mean().item()) if count else None,
        count,
        total,
        "mean(abs(predicted_depth-target_depth))",
    )


def depth_rmse(
    predicted: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> MetricResult:
    """Return root mean squared depth error in the caller-declared units."""

    errors, count, total = _values(predicted, target, valid_mask)
    value = float(torch.sqrt((errors * errors).mean()).item()) if count else None
    return MetricResult(
        "depth_rmse", value, count, total, "sqrt(mean((predicted_depth-target_depth)^2))"
    )


def evaluate_depth(
    predicted: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> dict[str, MetricResult]:
    return {
        "depth_mae": depth_mae(predicted, target, valid_mask),
        "depth_rmse": depth_rmse(predicted, target, valid_mask),
    }


compute_depth_mae = depth_mae
compute_depth_rmse = depth_rmse
compute_depth_metrics = evaluate_depth
mae = depth_mae
rmse = depth_rmse
