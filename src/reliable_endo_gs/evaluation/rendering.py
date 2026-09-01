"""Dependency-free development rendering metrics."""

from __future__ import annotations

import torch

from reliable_endo_gs.contracts import RenderOutput
from reliable_endo_gs.evaluation.stereo import MetricInputError, MetricResult


class UnsupportedMetricError(RuntimeError):
    """Raised for a metric requiring an undeclared optional dependency."""


def _pair(predicted: torch.Tensor, target: torch.Tensor) -> None:
    if not isinstance(predicted, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise MetricInputError("render images must be torch.Tensor values")
    if tuple(predicted.shape) != tuple(target.shape) or predicted.ndim != 4:
        raise MetricInputError("render images must have identical shape [B, C, H, W]")
    if not torch.is_floating_point(predicted) or not torch.is_floating_point(target):
        raise MetricInputError("render images must be floating-point")
    if predicted.device != target.device:
        raise MetricInputError("render images must share a device")


def _mask(image: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(image, dtype=torch.bool)
    if not isinstance(valid_mask, torch.Tensor) or tuple(valid_mask.shape) not in (
        tuple(image.shape),
        (image.shape[0], 1, image.shape[2], image.shape[3]),
    ):
        raise MetricInputError("render valid_mask must be [B, 1, H, W] or image-shaped")
    if valid_mask.dtype != torch.bool or valid_mask.device != image.device:
        raise MetricInputError("render valid_mask must be boolean and on the image device")
    return valid_mask.expand_as(image)


def psnr(
    predicted: torch.Tensor | RenderOutput,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    data_range: float = 1.0,
) -> MetricResult:
    """Return PSNR using the caller-declared data range (default ``[0, 1]``)."""

    if isinstance(predicted, RenderOutput):
        predicted = predicted.image
    _pair(predicted, target)
    if data_range <= 0:
        raise MetricInputError("data_range must be positive")
    mask = _mask(predicted, valid_mask) & torch.isfinite(predicted) & torch.isfinite(target)
    count = int(mask.sum().item())
    total = int(mask.numel())
    if not count:
        value = None
    else:
        mse = ((predicted - target) ** 2)[mask].mean()
        value = (
            float("inf")
            if float(mse.item()) == 0.0
            else float((-10.0 * torch.log10(mse / data_range**2)).item())
        )
    return MetricResult("psnr", value, count, total, "10*log10(data_range^2/MSE)")


def ssim(
    predicted: torch.Tensor | RenderOutput,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> MetricResult:
    """Return a global SSIM estimate, averaged over batch and channels."""

    if isinstance(predicted, RenderOutput):
        predicted = predicted.image
    _pair(predicted, target)
    if data_range <= 0 or k1 < 0 or k2 < 0:
        raise MetricInputError("SSIM data_range and constants must be non-negative (range > 0)")
    mask = _mask(predicted, valid_mask) & torch.isfinite(predicted) & torch.isfinite(target)
    count = int(mask.sum().item())
    total = int(mask.numel())
    if not count:
        value = None
    else:
        x = predicted[mask]
        y = target[mask]
        ux, uy = x.mean(), y.mean()
        vx = ((x - ux) ** 2).mean()
        vy = ((y - uy) ** 2).mean()
        covariance = ((x - ux) * (y - uy)).mean()
        c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
        score = ((2 * ux * uy + c1) * (2 * covariance + c2)) / (
            (ux * ux + uy * uy + c1) * (vx + vy + c2)
        )
        value = float(score.item())
    return MetricResult("ssim", value, count, total, "global luminance/contrast/structure SSIM")


def lpips(*args: object, **kwargs: object) -> MetricResult:
    """LPIPS is intentionally unavailable without a pinned optional backend."""

    del args, kwargs
    raise UnsupportedMetricError("LPIPS requires an explicitly pinned optional backend")


def evaluate_rendering(
    predicted: torch.Tensor | RenderOutput,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    data_range: float = 1.0,
) -> dict[str, MetricResult]:
    return {
        "psnr": psnr(predicted, target, valid_mask, data_range=data_range),
        "ssim": ssim(predicted, target, valid_mask, data_range=data_range),
    }


compute_psnr = psnr
compute_ssim = ssim
compute_rendering_metrics = evaluate_rendering
