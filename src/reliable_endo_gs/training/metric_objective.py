"""Differentiable TRAIN depth supervision using the canonical metric geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch.nn import functional as F

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch
from reliable_endo_gs.geometry.rectified import reconstruct_rectified
from reliable_endo_gs.training.metric_quality import OfflineMetricTruth


@dataclass(frozen=True, slots=True)
class MetricDepthObjective:
    """Scalar loss/MAE and count on a fixed GT+sensor [B,1,H,W] support."""

    loss: torch.Tensor
    mae_mm: float | None
    pixels: int


def metric_depth_objective(
    prediction: torch.Tensor,
    batch: DeploymentStereoBatch,
    truth: OfflineMetricTruth,
    *,
    sample_id: str,
    scale_mm: float,
    region_mask: torch.Tensor | None = None,
) -> MetricDepthObjective:
    """SmoothL1 on depth error in explicit mm units, not disparity error.

    GT is an independent, detached training target, never a deployment input.
    Sensor and optional pre-action region masks fix the denominator. Invalid
    predictions on that support fail instead of deleting difficult pixels.
    An empty GT support yields a differentiable zero, not a safety label.
    """
    truth.validate(sample_id, batch.support.shape)
    if not isfinite(scale_mm) or scale_mm <= 0:
        raise ValueError("metric depth loss scale must be finite and positive")
    if prediction.shape != batch.support.shape or prediction.device != batch.left.device:
        raise ValueError("metric depth prediction must match deployment [B,1,H,W] grid/device")
    support = batch.support & truth.valid_mask.to(prediction.device)
    if region_mask is not None:
        if region_mask.shape != support.shape or region_mask.dtype != torch.bool:
            raise ValueError("metric depth region must be a boolean mask on the output grid")
        if region_mask.device != prediction.device:
            raise ValueError("metric depth region device mismatch")
        support = support & region_mask.detach()
    with torch.no_grad():
        target = reconstruct_rectified(
            truth.disparity.detach().to(prediction),
            batch.camera,
            support,
            min_effective_disparity=1e-6,
        ).depth
    fixed = target.valid_mask.detach()
    pixels = int(fixed.sum())
    if not pixels:
        zero = torch.where(torch.isfinite(prediction), prediction, 0).sum() * 0
        return MetricDepthObjective(zero, None, 0)
    predicted = reconstruct_rectified(
        prediction, batch.camera, fixed, min_effective_disparity=1e-6
    ).depth
    if not bool(predicted.valid_mask[fixed].all()):
        raise ValueError("invalid metric depth prediction on fixed TRAIN truth support")
    error_mm = (predicted.depth[fixed] - target.depth[fixed]) * 1000
    loss = F.smooth_l1_loss(error_mm / scale_mm, torch.zeros_like(error_mm))
    return MetricDepthObjective(loss, float(error_mm.detach().abs().mean()), pixels)
