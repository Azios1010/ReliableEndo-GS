"""Canonical stereo KD loss; teacher/cache values never receive gradients."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch
from reliable_endo_gs.teachers import AlignedTeacherTarget
from reliable_endo_gs.teachers.quality import screen_target


@dataclass(frozen=True)
class DistillationResult:
    """Scalar loss and detached diagnostics on the student [B,1,H,W] grid."""

    loss: torch.Tensor
    diagnostics: dict[str, float]


def stereo_distillation(
    prediction: torch.Tensor,
    target: AlignedTeacherTarget,
    batch: DeploymentStereoBatch,
    config: InterventionConfig | None = None,
) -> DistillationResult:
    """SmoothL1 disparity KD in current-grid pixels on fixed teacher support.

    Legacy mode retains unscaled SmoothL1. Intervention mode screens physical
    geometry/FOV first and divides by the explicit KD pixel scale. Screening
    does not certify teacher correctness. An empty mask retains a zero-valued
    student gradient path and is reported rather than inventing a target.
    """
    if prediction.shape != batch.support.shape or not prediction.is_floating_point():
        raise ValueError("stereo KD prediction must match [B,1,H,W] support")
    teacher = target.disparity.detach().to(device=prediction.device, dtype=prediction.dtype)
    if teacher.shape != prediction.shape:
        raise ValueError("stereo KD teacher grid mismatch")
    declared = (
        batch.support
        & target.support.to(prediction.device)
        & target.validity.to(prediction.device)
        & torch.isfinite(teacher)
    )
    diagnostics: dict[str, float] = {"kd_declared_pixels": float(declared.sum())}
    if config is None:
        mask = declared
        scale = 1.0
    else:
        quality = screen_target(batch, target, config)
        mask = quality.mask
        scale = config.kd_scale_px
        diagnostics.update(
            kd_declared_pixels=float(quality.declared_count),
            kd_below_near_pixels=float(quality.below_near_count),
            kd_outside_right_pixels=float(quality.outside_right_count),
        )
    if not bool(torch.isfinite(prediction[mask]).all()):
        raise ValueError("stereo KD has nonfinite supported student predictions")
    loss = (
        F.smooth_l1_loss(prediction[mask], teacher[mask]) / scale
        if bool(mask.any())
        else (
            torch.where(torch.isfinite(prediction), prediction, torch.zeros_like(prediction)).sum()
            * 0
        )
    )
    diagnostics.update(kd_accepted_pixels=float(mask.sum()), kd_empty=float(not bool(mask.any())))
    return DistillationResult(loss, diagnostics)
