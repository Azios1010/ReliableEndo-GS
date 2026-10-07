"""Training-only cached target screening; no GT and no cache mutation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch
from reliable_endo_gs.geometry.rectified import reconstruct_rectified
from reliable_endo_gs.geometry.stereo_matching import correspondence_error
from reliable_endo_gs.teachers.targets import AlignedTeacherTarget


@dataclass(frozen=True, slots=True)
class TargetQuality:
    mask: torch.Tensor
    declared_count: int
    accepted_count: int
    outside_right_count: int
    below_near_count: int


@torch.no_grad()
def screen_target(
    batch: DeploymentStereoBatch, target: AlignedTeacherTarget, config: InterventionConfig
) -> TargetQuality:
    """Finite positive effective depth, observable right support and render range.

    A below-near target is excluded from THIS reconstruction KD objective,
    not declared physically impossible in SCARED. Optional photometric cutoff
    requires an explicitly configured pilot value; none is inferred from GT.
    """
    disparity = target.disparity.detach().to(batch.left.device)
    if disparity.shape != batch.support.shape:
        raise ValueError("KD target must match deployment grid")
    declared = (
        batch.support
        & target.support.to(disparity.device)
        & target.validity.to(disparity.device)
        & torch.isfinite(disparity)
    )
    geometry = reconstruct_rectified(
        disparity, batch.camera, declared, min_effective_disparity=1e-6
    )
    error, corresponding = correspondence_error(
        batch.left, batch.right, disparity, right_support=batch.support
    )
    near = geometry.depth.valid_mask & (geometry.depth.depth >= config.kd_min_depth_m)
    accepted = declared & corresponding & near
    if config.kd_max_photometric_error is not None:
        accepted = accepted & (error <= config.kd_max_photometric_error)
    return TargetQuality(
        accepted,
        int(declared.sum()),
        int(accepted.sum()),
        int((declared & ~corresponding).sum()),
        int((declared & geometry.depth.valid_mask & ~near).sum()),
    )
