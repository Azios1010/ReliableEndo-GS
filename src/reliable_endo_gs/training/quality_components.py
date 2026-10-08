"""Canonical local/global intervention quality, fixed stereo view footprints."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from reliable_endo_gs.actions.repair import ActionResult, RepairState
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.geometry.stereo_matching import correspondence_error, project_left_support
from reliable_endo_gs.regions.grid import Region
from reliable_endo_gs.rendering.cross_view import RobustLossConfig, cross_view_loss
from reliable_endo_gs.training.interventions import reconstruction_loss
from reliable_endo_gs.training.repairability import outside_loss, stereo_consistency_loss


@dataclass(frozen=True, slots=True)
class QualityComponents:
    global_rgb: torch.Tensor
    global_warp: torch.Tensor
    local_rgb: torch.Tensor
    local_warp: torch.Tensor
    outside_rgb: torch.Tensor
    outside_warp: torch.Tensor

    def values(self) -> dict[str, float]:
        return {name: float(getattr(self, name).detach()) for name in self.__dataclass_fields__}


def local_reconstruction_loss(
    state: RepairState, outcome: ActionResult, region: Region
) -> torch.Tensor:
    """Left core plus its fixed pre-action right projection; available views only."""
    left_core = region.mask(state.batch.left) & state.support
    right_core = project_left_support(left_core, state.base.target.disparity) & state.batch.support
    losses = []
    for render, image, mask in (
        (outcome.left, state.batch.left, left_core),
        (outcome.right, state.batch.right, right_core),
    ):
        result = cross_view_loss(
            render, image.to(render.image.dtype), mask, RobustLossConfig(kind="charbonnier")
        )
        if result.loss is not None:
            losses.append(result.loss)
    if not losses:
        raise ValueError("no observable local RGB support")
    return torch.stack(losses).mean()


def measure_quality(state: RepairState, outcome: ActionResult, region: Region) -> QualityComponents:
    """Same masks/denominators before and after; GT never enters."""
    error, _ = correspondence_error(
        state.batch.left, state.batch.right, outcome.disparity, right_support=state.batch.support
    )
    outside = state.support & ~region.mask(state.batch.left)
    outside_warp = error[outside].mean() if bool(outside.any()) else error.sum() * 0
    return QualityComponents(
        reconstruction_loss(state, outcome),
        stereo_consistency_loss(state, outcome.disparity),
        local_reconstruction_loss(state, outcome, region),
        stereo_consistency_loss(state, outcome.disparity, region),
        outside_loss(state, outcome, region),
        outside_warp,
    )


def component_targets(
    before: dict[str, float], after: dict[str, float], config: InterventionConfig
) -> tuple[float, float, bool]:
    """Return ``(global_decision_gain, local_auxiliary_gain, harm)``.

    The two gains intentionally have separate owners.  ``global_decision_gain``
    is the only value passed to allocation/oracle ranking; ``local_auxiliary_gain``
    supervises the local branch and is never added to the global value.  This
    prevents a small region's reward from being counted once through its local
    mean and again through the full-frame mean.
    """
    if set(before) != set(QualityComponents.__dataclass_fields__) or set(after) != set(before):
        raise ValueError("quality component schema mismatch")
    if not all(math.isfinite(v) for d in (before, after) for v in d.values()):
        raise ValueError("quality components must be finite")
    delta = {
        name: (before[name] - after[name])
        / (config.rgb_scale if name.endswith("rgb") else config.stereo_scale)
        for name in before
    }
    collateral = max(0.0, -delta["outside_rgb"]) + config.stereo_weight * max(
        0.0, -delta["outside_warp"]
    )
    global_gain = (
        delta["global_rgb"]
        + config.stereo_weight * delta["global_warp"]
        - config.collateral_weight * collateral
    )
    local_gain = delta["local_rgb"] + config.stereo_weight * delta["local_warp"]
    harmful = any(value < -config.harm_tolerance for value in delta.values())
    return global_gain, local_gain, harmful
