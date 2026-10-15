"""Canonical intervention gain/harm protocol and joint training objectives."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from reliable_endo_gs.actions.repair import (
    Action,
    ActionResult,
    RepairActions,
    RepairState,
    action_eligible,
    eligible,
)
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.geometry.stereo_matching import correspondence_error, project_left_support
from reliable_endo_gs.reconstruction.model import ReferenceReconstructionModel
from reliable_endo_gs.regions.grid import Region, grid_regions
from reliable_endo_gs.routing.repairability import RepairabilitySelector, repairability_features
from reliable_endo_gs.teachers import AlignedTeacherTarget
from reliable_endo_gs.training.distillation import stereo_distillation
from reliable_endo_gs.training.interventions import (
    Intervention,
    InterventionHarness,
    intervention_costs,
    reconstruction_loss,
)

if TYPE_CHECKING:
    from reliable_endo_gs.training.student_learning import StudentLearner


def stereo_consistency_loss(
    state: RepairState, disparity: torch.Tensor, region: Region | None = None
) -> torch.Tensor:
    """Fixed sensor/prediction support; invalid after-action warps cost one.

    Deployment RGB-warp proxy, not GT depth error. No prediction-dependent
    post-action denominator may hide damaged support.
    """
    error, _ = correspondence_error(
        state.batch.left, state.batch.right, disparity, right_support=state.batch.support
    )
    mask = state.support if region is None else state.support & region.mask(state.batch.left)
    if not bool(mask.any()):
        raise ValueError("empty stereo consistency support")
    return error[mask].mean()


def outside_loss(state: RepairState, outcome: ActionResult, region: Region) -> torch.Tensor:
    """Mean RGB MAE outside the selected image core, on fixed view supports."""
    core = region.mask(state.batch.left)
    right_core = project_left_support(core, state.base.target.disparity)
    values = []
    for render, image, support, view_core in (
        (outcome.left, state.batch.left, state.support, core),
        (outcome.right, state.batch.right, state.batch.support, right_core),
    ):
        mask = support & ~view_core
        error = (render.image.float() - image.float()).abs().mean(1, keepdim=True)
        values.append(error[mask].mean() if bool(mask.any()) else error.sum() * 0)
    return torch.stack(values).mean()


def quality_gain(
    rgb_before: float,
    rgb_after: float,
    stereo_before: float,
    stereo_after: float,
    outside_before: float,
    outside_after: float,
    config: InterventionConfig,
) -> tuple[float, bool]:
    """Normalized improvement and harm; cost is owned by the budget constraint."""
    if not all(
        math.isfinite(v)
        for v in (rgb_before, rgb_after, stereo_before, stereo_after, outside_before, outside_after)
    ):
        raise ValueError("quality components must be finite")
    rgb_delta = (rgb_before - rgb_after) / config.rgb_scale
    stereo_delta = (stereo_before - stereo_after) / config.stereo_scale
    collateral = max(0.0, outside_after - outside_before) / config.rgb_scale
    gain = rgb_delta + config.stereo_weight * stereo_delta - config.collateral_weight * collateral
    harmful = (
        stereo_delta < -config.harm_tolerance
        or collateral > config.harm_tolerance
        or rgb_delta < -config.harm_tolerance
    )
    return gain, harmful


@dataclass(frozen=True, slots=True)
class SelectionIntervention(Intervention):
    stereo_before: float
    stereo_after: float
    outside_before: float
    outside_after: float
    harmful: bool


class RepairabilityHarness(InterventionHarness):
    """Same registered actions; detached full counterfactual gain/harm labels."""

    def __init__(self, config: InterventionConfig) -> None:
        super().__init__(0.0)
        self.config = config

    def collect(
        self,
        state: RepairState,
        region: Region,
        actions: RepairActions,
        reference: ReferenceReconstructionModel,
    ) -> tuple[SelectionIntervention, ...]:
        if any(p.requires_grad for p in actions.parameters()):
            raise ValueError("temporarily disable action gradients during label collection")
        with torch.no_grad():
            base = actions.execute(state, region, Action.STOP, reference)
            rgb = float(reconstruction_loss(state, base))
            local = float(reconstruction_loss(state, base, region))
            stereo = float(stereo_consistency_loss(state, base.disparity, region))
            outside = float(outside_loss(state, base, region))
            rows = [
                SelectionIntervention(
                    Action.STOP,
                    rgb,
                    rgb,
                    local,
                    local,
                    0.0,
                    0.0,
                    stereo,
                    stereo,
                    outside,
                    outside,
                    False,
                )
            ]
            for action in (Action.STEREO, Action.GAUSSIAN):
                if not action_eligible(state, region, action):
                    rows.append(
                        SelectionIntervention(
                            action,
                            rgb,
                            rgb,
                            local,
                            local,
                            None,
                            0.0,
                            stereo,
                            stereo,
                            outside,
                            outside,
                            False,
                            action_available=False,
                            unavailable_reason="action_precondition",
                        )
                    )
                    continue
                device = state.batch.left.device
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                outcome = actions.execute(state, region, action, reference)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = (time.perf_counter() - started) * 1000
                after = float(reconstruction_loss(state, outcome))
                local_after = float(reconstruction_loss(state, outcome, region))
                stereo_after = float(stereo_consistency_loss(state, outcome.disparity, region))
                outside_after = float(outside_loss(state, outcome, region))
                gain, harm = quality_gain(
                    rgb, after, stereo, stereo_after, outside, outside_after, self.config
                )
                rows.append(
                    SelectionIntervention(
                        action,
                        rgb,
                        after,
                        local,
                        local_after,
                        elapsed,
                        gain,
                        stereo,
                        stereo_after,
                        outside,
                        outside_after,
                        harm,
                    )
                )
        return tuple(rows)


def repairability_step(
    learner: StudentLearner,
    sample: ScaredA00Sample,
    *,
    joint: bool,
    teacher_target: AlignedTeacherTarget | None,
    position: int,
) -> dict[str, float]:
    """All three parameter owners learn; no GT, teacher gradients or stale labels."""
    cfg = learner.intervention_config
    if (
        cfg is None
        or not isinstance(learner.selector, RepairabilitySelector)
        or not isinstance(learner.harness, RepairabilityHarness)
    ):
        raise ValueError("repairability construction mismatch")
    if sample.data_role != "final_train" or sample.sample_id.split("/")[0] not in {
        f"dataset_{i}" for i in range(1, 8)
    }:
        raise ValueError("joint learning cannot consume final_test")
    if learner.config.kd_weight and teacher_target is None:
        raise ValueError("KD requires an aligned target")
    learner.model.train().requires_grad_(True)
    learner.actions.train().requires_grad_(True)
    learner.selector.train()
    for optimizer in (
        learner.model_optimizer,
        learner.action_optimizer,
        learner.selector_optimizer,
    ):
        optimizer.zero_grad(set_to_none=True)
    state = learner.state(sample)
    height, width = state.batch.left.shape[-2:]
    full = grid_regions((height, width), (height, width), learner.routing.halo)[0]
    base = learner.actions.execute(state, full, Action.STOP, learner.model)
    rgb = reconstruction_loss(state, base)
    stereo = stereo_consistency_loss(state, base.disparity)
    kd = rgb.new_zeros(())
    diagnostics: dict[str, float] = {}
    if learner.config.kd_weight:
        assert teacher_target is not None
        learner.model._validate_target_metadata(teacher_target, calibration_metadata=None)
        distillation = stereo_distillation(base.disparity, teacher_target, state.batch, cfg)
        kd = distillation.loss
        diagnostics.update(distillation.diagnostics)
    repair = rgb.new_zeros(())
    selector_loss = rgb.new_zeros(())
    if joint:
        regions = grid_regions((height, width), learner.routing.tile_hw, learner.routing.halo)
        candidates = [r for r in regions if eligible(state, r)]
        if not candidates:
            raise ValueError("no eligible intervention regions")
        region = candidates[position % len(candidates)]
        features = repairability_features(
            state, (region,), cfg, scale_unit_m=learner.model.config.context_scale_m
        )
        learner.actions.requires_grad_(False)
        try:
            rows = learner.harness.collect(state, region, learner.actions, learner.model)
        finally:
            learner.actions.requires_grad_(True)
        gains = features.new_tensor([[r.utility for r in rows]])
        harms = features.new_tensor([[float(r.harmful) for r in rows]])
        costs, action_available = intervention_costs(rows, features)
        selector_ready = bool(action_available[1:].any())
        selector_loss = learner.selector.loss(
            features, gains, harms, cfg, action_available=action_available.unsqueeze(0)
        )
        repair_losses = []
        for action in (Action.STEREO, Action.GAUSSIAN):
            if not action_eligible(state, full, action):
                continue
            outcome = learner.actions.execute(state, full, action, learner.model)
            repair_losses.append(
                reconstruction_loss(state, outcome) / cfg.rgb_scale
                + cfg.stereo_weight
                * stereo_consistency_loss(state, outcome.disparity)
                / cfg.stereo_scale
            )
        repair = torch.stack(repair_losses).mean() if repair_losses else rgb * 0
        diagnostics.update({f"gain_{r.action.name.lower()}": r.utility for r in rows})
        diagnostics.update({f"harm_{r.action.name.lower()}": float(r.harmful) for r in rows})
    loss = (
        learner.config.reconstruction_weight
        * (rgb / cfg.rgb_scale + cfg.stereo_weight * stereo / cfg.stereo_scale)
        + learner.config.kd_weight * kd
        + (learner.config.repair_weight * repair if joint else 0)
    )
    if not bool(torch.isfinite(loss) & torch.isfinite(selector_loss)):
        raise ValueError("nonfinite repairability loss")
    loss.backward()
    if joint and selector_ready:
        selector_loss.backward()
    diagnostics.update(
        learner.commit_gradients(
            joint,
            update_selector=joint and selector_ready,
            active_repairs=tuple(
                action
                for action in (Action.STEREO, Action.GAUSSIAN)
                if joint and action_eligible(state, full, action)
            ),
        )
    )
    if joint:
        learner.record_action_costs(costs, action_available)
    return {
        "loss": float(loss.detach()),
        "reconstruction": float(rgb.detach()),
        "stereo_consistency": float(stereo.detach()),
        "kd": float(kd.detach()),
        "repair": float(repair.detach()),
        "selector": float(selector_loss.detach()),
        **diagnostics,
    }
