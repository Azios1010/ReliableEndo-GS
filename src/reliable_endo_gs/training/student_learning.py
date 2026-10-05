"""Student reconstruction/KD then alternating joint reconstruction/utility updates.

The selector has no upstream gradient: action choice remains discrete. Joint
reconstruction explicitly trains both repairs from the SAME student base, not
a chained/selected action. Utility supervision observes current state before
the update, so moving student/repair weights do not reuse stale frozen labels.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from reliable_endo_gs.actions.repair import Action, RepairActions, RepairState, eligible
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.reconstruction.model import ReferenceReconstructionResult
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.regions.grid import grid_regions
from reliable_endo_gs.routing.config import RoutingConfig
from reliable_endo_gs.routing.selector import ActionSelector, region_features, select_action
from reliable_endo_gs.teachers import AlignedTeacherTarget
from reliable_endo_gs.training.a04_reference import _move_batch
from reliable_endo_gs.training.interventions import InterventionHarness, reconstruction_loss
from reliable_endo_gs.training.student_config import StudentLearningConfig


@contextmanager
def evaluation_mode(*modules: nn.Module) -> Iterator[None]:
    """Restore every nested train/eval flag; never change requires_grad/state."""
    flags = [(item, item.training) for module in modules for item in module.modules()]
    try:
        for module in modules:
            module.eval()
        with torch.no_grad():
            yield
    finally:
        for item, flag in flags:
            item.training = flag


class StudentLearner:
    """Disjoint optimizer ownership: student+head, repairs, and selector.

    Teacher/reference targets are supervision arguments, never model inputs.
    No frozen reference is stored or optimized by this class.
    """

    def __init__(
        self,
        model: StudentReconstructionModel,
        config: StudentLearningConfig,
        routing: RoutingConfig,
    ) -> None:
        self.model, self.config, self.routing = model, config, routing
        first = model.stereo_student.features[0]
        if (
            not isinstance(first, nn.Conv2d)
            or first.out_channels != config.width
            or model.stereo_student.bins != config.bins
            or model.stereo_student.minimum_effective_px != config.minimum_effective_px
        ):
            raise ValueError("student architecture does not match learning config")
        if routing.halo < model.config.prior_policy.surface.neighborhood_radius:
            raise ValueError("repair halo must cover the surface neighbourhood")
        device = next(model.parameters()).device
        self.actions = RepairActions(
            model,
            width=routing.width,
            max_disparity_px=routing.max_disparity_px,
            composition_alpha_epsilon=routing.composition_alpha_epsilon,
        ).to(device)
        self.selector = ActionSelector(routing.width).to(device)
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.action_optimizer = torch.optim.Adam(
            self.actions.parameters(), lr=routing.learning_rate
        )
        self.selector_optimizer = torch.optim.Adam(
            self.selector.parameters(), lr=routing.learning_rate
        )
        self.harness = InterventionHarness(routing.cost_weight_per_ms)
        self.cost_max_ms = torch.zeros(3, device=device)
        self.cost_observations = 0

    def state(self, sample: ScaredA00Sample) -> RepairState:
        """GT exclusion before forward; preserve student graph during training."""
        if sample.supervision is not None:
            raise ValueError("student deployment source must not load GT")
        batch = _move_batch(sample.batch, next(self.model.parameters()).device)
        result = self.model(
            batch,
            sample_ids=(sample.sample_id,),
            data_role=sample.data_role,
            calibration_metadata=sample.calibration_metadata,
        )
        return RepairState(batch, result)

    def step(
        self,
        sample: ScaredA00Sample,
        *,
        joint: bool,
        teacher_target: AlignedTeacherTarget | None = None,
        position: int = 0,
    ) -> dict[str, float]:
        """One bounded update; finite gradients checked before any optimizer step."""
        if sample.data_role != "final_train" or sample.sample_id.split("/")[0] not in {
            f"dataset_{i}" for i in range(1, 8)
        }:
            raise ValueError("student learning cannot consume final_test")
        if self.config.kd_weight and teacher_target is None:
            raise ValueError("KD requires explicit aligned frozen teacher supervision")
        self.model.train().requires_grad_(True)
        self.actions.train().requires_grad_(True)
        self.selector.train()
        for optimizer in (self.model_optimizer, self.action_optimizer, self.selector_optimizer):
            optimizer.zero_grad(set_to_none=True)
        state = self.state(sample)
        height, width = state.batch.left.shape[-2:]
        full = grid_regions((height, width), (height, width), self.routing.halo)[0]
        base = self.actions.execute(state, full, Action.STOP, self.model)
        rgb = reconstruction_loss(state, base)
        kd = rgb.new_zeros(())
        if self.config.kd_weight:
            assert teacher_target is not None
            target = teacher_target.disparity.detach().to(state.batch.left.device).clone()
            if target.shape != base.disparity.shape:
                raise ValueError("KD target must be aligned to student grid")
            self.model._validate_target_metadata(teacher_target, calibration_metadata=None)
            mask = (
                state.batch.support
                & teacher_target.support.to(target.device)
                & teacher_target.validity.to(target.device)
                & torch.isfinite(target)
            )
            if not bool(mask.any()):
                raise ValueError("empty KD supervision mask")
            # Mean SmoothL1 in input-grid pixels. Teacher disparity is detached.
            kd = F.smooth_l1_loss(base.disparity[mask], target[mask])
        repair = rgb.new_zeros(())
        selector_loss = rgb.new_zeros(())
        if joint:
            regions = grid_regions((height, width), self.routing.tile_hw, self.routing.halo)
            candidates = [region for region in regions if eligible(state, region)]
            if not candidates:
                raise ValueError("no eligible joint intervention regions")
            region = candidates[position % len(candidates)]
            features = region_features(state, (region,)).detach()
            # Measured alternatives before updating any optimizer, fixed weights.
            self.actions.requires_grad_(False)
            try:
                rows = self.harness.collect(state, region, self.actions, self.model)
            finally:
                self.actions.requires_grad_(True)
            labels = features.new_tensor([[row.utility for row in rows]])
            selector_loss = F.mse_loss(self.selector(features)[:, 1:], labels[:, 1:])
            repair = torch.stack(
                [
                    reconstruction_loss(
                        state, self.actions.execute(state, full, action, self.model)
                    )
                    for action in (Action.STEREO, Action.GAUSSIAN)
                ]
            ).mean()
            costs = features.new_tensor([row.cost_ms for row in rows])
        loss = (
            self.config.reconstruction_weight * rgb
            + self.config.kd_weight * kd
            + (self.config.repair_weight * repair if joint else 0)
        )
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite student loss")
        loss.backward()
        if joint:
            selector_loss.backward()
        groups: list[tuple[nn.Module, torch.optim.Adam]] = [(self.model, self.model_optimizer)]
        if joint:
            groups += [
                (self.actions, self.action_optimizer),
                (self.selector, self.selector_optimizer),
            ]
        for module, _ in groups:
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                raise ValueError("missing/nonfinite student/joint gradients")
            nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
        for _, optimizer in groups:
            optimizer.step()
        if not all(
            bool(torch.isfinite(p).all()) for module, _ in groups for p in module.parameters()
        ):
            raise ValueError("nonfinite student/joint parameters")
        if joint:
            self.cost_max_ms = torch.maximum(self.cost_max_ms, costs)
            self.cost_observations += 1
        return {
            "loss": float(loss.detach()),
            "reconstruction": float(rgb.detach()),
            "kd": float(kd.detach()),
            "repair": float(repair.detach()),
            "selector": float(selector_loss.detach()),
        }

    def infer(
        self, sample: ScaredA00Sample, *, selected: bool = False
    ) -> ReferenceReconstructionResult:
        """Student-only deployment/evaluation; no teacher, labels, or state updates."""
        with evaluation_mode(self.model, self.actions, self.selector):
            state = self.state(sample)
            if not selected:
                return state.base
            if not self.cost_observations:
                raise ValueError("selected inference requires measured cost calibration")
            height, width = state.batch.left.shape[-2:]
            regions = grid_regions((height, width), self.routing.tile_hw, self.routing.halo)
            features = region_features(state, regions)
            valid = torch.tensor([eligible(state, r) for r in regions], device=features.device)
            decision = select_action(
                self.selector(features),
                self.cost_max_ms.expand(len(regions), -1),
                valid,
                self.routing.budget_ms,
            )
            region = regions[decision.region_index or 0]
            outcome = self.actions.execute(state, region, decision.action, self.model)
            return replace(
                state.base,
                gaussians=outcome.field,
                render_left=outcome.left,
                render_right=outcome.right,
                stereo=replace(state.base.stereo, disparity=outcome.disparity),
                target=replace(state.base.target, disparity=outcome.disparity),
                diagnostics={
                    **state.base.diagnostics,
                    "selected_action": int(decision.action),
                    "prior_is_pre_action_correspondence": True,
                },
            )
