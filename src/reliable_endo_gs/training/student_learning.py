"""Student reconstruction/KD then alternating joint reconstruction/utility updates.

The selector has no upstream gradient: action choice remains discrete. Joint
reconstruction explicitly trains both repairs from the SAME student base, not
a chained/selected action. Utility supervision observes current state before
the update, so moving student/repair weights do not reuse stale frozen labels.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from reliable_endo_gs.actions.repair import (
    Action,
    RepairActions,
    RepairState,
    eligible,
    primitive_core,
)
from reliable_endo_gs.actions.stereo_matching import MatchingStereoRepair
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.data.target_masks import TargetMaskIndex
from reliable_endo_gs.reconstruction.model import ReferenceReconstructionResult
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.regions.grid import Region, grid_regions
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.config import RoutingConfig
from reliable_endo_gs.routing.context import balanced_features
from reliable_endo_gs.routing.masks import MaskProposalConfig, mask_features, propose_masks
from reliable_endo_gs.routing.repairability import (
    RepairabilitySelector,
    repairability_features,
    select_repairability,
)
from reliable_endo_gs.routing.selector import (
    ActionSelector,
    Decision,
    region_features,
    select_action,
)
from reliable_endo_gs.teachers import AlignedTeacherTarget
from reliable_endo_gs.training.a04_reference import _move_batch
from reliable_endo_gs.training.gradient_balance import GradientBalance
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
        *,
        intervention_config: InterventionConfig | None = None,
        mask_config: MaskProposalConfig | None = None,
        target_masks: TargetMaskIndex | None = None,
    ) -> None:
        self.model, self.config, self.routing = model, config, routing
        if (mask_config is None) != (target_masks is None):
            raise ValueError("mask routing requires both proposal config and deployment ROI index")
        if mask_config is not None and (
            intervention_config is None or intervention_config.schema != "intervention_selection.v2"
        ):
            raise ValueError("mask routing requires balanced local/global/collateral supervision")
        self.mask_config, self.target_masks = mask_config, target_masks
        self.intervention_config = intervention_config
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
        matching_head = (
            None
            if intervention_config is None
            else MatchingStereoRepair(
                intervention_config.matching_width,
                intervention_config.search_radius_px,
                intervention_config.search_step_px,
                intervention_config.matching_temperature,
                routing.width,
            )
        )
        self.actions = RepairActions(
            model,
            width=routing.width,
            max_disparity_px=routing.max_disparity_px
            if intervention_config is None
            else float(intervention_config.search_radius_px),
            composition_alpha_epsilon=routing.composition_alpha_epsilon,
            matching_head=matching_head,
            effective_fraction=0.25
            if intervention_config is None
            else intervention_config.max_effective_fraction,
        ).to(device)
        self.selector: ActionSelector = (
            ActionSelector(routing.width)
            if intervention_config is None
            else RepairabilitySelector(routing.width)
        ).to(device)
        self.objective_balancer: GradientBalance | None = None
        self.selector_balancer: GradientBalance | None = None
        if (
            intervention_config is not None
            and intervention_config.schema == "intervention_selection.v2"
        ):
            self.selector = BalancedSelector(routing.width, intervention_config).to(device)
            self.objective_balancer = GradientBalance(
                ("rgb", "warp", "kd", "repair"), intervention_config
            ).to(device)
            self.selector_balancer = GradientBalance(
                ("global", "local", "ranking"), intervention_config
            ).to(device)
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.action_optimizer = torch.optim.Adam(
            self.actions.parameters(), lr=routing.learning_rate
        )
        self.selector_optimizer = torch.optim.Adam(
            self.selector.parameters(), lr=routing.learning_rate
        )
        self.harness: InterventionHarness = InterventionHarness(routing.cost_weight_per_ms)
        if intervention_config is not None:
            from reliable_endo_gs.training.repairability import RepairabilityHarness

            self.harness = RepairabilityHarness(intervention_config)
            if intervention_config.schema == "intervention_selection.v2":
                from reliable_endo_gs.training.balanced_interventions import BalancedHarness

                self.harness = BalancedHarness(intervention_config)
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
            include_matching_evidence=self.intervention_config is not None
            and self.intervention_config.schema == "intervention_selection.v2",
        )
        return RepairState(batch, result)

    def selection_features(self, state: RepairState, regions: tuple[Region, ...]) -> torch.Tensor:
        """Canonical feature dispatch shared by training, inference and benchmark."""
        if self.intervention_config is None:
            return region_features(state, regions)
        if self.mask_config is not None:
            return mask_features(
                state,
                regions,
                self.intervention_config,
                self.mask_config,
                scale_unit_m=self.model.config.context_scale_m,
            )
        producer = (
            balanced_features
            if self.intervention_config.schema == "intervention_selection.v2"
            else repairability_features
        )
        return producer(
            state, regions, self.intervention_config, scale_unit_m=self.model.config.context_scale_m
        )

    def candidate_regions(self, state: RepairState, sample_id: str) -> tuple[Region, ...]:
        """Same proposal owner for real interventions, learning and inference."""
        h, w = state.batch.left.shape[-2:]
        if self.mask_config is None:
            return grid_regions((h, w), self.routing.tile_hw, self.routing.halo)
        assert self.target_masks is not None
        target = self.target_masks.load(sample_id, (h, w), state.batch.left.device)
        return propose_masks(state, target, self.mask_config, halo=self.routing.halo)

    def step(
        self,
        sample: ScaredA00Sample,
        *,
        joint: bool,
        teacher_target: AlignedTeacherTarget | None = None,
        position: int = 0,
    ) -> Mapping[str, object]:
        """One bounded update; finite gradients checked before any optimizer step."""
        if self.intervention_config is not None:
            if self.intervention_config.schema == "intervention_selection.v2":
                from reliable_endo_gs.training.balanced_learning import balanced_step

                return balanced_step(
                    self, sample, joint=joint, teacher_target=teacher_target, position=position
                )
            from reliable_endo_gs.training.repairability import repairability_step

            return repairability_step(
                self, sample, joint=joint, teacher_target=teacher_target, position=position
            )
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
        diagnostics: dict[str, float] = {}
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
            diagnostics.update({f"utility_{row.action.name.lower()}": row.utility for row in rows})
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
        diagnostics.update(self.commit_gradients(joint))
        if joint:
            self.cost_max_ms = torch.maximum(self.cost_max_ms, costs)
            self.cost_observations += 1
        return {
            "loss": float(loss.detach()),
            "reconstruction": float(rgb.detach()),
            "kd": float(kd.detach()),
            "repair": float(repair.detach()),
            "selector": float(selector_loss.detach()),
            **diagnostics,
        }

    def commit_gradients(self, joint: bool) -> dict[str, float]:
        """One shared finite-gradient/ownership/update path for both protocols."""
        diagnostics: dict[str, float] = {}
        groups: list[tuple[nn.Module, torch.optim.Adam]] = [(self.model, self.model_optimizer)]
        if joint:
            groups += [
                (self.actions, self.action_optimizer),
                (self.selector, self.selector_optimizer),
            ]
        components = {
            "student": self.model.stereo_student,
            "head": self.model.head,
        }
        if joint:
            components.update(
                stereo_repair=self.actions.stereo,
                gaussian_repair=self.actions.gaussian,
                selector=self.selector,
            )
        before_updates = {}
        if isinstance(self.selector, BalancedSelector):
            before_updates = {
                name: tuple(p.detach().clone() for p in module.parameters())
                for name, module in components.items()
            }
        for name, module in components.items():
            component_gradients = [p.grad for p in module.parameters()]
            if any(g is None or not bool(torch.isfinite(g).all()) for g in component_gradients):
                raise ValueError(f"missing/nonfinite gradients in {name}")
            diagnostics[f"grad_norm_{name}"] = float(
                torch.linalg.vector_norm(
                    torch.stack(
                        [
                            torch.linalg.vector_norm(g.detach())
                            for g in component_gradients
                            if g is not None
                        ]
                    )
                )
            )
        for module, _ in groups:
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                raise ValueError("missing/nonfinite student/joint gradients")
            if isinstance(module, BalancedSelector):
                for network in (module.gain_network, module.harm_network):
                    nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
            else:
                nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
        for _, optimizer in groups:
            optimizer.step()
        if before_updates:
            for name, module in components.items():
                before = before_updates[name]
                norm = torch.linalg.vector_norm(torch.stack([p.norm() for p in before]))
                update = torch.linalg.vector_norm(
                    torch.stack(
                        [
                            (p.detach() - old).norm()
                            for p, old in zip(module.parameters(), before, strict=True)
                        ]
                    )
                )
                diagnostics[f"parameter_norm_before_{name}"] = float(norm)
                diagnostics[f"optimizer_update_norm_{name}"] = float(update)
                diagnostics[f"relative_update_norm_{name}"] = float(update / norm.clamp_min(1e-12))
                gradients = [p.grad for p in module.parameters() if p.grad is not None]
                diagnostics[f"grad_norm_after_clip_{name}"] = float(
                    torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in gradients]))
                )
        if not all(
            bool(torch.isfinite(p).all()) for module, _ in groups for p in module.parameters()
        ):
            raise ValueError("nonfinite student/joint parameters")
        return diagnostics

    def infer(
        self, sample: ScaredA00Sample, *, selected: bool = False
    ) -> ReferenceReconstructionResult:
        """Student-only deployment/evaluation; no teacher, labels, or state updates."""
        with evaluation_mode(self.model, self.actions, self.selector):
            device = next(self.model.parameters()).device
            if self.intervention_config is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            state = self.state(sample)
            if self.intervention_config is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            base_ms = (time.perf_counter() - started) * 1000
            if not selected:
                return state.base
            started = time.perf_counter()
            regions = self.candidate_regions(state, sample.sample_id)
            if not regions:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                routing_ms = (time.perf_counter() - started) * 1000
                return replace(
                    state.base,
                    diagnostics={
                        **state.base.diagnostics,
                        "selected_action": int(Action.STOP),
                        "mask_stop_reason": "no_eligible_target_candidates",
                        "mask_config_hash": None
                        if self.mask_config is None
                        else self.mask_config.config_hash,
                        "target_mask_index_sha256": None
                        if self.target_masks is None
                        else self.target_masks.identity,
                        "base_ms": base_ms,
                        "routing_ms": routing_ms,
                        "actual_action_ms": 0.0,
                        "actual_policy_ms": base_ms + routing_ms,
                    },
                )
            if isinstance(self.selector, BalancedSelector) and not bool(
                self.selector.normalization.fitted
            ):
                raise ValueError("balanced deployment requires fitted training normalization")
            if not self.cost_observations:
                raise ValueError("selected inference requires measured cost calibration")
            features = (
                region_features(state, regions)
                if self.intervention_config is None
                else self.selection_features(
                    state,
                    regions,
                )
            )
            valid = torch.tensor([eligible(state, r) for r in regions], device=features.device)
            budget_metadata: dict[str, float] = {}
            if self.intervention_config is None:
                decision = select_action(
                    self.selector(features),
                    self.cost_max_ms.expand(len(regions), -1),
                    valid,
                    self.routing.budget_ms,
                )
            else:
                assert isinstance(self.selector, RepairabilitySelector)
                gains, harms = self.selector.predict(features)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                routing_ms = (time.perf_counter() - started) * 1000
                remaining = max(
                    0.0, self.intervention_config.total_budget_ms - base_ms - routing_ms
                )
                decision = select_repairability(
                    gains,
                    harms,
                    self.cost_max_ms.expand(len(regions), -1),
                    valid,
                    remaining,
                    self.intervention_config,
                )
                budget_metadata = {
                    # Include allocation/host synchronization in routing cost.
                    "base_ms": base_ms,
                    "routing_ms": routing_ms,
                    "remaining_ms": remaining,
                    "estimated_action_ms": decision.estimated_ms,
                }
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                routing_ms = (time.perf_counter() - started) * 1000
                remaining = max(
                    0.0, self.intervention_config.total_budget_ms - base_ms - routing_ms
                )
                if decision.estimated_ms > remaining:
                    decision = Decision(None, Action.STOP, 0.0, 0.0)
                budget_metadata.update(
                    routing_ms=routing_ms,
                    remaining_ms=remaining,
                    estimated_action_ms=decision.estimated_ms,
                )
            region = regions[decision.region_index or 0]
            started = time.perf_counter()
            outcome = self.actions.execute(state, region, decision.action, self.model)
            if self.intervention_config is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            if self.intervention_config is not None:
                action_ms = (time.perf_counter() - started) * 1000
                budget_metadata.update(
                    actual_action_ms=action_ms,
                    actual_policy_ms=base_ms + budget_metadata["routing_ms"] + action_ms,
                    budget_overrun_ms=max(
                        0.0,
                        base_ms
                        + budget_metadata["routing_ms"]
                        + action_ms
                        - self.intervention_config.total_budget_ms,
                    ),
                )
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
                    "region_protocol": "fixed_grid.v1"
                    if self.mask_config is None
                    else self.mask_config.schema,
                    "selected_pixels": 0 if decision.action == Action.STOP else region.pixel_count,
                    "selected_primitives": 0
                    if decision.action == Action.STOP
                    else int(primitive_core(state, region).sum()),
                    "mask_config_hash": None
                    if self.mask_config is None
                    else self.mask_config.config_hash,
                    "target_mask_index_sha256": None
                    if self.target_masks is None
                    else self.target_masks.identity,
                    **budget_metadata,
                },
            )
