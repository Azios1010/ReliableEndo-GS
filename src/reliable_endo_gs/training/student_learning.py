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
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from reliable_endo_gs.actions.repair import (
    Action,
    RepairActions,
    RepairState,
    action_eligible,
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
from reliable_endo_gs.routing.artifact_identity import (
    repair_execution_config,
    repair_execution_source_sha256,
)
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.calibrated_selector import (
    CalibratedSelector,
    load_calibrated_selector,
)
from reliable_endo_gs.routing.config import RoutingConfig
from reliable_endo_gs.routing.context import balanced_features
from reliable_endo_gs.routing.cost import (
    DeploymentCostCalibration,
    calibration_metadata,
    capture_runtime_identity,
    estimated_costs,
    new_cost_history,
    record_costs,
)
from reliable_endo_gs.routing.masks import MaskProposalConfig, mask_features, propose_masks
from reliable_endo_gs.routing.metric_selection import MetricRepairSelector, module_state_sha256
from reliable_endo_gs.routing.outcome_predictor import load_outcome_predictor
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
from reliable_endo_gs.routing.support_grid import AutomaticSupportGrid
from reliable_endo_gs.routing.tokens import mask_feature_tokens
from reliable_endo_gs.routing.tradeoff import TradeoffConfig
from reliable_endo_gs.teachers import AlignedTeacherTarget
from reliable_endo_gs.training.a04_reference import _move_batch
from reliable_endo_gs.training.distillation import DisparityTarget, stereo_distillation
from reliable_endo_gs.training.gradient_balance import GradientBalance
from reliable_endo_gs.training.interventions import (
    InterventionHarness,
    intervention_costs,
    reconstruction_loss,
)
from reliable_endo_gs.training.metric_quality import OfflineMetricTruth
from reliable_endo_gs.training.selector_optimization import make_selector_balancer
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
        target_masks: TargetMaskIndex | AutomaticSupportGrid | None = None,
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
        if (
            intervention_config is not None
            and intervention_config.selector_feature_mode == "mask_tokens"
        ):
            if mask_config is None or intervention_config.selector_token_channels != config.width:
                raise ValueError(
                    "mask tokens require target masks and channels matching student width"
                )
        self.student_frozen = (
            intervention_config is not None and intervention_config.student_update_mode == "frozen"
        )
        if self.student_frozen and config.student_epochs:
            raise ValueError("frozen student comparison excludes student-only warmup")
        if self.student_frozen and config.model_lr_schedule != "constant":
            raise ValueError("frozen student must preserve constant model Adam settings")
        self.frozen_student_start_step: int | None = 0 if self.student_frozen else None
        if (
            model.stereo_student.width != config.width
            or model.stereo_student.architecture != config.stereo_architecture
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
            # Matching search is evidence collection; it must never widen the
            # physical disparity correction bound.  Legacy callers without an
            # intervention protocol retain the routing bound, while v2 may
            # declare an explicit action bound.
            max_disparity_px=(
                routing.max_disparity_px
                if intervention_config is None
                or intervention_config.repair_max_disparity_px is None
                else intervention_config.repair_max_disparity_px
            ),
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
                ("rgb", "warp", "kd", "repair")
                + (("geometry",) if intervention_config.geometry_supervision_weight else ()),
                intervention_config,
            ).to(device)
            self.selector_balancer = make_selector_balancer(intervention_config).to(device)
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.action_optimizer = torch.optim.Adam(
            self.actions.parameters(), lr=routing.learning_rate
        )
        if (
            isinstance(self.selector, BalancedSelector)
            and self.selector.config.selector_gain_only_refinement
        ):
            self.selector_optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.selector.gain_network.parameters(),
                        "name": "gain",
                        "weight_decay": self.selector.config.selector_weight_decay,
                    },
                    {
                        "params": self.selector.harm_network.parameters(),
                        "name": "harm",
                        "weight_decay": 0.0,
                    },
                ],
                lr=routing.learning_rate,
            )
        else:
            self.selector_optimizer = torch.optim.Adam(
                self.selector.parameters(),
                lr=routing.learning_rate,
                weight_decay=0.0
                if intervention_config is None
                else intervention_config.selector_weight_decay,
            )
        self.optimizer_update_counts = {"model": 0, "actions": 0, "selector": 0}
        self.action_update_counts = {"stereo": 0, "gaussian": 0}
        self.harness: InterventionHarness = InterventionHarness(routing.cost_weight_per_ms)
        if intervention_config is not None:
            from reliable_endo_gs.training.repairability import RepairabilityHarness

            self.harness = RepairabilityHarness(intervention_config)
            if intervention_config.schema == "intervention_selection.v2":
                from reliable_endo_gs.training.balanced_interventions import BalancedHarness

                self.harness = BalancedHarness(intervention_config)
        self.cost_max_ms = torch.zeros(3, device=device)
        self.cost_observations = 0
        self.cost_history = new_cost_history(device)
        self.cost_history_observations = 0
        self.deployment_cost_calibration: DeploymentCostCalibration | None = None
        self.metric_selector: MetricRepairSelector | CalibratedSelector | None = None
        self._frozen_versions: tuple[int, ...] | None = None
        self._cost_versions: tuple[int, ...] | None = None

    def frozen_action_identity(self) -> dict[str, object]:
        """Identity of the exact model/actions that produce outcome labels."""
        return {
            "model_state_sha256": module_state_sha256(self.model),
            "action_state_sha256": module_state_sha256(self.actions),
            "student_config_hash": self.config.config_hash,
            "routing_config_hash": self.routing.config_hash,
            "intervention_config_hash": None
            if self.intervention_config is None
            else self.intervention_config.config_hash,
            "mask_config_hash": None if self.mask_config is None else self.mask_config.config_hash,
        }

    def operative_action_identity(self) -> dict[str, object]:
        """Frozen forward identity shared across study variants; legacy resume is unchanged."""
        return {
            "schema": "operative_repair_identity.v1",
            "execution_source_sha256": repair_execution_source_sha256(),
            "renderer_identity": self.model._renderer_identity(),
            "model_state_sha256": module_state_sha256(self.model),
            "action_state_sha256": module_state_sha256(self.actions),
            "construction_config_hash": self.model.config.config_hash,
            "student_config_hash": self.config.config_hash,
            "routing_config_hash": self.routing.config_hash,
            "repair_execution_config": None
            if self.intervention_config is None
            else repair_execution_config(self.intervention_config),
        }

    def candidate_protocol_identity(self) -> dict[str, object]:
        """Bind proposal limits/ROI separately from immutable repair weights."""
        return {
            "schema": "repair_candidate_protocol.v1",
            "mask_config_hash": None if self.mask_config is None else self.mask_config.config_hash,
            "target_mask_identity": None
            if self.target_masks is None
            else self.target_masks.identity,
            "tile_hw": list(self.routing.tile_hw),
            "halo": self.routing.halo,
        }

    def load_metric_selector(self, path: Path, config: TradeoffConfig, candidate_hash: str) -> None:
        """Bind an offline TRAIN-fitted/calibrated predictor to frozen actions."""
        if self.intervention_config is None or self.mask_config is None:
            raise ValueError("metric mask selector requires the mask intervention protocol")
        if self.intervention_config.selector_feature_mode != "statistics":
            raise ValueError(
                "historical outcome predictor expects statistics; use the trained mask-token selector"
            )
        identity = self.frozen_action_identity()
        predictor = load_outcome_predictor(
            path,
            expected_config=config,
            expected_action_identity=identity,
            expected_candidate_hash=candidate_hash,
        ).to(next(self.model.parameters()).device)
        self.metric_selector = MetricRepairSelector(predictor, config, identity, candidate_hash)
        self._frozen_versions = tuple(
            p._version for m in (self.model, self.actions) for p in m.parameters()
        )

    def attach_cost_calibration(self, calibration: DeploymentCostCalibration) -> None:
        """Attach an explicit deployment timing artifact.

        Training timing history remains diagnostic; deployment allocation uses
        this artifact only after its runtime and output-grid identity match.
        """
        operative = calibration.binding.get("operative_action_identity")
        if (
            operative is not None
            and calibration.binding.get("candidate_protocol_identity")
            != self.candidate_protocol_identity()
        ):
            raise ValueError("deployment cost calibration candidate protocol mismatch")
        if calibration.binding.get("collection_role") != "train" or (
            operative != self.operative_action_identity()
            if operative is not None
            else calibration.binding.get("frozen_action_identity") != self.frozen_action_identity()
        ):
            raise ValueError("cost calibration requires exact frozen TRAIN model/action identity")
        if "target_mask_index_sha256" in calibration.binding and (
            calibration.binding["target_mask_index_sha256"]
            != (None if self.target_masks is None else self.target_masks.identity)
        ):
            raise ValueError("deployment cost calibration target-mask identity mismatch")
        self.deployment_cost_calibration = calibration
        self._cost_versions = tuple(
            p._version for m in (self.model, self.actions) for p in m.parameters()
        )

    def load_refit_selector(
        self, path: Path, source_sha256: str, *, study_artifact: bool = False
    ) -> None:
        """Attach a TRAIN-only refit; any subsequent joint update invalidates it."""
        if self.intervention_config is None or self.target_masks is None:
            raise ValueError("refit selection requires physical masked reconstruction")
        self.metric_selector = load_calibrated_selector(
            path,
            expected_action_identity=self.operative_action_identity()
            if study_artifact
            else self.frozen_action_identity(),
            expected_config=self.intervention_config,
            expected_mask_identity=self.target_masks.identity,
            expected_source_sha256=source_sha256,
            **(
                {"expected_candidate_identity": self.candidate_protocol_identity()}
                if study_artifact
                else {}
            ),
        ).to(next(self.model.parameters()).device)
        self._frozen_versions = tuple(
            p._version for m in (self.model, self.actions) for p in m.parameters()
        )

    def record_action_costs(
        self, costs_ms: torch.Tensor, action_available: torch.Tensor | None = None
    ) -> None:
        """Persist measured full-frame action costs for robust allocation."""
        rows = costs_ms.reshape(-1, 3)
        if action_available is not None:
            available = action_available.reshape(-1, 3)
            if available.shape != rows.shape or available.dtype != torch.bool:
                raise ValueError("action cost availability must be bool [R,3]")
            # A cost row is only a complete timing observation when both real
            # repairs were actually executed.  Never turn an unavailable action
            # into a zero-latency sample.
            rows = rows[available.all(1)]
            if rows.numel() == 0:
                return
        self.cost_max_ms = torch.maximum(self.cost_max_ms, rows.detach().amax(0))
        self.cost_history_observations = record_costs(
            self.cost_history, rows, self.cost_history_observations
        )
        # One observation is one training update; history_observations keeps
        # the finer per-region count separately for diagnostics.
        self.cost_observations += 1

    def estimated_action_costs(self, state: RepairState | None = None) -> torch.Tensor:
        if self.deployment_cost_calibration is not None:
            if state is None:
                raise ValueError("deployment cost calibration requires a deployment state")
            calibration = self.deployment_cost_calibration
            versions = tuple(p._version for m in (self.model, self.actions) for p in m.parameters())
            if versions != self._cost_versions:
                raise ValueError("deployment timing action state changed; recalibrate")
            runtime = capture_runtime_identity(state.batch.left.device)
            shape = (int(state.batch.left.shape[-2]), int(state.batch.left.shape[-1]))
            if not calibration.matches(calibration.binding, runtime, shape):
                raise ValueError("deployment cost calibration runtime/grid mismatch")
            return calibration.estimated_ms.to(self.cost_max_ms)
        return estimated_costs(
            self.cost_history,
            self.cost_max_ms,
            self.cost_observations,
            history_observations=self.cost_history_observations,
        )

    def cost_calibration_metadata(self) -> dict[str, object]:
        return calibration_metadata(
            self.cost_history,
            self.cost_max_ms,
            self.cost_observations,
            history_observations=self.cost_history_observations,
            calibration=self.deployment_cost_calibration,
        )

    def state(
        self, sample: ScaredA00Sample, *, include_feature_tokens: bool = False
    ) -> RepairState:
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
            include_feature_tokens=include_feature_tokens
            or (
                self.intervention_config is not None
                and self.intervention_config.selector_feature_mode == "mask_tokens"
            ),
        )
        return RepairState(batch, result)

    def selection_features(self, state: RepairState, regions: tuple[Region, ...]) -> torch.Tensor:
        """Canonical feature dispatch shared by training, inference and benchmark."""
        if self.intervention_config is None:
            return region_features(state, regions)
        if self.mask_config is not None:
            features = mask_features(
                state,
                regions,
                self.intervention_config,
                self.mask_config,
                scale_unit_m=self.model.config.context_scale_m,
            )
            if self.intervention_config.selector_feature_mode == "mask_tokens":
                tokens = mask_feature_tokens(
                    state, regions, self.intervention_config.selector_token_channels
                )
                if self.mask_config.feature_evidence == "gaussian":
                    tokens = torch.zeros_like(tokens)
                features = torch.cat((features, tokens), 1)
            return features.detach()
        if self.intervention_config.selector_feature_mode == "mask_tokens":
            raise ValueError("mask token selector requires the deployment target-mask contract")
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
        if isinstance(self.target_masks, AutomaticSupportGrid):
            return self.target_masks.propose(state, halo=self.routing.halo)
        target = self.target_masks.load(sample_id, (h, w), state.batch.left.device)
        return propose_masks(state, target, self.mask_config, halo=self.routing.halo)

    def step(
        self,
        sample: ScaredA00Sample,
        *,
        joint: bool,
        teacher_target: AlignedTeacherTarget | None = None,
        position: int = 0,
        metric_truth: OfflineMetricTruth | None = None,
        selector_context: tuple[tuple[ScaredA00Sample, OfflineMetricTruth], ...] = (),
        gt_target: DisparityTarget | None = None,
    ) -> Mapping[str, object]:
        """One bounded update; finite gradients checked before any optimizer step.

        ``gt_target`` (required iff ``config.gt_weight``) is the TRAIN ground-truth disparity of the plain student path; it goes
        through the same ``stereo_distillation`` loss as the teacher target, with its diagnostics renamed ``gt_*``.
        """
        if self.config.gt_weight and (gt_target is None or self.intervention_config is not None):
            raise ValueError(
                "GT-disparity supervision needs an explicit GT target and the plain student path"
            )
        if gt_target is not None and not self.config.gt_weight:
            raise ValueError("a GT target was given but gt_weight is zero")
        if metric_truth is not None and (
            self.intervention_config is None or self.intervention_config.metric_quality is None
        ):
            raise ValueError("offline truth requires explicit physical metric supervision")
        if selector_context and (
            not joint
            or self.intervention_config is None
            or self.intervention_config.metric_quality is None
            or self.intervention_config.schema != "intervention_selection.v2"
        ):
            raise ValueError("fresh selector context requires physical joint learning")
        # An optimizer update changes the counterfactual label/cost generator.
        # No frozen predictor or deployment timing artifact may survive it.
        self.metric_selector = None
        self.deployment_cost_calibration = None
        self._frozen_versions = None
        self._cost_versions = None
        if self.intervention_config is not None:
            if self.intervention_config.schema == "intervention_selection.v2":
                from reliable_endo_gs.training.balanced_learning import balanced_step

                return balanced_step(
                    self,
                    sample,
                    joint=joint,
                    teacher_target=teacher_target,
                    position=position,
                    metric_truth=metric_truth,
                    selector_context=selector_context,
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
        rgb = reconstruction_loss(
            state, base, mask_right_strip=self.config.right_visibility_mask
        )
        kd = rgb.new_zeros(())
        diagnostics: dict[str, float] = {}
        if self.config.kd_weight:
            assert teacher_target is not None
            self.model._validate_target_metadata(teacher_target, calibration_metadata=None)
            distillation = stereo_distillation(base.disparity, teacher_target, state.batch)
            kd = distillation.loss
            diagnostics.update(distillation.diagnostics)
        gt = rgb.new_zeros(())
        if self.config.gt_weight:
            assert gt_target is not None
            ground_truth = stereo_distillation(base.disparity, gt_target, state.batch)
            gt = ground_truth.loss
            diagnostics.update(
                {key.replace("kd_", "gt_", 1): value for key, value in ground_truth.diagnostics.items()}
            )
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
            costs, action_available = intervention_costs(rows, features)
            selector_ready = bool(action_available[1:].any())
            selector_error = F.mse_loss(
                self.selector(features)[:, 1:], labels[:, 1:], reduction="none"
            )
            selector_loss = (selector_error * action_available[1:]).sum() / (
                action_available[1:].sum().clamp_min(1)
            )
            repair_losses = [
                reconstruction_loss(state, self.actions.execute(state, full, action, self.model))
                for action in (Action.STEREO, Action.GAUSSIAN)
                if action_eligible(state, full, action)
            ]
            repair = torch.stack(repair_losses).mean() if repair_losses else rgb * 0
            diagnostics.update({f"utility_{row.action.name.lower()}": row.utility for row in rows})
        loss = (
            self.config.reconstruction_weight * rgb
            + self.config.kd_weight * kd
            + (self.config.gt_weight * gt if self.config.gt_weight else 0)
            + (self.config.repair_weight * repair if joint else 0)
        )
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite student loss")
        loss.backward()
        if joint and selector_ready:
            selector_loss.backward()
        diagnostics.update(
            self.commit_gradients(
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
            self.record_action_costs(costs, action_available)
        return {
            "loss": float(loss.detach()),
            "reconstruction": float(rgb.detach()),
            "kd": float(kd.detach()),
            **({"gt": float(gt.detach())} if self.config.gt_weight else {}),
            "repair": float(repair.detach()),
            "selector": float(selector_loss.detach()),
            **diagnostics,
        }

    def commit_gradients(
        self,
        joint: bool,
        *,
        update_selector: bool = True,
        selector_only: bool = False,
        active_repairs: tuple[Action, ...] | None = None,
    ) -> dict[str, float]:
        """One shared finite-gradient/ownership/update path for both protocols."""
        update_selector = joint and update_selector
        if selector_only and (
            not update_selector or not isinstance(self.selector, BalancedSelector)
        ):
            raise ValueError("selector-only updates require joint balanced supervision")
        diagnostics: dict[str, float] = {}
        groups: list[tuple[nn.Module, torch.optim.Adam]] = (
            [] if self.student_frozen else [(self.model, self.model_optimizer)]
        )
        if active_repairs is None:
            active_repairs = (Action.STEREO, Action.GAUSSIAN) if joint else ()
        if len(set(active_repairs)) != len(active_repairs) or any(
            action not in (Action.STEREO, Action.GAUSSIAN) for action in active_repairs
        ):
            raise ValueError("active repairs must be unique stereo/Gaussian actions")
        repair_components = {
            name: module
            for action, name, module in (
                (Action.STEREO, "stereo", self.actions.stereo),
                (Action.GAUSSIAN, "gaussian", self.actions.gaussian),
            )
            if joint and action in active_repairs
        }
        if joint:
            if repair_components:
                groups.append((self.actions, self.action_optimizer))
            if update_selector:
                groups.append((self.selector, self.selector_optimizer))
        components = (
            {}
            if self.student_frozen
            else {
                "student": self.model.stereo_student,
                "head": self.model.head,
            }
        )
        if joint:
            components.update(
                {f"{name}_repair": module for name, module in repair_components.items()}
            )
            if update_selector:
                components["selector"] = self.selector
        if selector_only:
            groups = [(self.selector, self.selector_optimizer)]
            components = {
                "selector": self.selector.gain_network
                if isinstance(self.selector, BalancedSelector)
                and self.selector.config.selector_gain_only_refinement
                else self.selector
            }
            if (
                isinstance(self.selector, BalancedSelector)
                and self.selector.config.selector_gain_only_refinement
                and any(p.grad is not None for p in self.selector.harm_network.parameters())
            ):
                raise ValueError("gain-only refinement must not carry harm gradients")
        before_updates = {}
        if isinstance(self.selector, BalancedSelector):
            if update_selector:
                components.update(
                    {
                        f"selector_{name}": network
                        for name, network in self.selector.optimization_components()
                        if any(p.grad is not None for p in network.parameters())
                    }
                )
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
                for network in module.optimization_networks():
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
        if not selector_only and not self.student_frozen:
            self.optimizer_update_counts["model"] += 1
        if self.student_frozen and not selector_only:
            # Model gradients exist solely for matched balancing probes and
            # input derivatives through the frozen head. Never step its Adam.
            self.model_optimizer.zero_grad(set_to_none=True)
        if joint and repair_components and not selector_only:
            self.optimizer_update_counts["actions"] += 1
            for name in repair_components:
                self.action_update_counts[name] += 1
        if update_selector:
            self.optimizer_update_counts["selector"] += 1
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
                        "full_frame_action_execution": False,
                        "cost_calibration": self.cost_calibration_metadata(),
                    },
                )
            if (
                self.metric_selector is None
                and isinstance(self.selector, BalancedSelector)
                and not bool(self.selector.normalization.fitted)
            ):
                raise ValueError("balanced deployment requires fitted training normalization")
            if (
                self.intervention_config is not None
                and self.intervention_config.metric_quality is not None
                and self.deployment_cost_calibration is None
            ):
                raise ValueError(
                    "physical metric deployment requires dedicated TRAIN inference cost calibration"
                )
            if not self.cost_observations and self.deployment_cost_calibration is None:
                raise ValueError("selected inference requires measured cost calibration")
            if self.metric_selector is not None:
                versions = tuple(
                    p._version for m in (self.model, self.actions) for p in m.parameters()
                )
                if self._frozen_versions != versions:
                    raise ValueError(
                        "frozen metric selector action state changed; recollect outcomes"
                    )
                if self.deployment_cost_calibration is None:
                    raise ValueError(
                        "metric deployment requires dedicated TRAIN inference cost calibration"
                    )
            features = (
                region_features(state, regions)
                if self.intervention_config is None
                else self.selection_features(
                    state,
                    regions,
                )
            )
            row_valid = torch.tensor(
                [eligible(state, r) for r in regions], device=features.device, dtype=torch.bool
            )
            # Current actions reject fields carrying covariance metadata because
            # they do not yet propagate the uncertainty tensors.  Encode this
            # as action-specific ineligibility instead of allowing the selector
            # to predict an executable repair that will raise at dispatch.
            covariance_available = (
                state.base.gaussians.cov_center is not None
                or state.base.gaussians.cov_effective is not None
            )
            action_valid = torch.stack(
                (
                    torch.zeros_like(row_valid),
                    torch.tensor(
                        [action_eligible(state, r, Action.STEREO) for r in regions],
                        device=features.device,
                    )
                    & (not covariance_available),
                    torch.tensor(
                        [action_eligible(state, r, Action.GAUSSIAN) for r in regions],
                        device=features.device,
                    )
                    & (not covariance_available),
                ),
                dim=1,
            )
            budget_metadata: dict[str, object] = {}
            estimated_cost = self.estimated_action_costs(state)
            if self.intervention_config is None:
                decision = select_action(
                    self.selector(features),
                    estimated_cost.expand(len(regions), -1),
                    action_valid,
                    self.routing.budget_ms,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                routing_ms = (time.perf_counter() - started) * 1000
                budget_metadata.update(
                    base_ms=base_ms,
                    routing_ms=routing_ms,
                    remaining_ms=max(0.0, self.routing.budget_ms - base_ms - routing_ms),
                    estimated_action_ms=decision.estimated_ms,
                )
            else:
                assert isinstance(self.selector, RepairabilitySelector)
                if self.metric_selector is None:
                    gains, harms = self.selector.predict(features)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                routing_ms = (time.perf_counter() - started) * 1000
                remaining = max(
                    0.0, self.intervention_config.total_budget_ms - base_ms - routing_ms
                )
                if self.metric_selector is not None:
                    decision = self.metric_selector.select(
                        features, estimated_cost.expand(len(regions), -1), action_valid, remaining
                    )
                else:
                    decision = select_repairability(
                        gains,
                        harms,
                        estimated_cost.expand(len(regions), -1),
                        action_valid,
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
            budget_metadata.update(
                cost_calibration=self.cost_calibration_metadata(),
                estimated_cost_ms=[float(x) for x in estimated_cost.detach().cpu()],
                max_cost_ms=[float(x) for x in self.cost_max_ms.detach().cpu()],
                full_frame_action_execution=bool(decision.action != Action.STOP),
                selector_target_event="physical_metric_guards"
                if self.metric_selector is not None
                else "uncalibrated_physical_metric_guards"
                if self.intervention_config is not None
                and self.intervention_config.metric_quality is not None
                else "uncalibrated_rgb_warp_proxy",
                metric_guard_config_hash=None
                if self.metric_selector is None
                else self.metric_selector.config.config_hash,
            )
            if (
                self.intervention_config is not None
                and self.intervention_config.metric_quality is not None
            ):
                budget_metadata.update(
                    metric_quality_config_hash=self.intervention_config.config_hash,
                    action_risk_cutoffs=[
                        self.intervention_config.maximum_stereo_harm_probability,
                        self.intervention_config.maximum_gaussian_harm_probability,
                    ],
                    gaussian_guard_multiplier=self.intervention_config.gaussian_guard_multiplier,
                    risk_calibrated=isinstance(self.metric_selector, CalibratedSelector)
                    and bool(self.metric_selector.calibration_available.all()),
                    refit_calibration_available=None
                    if not isinstance(self.metric_selector, CalibratedSelector)
                    else self.metric_selector.calibration_available.cpu().tolist(),
                )
            region = regions[decision.region_index or 0]
            started = time.perf_counter()
            outcome = self.actions.execute(state, region, decision.action, self.model)
            if self.intervention_config is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            if self.intervention_config is not None:
                action_ms = (time.perf_counter() - started) * 1000
                routing_elapsed = routing_ms
                budget_metadata.update(
                    actual_action_ms=action_ms,
                    actual_policy_ms=base_ms + routing_elapsed + action_ms,
                    budget_overrun_ms=max(
                        0.0,
                        base_ms
                        + routing_elapsed
                        + action_ms
                        - self.intervention_config.total_budget_ms,
                    ),
                )
            else:
                action_ms = (
                    0.0
                    if decision.action == Action.STOP
                    else (time.perf_counter() - started) * 1000
                )
                budget_metadata.update(
                    actual_action_ms=action_ms,
                    actual_policy_ms=base_ms + routing_ms + action_ms,
                    budget_overrun_ms=max(
                        0.0,
                        base_ms + routing_ms + action_ms - self.routing.budget_ms,
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
