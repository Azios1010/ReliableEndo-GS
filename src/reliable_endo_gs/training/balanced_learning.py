"""Joint local actions, global preservation and train-only balanced routing."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from reliable_endo_gs.actions.repair import Action, action_eligible, eligible
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.regions.grid import grid_regions
from reliable_endo_gs.regions.masks import MaskRegion
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.teachers import AlignedTeacherTarget
from reliable_endo_gs.training.balanced_interventions import BalancedHarness, BalancedIntervention
from reliable_endo_gs.training.distillation import stereo_distillation
from reliable_endo_gs.training.interventions import intervention_costs, reconstruction_loss
from reliable_endo_gs.training.metric_objective import metric_depth_objective
from reliable_endo_gs.training.metric_quality import OfflineMetricTruth
from reliable_endo_gs.training.quality_components import measure_quality
from reliable_endo_gs.training.repairability import stereo_consistency_loss
from reliable_endo_gs.training.selector_optimization import selector_objective

if TYPE_CHECKING:
    from reliable_endo_gs.training.student_learning import StudentLearner


def _risk_components(
    tables: list[tuple[BalancedIntervention, ...]], features: torch.Tensor
) -> torch.Tensor:
    """[R,3,2] local/global labels; unknown placeholders are always excluded."""
    for table in tables:
        for row in table:
            if row.metric_observed and (row.risk_local is None or row.risk_global is None):
                raise ValueError("observed grouped risk requires both canonical event labels")
    return features.new_tensor(
        [
            [[float(row.risk_local or False), float(row.risk_global or False)] for row in table]
            for table in tables
        ]
    )


def balanced_step(
    learner: StudentLearner,
    sample: ScaredA00Sample,
    *,
    joint: bool,
    teacher_target: AlignedTeacherTarget | None,
    position: int,
    metric_truth: OfflineMetricTruth | None = None,
    selector_context: tuple[tuple[ScaredA00Sample, OfflineMetricTruth], ...] = (),
) -> dict[str, object]:
    cfg = learner.intervention_config
    if (
        cfg is None
        or not isinstance(learner.selector, BalancedSelector)
        or not isinstance(learner.harness, BalancedHarness)
        or learner.objective_balancer is None
        or learner.selector_balancer is None
    ):
        raise ValueError("balanced joint construction mismatch")
    if (
        sample.supervision is not None
        or sample.data_role != "final_train"
        or sample.sample_id.split("/")[0] not in {f"dataset_{i}" for i in range(1, 8)}
    ):
        raise ValueError("balanced learning excludes GT/final_test inputs")
    if learner.config.kd_weight and teacher_target is None:
        raise ValueError("KD requires explicit aligned target")
    if selector_context and (not joint or cfg.metric_quality is None):
        raise ValueError("fresh selector context requires physical joint learning")
    if len(selector_context) >= cfg.selector_context_frames:
        raise ValueError("too many fresh selector context frames")
    seen_context = {sample.sample_id}
    for context_sample, context_truth in selector_context:
        if context_sample.supervision is not None or context_sample.sample_id in seen_context:
            raise ValueError(
                "fresh selector context must be distinct deployment-only TRAIN samples"
            )
        context_truth.validate(context_sample.sample_id, context_sample.batch.support.shape)
        if context_sample.data_role != "final_train":
            raise ValueError("fresh selector context cannot consume final_test")
        seen_context.add(context_sample.sample_id)
    if cfg.metric_quality is not None and joint:
        if metric_truth is None:
            raise ValueError("metric joint learning requires separate offline TRAIN truth")
        metric_truth.validate(sample.sample_id, sample.batch.left[:, :1].shape)
    # The frozen ablation retains virtual model derivatives for the SAME
    # balancing probes as joint. Its optimizer is excluded and gradients discarded.
    learner.model.train(not learner.student_frozen).requires_grad_(True)
    learner.actions.train().requires_grad_(True)
    learner.selector.train()
    for opt in (learner.model_optimizer, learner.action_optimizer, learner.selector_optimizer):
        opt.zero_grad(set_to_none=True)
    state = learner.state(sample)
    if metric_truth is not None:
        metric_truth = replace(
            metric_truth,
            disparity=metric_truth.disparity.to(state.batch.left.device),
            valid_mask=metric_truth.valid_mask.to(state.batch.left.device),
        )
    h, w = state.batch.left.shape[-2:]
    full = grid_regions((h, w), (h, w), learner.routing.halo)[0]
    base = learner.actions.execute(state, full, Action.STOP, learner.model)
    rgb = reconstruction_loss(state, base)
    warp = stereo_consistency_loss(state, base.disparity)
    kd = rgb.new_zeros(())
    diag: dict[str, object] = {
        "student_update_mode": cfg.student_update_mode,
        "model_update_enabled": not learner.student_frozen,
        "frozen_student_start_step": learner.frozen_student_start_step,
    }
    if metric_truth is not None:
        digest = hashlib.sha256()
        for tensor in (metric_truth.disparity, metric_truth.valid_mask):
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
        diag.update(
            offline_metric_truth_sample_id=metric_truth.sample_id,
            offline_metric_truth_sha256=digest.hexdigest(),
            offline_metric_truth_resampling_identity=metric_truth.resampling_identity,
        )
    if learner.config.kd_weight:
        assert teacher_target is not None
        learner.model._validate_target_metadata(teacher_target, calibration_metadata=None)
        distillation = stereo_distillation(base.disparity, teacher_target, state.batch, cfg)
        kd = distillation.loss
        diag.update(distillation.diagnostics)
    repair = rgb.new_zeros(())
    geometry = rgb.new_zeros(())
    geometry_repairs: list[torch.Tensor] = []
    if cfg.geometry_supervision_weight:
        if metric_truth is None:
            raise ValueError("metric geometry learning requires separate offline TRAIN truth")
        depth_objective = metric_depth_objective(
            base.disparity,
            state.batch,
            metric_truth,
            sample_id=sample.sample_id,
            scale_mm=cfg.geometry_loss_scale_mm,
        )
        geometry = depth_objective.loss
        diag.update(
            geometry_base_loss=float(geometry.detach()),
            geometry_base_mae_mm=depth_objective.mae_mm,
            geometry_base_pixels=depth_objective.pixels,
            geometry_supervision_protocol="fixed_train_gt_depth_smooth_l1.v1",
        )
    selector_loss = rgb.new_zeros(())
    selector_pending = None
    # In burn-in mode the selector sees only detached labels while TRAIN
    # normalization is collected.  Student and action owners continue to
    # optimize; selector parameters begin updating only on a later snapshot
    # with the frozen transform.
    selector_training_ready = cfg.normalization_mode == "legacy" or bool(
        learner.selector.normalization.fitted
    )
    regions = (
        tuple(r for r in learner.candidate_regions(state, sample.sample_id) if eligible(state, r))
        if joint
        else ()
    )
    if joint and not regions and learner.mask_config is not None:
        # Empty/quiet target: update student/head only; never manufacture labels
        # or make background repairs just to give action parameters gradients.
        joint = False
        diag["mask_joint_skipped"] = "no_eligible_target_candidates"
    if joint:
        if not regions:
            raise ValueError("no eligible local intervention regions")
        features = learner.selection_features(state, regions)
        learner.actions.requires_grad_(False)
        try:
            tables = [
                learner.harness.collect(
                    state, r, learner.actions, learner.model, metric_truth=metric_truth
                )
                for r in regions
            ]
        finally:
            learner.actions.requires_grad_(True)
        global_targets = features.new_tensor([[row.utility for row in table] for table in tables])
        local_targets = features.new_tensor([[row.local_gain for row in table] for table in tables])
        # Counterfactuals are detached supervision.  STOP is always an exact
        # zero-risk/zero-gain reference; only repair actions carry risk labels.
        harms = features.new_tensor([[float(row.harmful) for row in table] for table in tables])
        action_available = torch.tensor(
            [[row.action_available and row.metric_observed for row in table] for table in tables],
            device=features.device,
            dtype=torch.bool,
        )
        action_available[:, 0] = True
        label_available = action_available[:, 1:].any(-1)
        # Only labels/features are batched. Every historical sample is run again
        # with this exact current actor snapshot; no stale outcome replay.
        fit_features = [features]
        fit_global = [global_targets]
        fit_local = [local_targets]
        fit_harms = [harms]
        fit_risk = (
            [_risk_components(tables, features)] if cfg.selector_risk_mode == "local_global" else []
        )
        fit_available = [action_available]
        context_diagnostics = []
        if selector_context:
            from reliable_endo_gs.training.student_learning import evaluation_mode

            action_flags = [(p, p.requires_grad) for p in learner.actions.parameters()]
            learner.actions.requires_grad_(False)
            try:
                with torch.no_grad(), evaluation_mode(learner.model, learner.actions):
                    for context_sample, context_truth in selector_context:
                        context_state = learner.state(context_sample)
                        context_regions = tuple(
                            r
                            for r in learner.candidate_regions(
                                context_state, context_sample.sample_id
                            )
                            if eligible(context_state, r)
                        )
                        if not context_regions:
                            continue
                        context_truth = replace(
                            context_truth,
                            disparity=context_truth.disparity.to(context_state.batch.left.device),
                            valid_mask=context_truth.valid_mask.to(context_state.batch.left.device),
                        )
                        context_features = learner.selection_features(
                            context_state, context_regions
                        )
                        context_tables = [
                            learner.harness.collect(
                                context_state,
                                r,
                                learner.actions,
                                learner.model,
                                metric_truth=context_truth,
                            )
                            for r in context_regions
                        ]
                        context_global = context_features.new_tensor(
                            [[row.utility for row in t] for t in context_tables]
                        )
                        context_local = features.new_tensor(
                            [[row.local_gain for row in t] for t in context_tables]
                        )
                        context_harms = context_features.new_tensor(
                            [[float(row.harmful) for row in t] for t in context_tables]
                        )
                        context_action_available = torch.tensor(
                            [
                                [row.action_available and row.metric_observed for row in t]
                                for t in context_tables
                            ],
                            device=features.device,
                            dtype=torch.bool,
                        )
                        context_action_available[:, 0] = True
                        context_available = context_action_available[:, 1:].any(-1)
                        fit_features.append(context_features)
                        fit_global.append(context_global)
                        fit_local.append(context_local)
                        fit_harms.append(context_harms)
                        if cfg.selector_risk_mode == "local_global":
                            fit_risk.append(_risk_components(context_tables, context_features))
                        fit_available.append(context_action_available)
                        context_diagnostics.append(
                            {
                                "sample_id": context_sample.sample_id,
                                "snapshot_optimizer_step": position,
                                "features_raw": context_features.cpu().tolist(),
                                "global_targets": context_global.cpu().tolist(),
                                "local_targets": context_local.cpu().tolist(),
                                "harms": context_harms.cpu().tolist(),
                                "risk_events": [
                                    [
                                        [
                                            row.risk_local,
                                            row.risk_global,
                                            row.harmful if row.metric_observed else None,
                                        ]
                                        for row in table
                                    ]
                                    for table in context_tables
                                ]
                                if cfg.selector_risk_mode == "local_global"
                                else None,
                                "label_available": context_available.cpu().tolist(),
                                "selector_label_available": context_action_available.cpu().tolist(),
                                "action_available": [
                                    [row.action_available for row in table]
                                    for table in context_tables
                                ],
                            }
                        )
            finally:
                for parameter, flag in action_flags:
                    parameter.requires_grad_(flag)
        snapshot_features = torch.cat(fit_features).detach()
        snapshot_global = torch.cat(fit_global).detach()
        snapshot_local = torch.cat(fit_local).detach()
        snapshot_harms = torch.cat(fit_harms).detach()
        snapshot_risk = torch.cat(fit_risk).detach() if fit_risk else None
        snapshot_action_available = torch.cat(fit_available)
        snapshot_available = snapshot_action_available[:, 1:].any(-1)
        selector_training_ready = selector_training_ready and bool(snapshot_available.any())
        diag["selector_snapshot_context"] = context_diagnostics
        diag["selector_snapshot_known_regions"] = int(snapshot_available.sum())
        diag["selector_snapshot_known_actions"] = int(snapshot_action_available[:, 1:].sum())
        diag["selector_snapshot_frames"] = 1 + len(context_diagnostics)
        diag["selector_snapshot_protocol"] = "fresh_current_actor_multi_frame.v1"
        diag["selector_observed_regions"] = int(label_available.sum())
        diag["selector_unknown_metric_regions"] = int((~label_available).sum())
        if not bool((global_targets[:, 0] == 0).all() & (local_targets[:, 0] == 0).all()):
            raise ValueError("balanced intervention labels must keep STOP gains at zero")
        if not bool((harms[:, 0] == 0).all()):
            raise ValueError("balanced intervention labels must keep STOP harm at zero")
        if selector_training_ready:
            selector_loss, selector_pending, selector_diag = selector_objective(
                learner.selector,
                learner.selector_balancer,
                snapshot_features[snapshot_available],
                snapshot_global[snapshot_available],
                snapshot_local[snapshot_available],
                snapshot_harms[snapshot_available],
                risk_components=None
                if snapshot_risk is None
                else snapshot_risk[snapshot_available],
                action_available=snapshot_action_available[snapshot_available],
                train_harm=not cfg.selector_gain_only_refinement
                or cfg.selector_updates_per_snapshot == 1,
            )
            diag.update(selector_diag)
        else:
            diag.update(
                selector_burn_in=cfg.normalization_mode == "burn_in"
                and not bool(learner.selector.normalization.fitted),
                selector_normalization_fitted=bool(learner.selector.normalization.fitted),
                selector_supervision_collected=bool(snapshot_available.any()),
                selector_skip_reason="normalization_burn_in"
                if bool(snapshot_available.any())
                else "no_observed_metric_outcomes",
            )
        diag["selector_supervision_semantics"] = (
            "physical_metric_decision_local_geometry_auxiliary"
            if cfg.metric_quality is not None
            else "global_decision_local_auxiliary"
        )
        repair_terms = []
        geometry_stereo_global_pixels = 0
        geometry_stereo_local_pixels = 0
        for region, table in zip(regions, tables, strict=True):
            before = table[0].components_before
            for action in (Action.STEREO, Action.GAUSSIAN):
                if not action_eligible(state, region, action):
                    continue
                outcome = learner.actions.execute(state, region, action, learner.model)
                if cfg.geometry_supervision_weight and action == Action.STEREO:
                    assert metric_truth is not None
                    global_depth = metric_depth_objective(
                        outcome.disparity,
                        state.batch,
                        metric_truth,
                        sample_id=sample.sample_id,
                        scale_mm=cfg.geometry_loss_scale_mm,
                    )
                    local_depth = metric_depth_objective(
                        outcome.disparity,
                        state.batch,
                        metric_truth,
                        sample_id=sample.sample_id,
                        scale_mm=cfg.geometry_loss_scale_mm,
                        region_mask=region.mask(state.batch.left),
                    )
                    if global_depth.pixels or local_depth.pixels:
                        # Keep both full-frame preservation and local repair
                        # support in the action objective.  Missing support is
                        # an explicit zero contribution, never a new label.
                        geometry_repairs.append(
                            cfg.stereo_repair_geometry_weight
                            * (global_depth.loss + local_depth.loss)
                            / (2 if global_depth.pixels and local_depth.pixels else 1)
                        )
                    geometry_stereo_global_pixels += global_depth.pixels
                    geometry_stereo_local_pixels += local_depth.pixels
                quality_components = measure_quality(state, outcome, region)
                local = (
                    quality_components.local_rgb / cfg.rgb_scale
                    + cfg.stereo_weight * quality_components.local_warp / cfg.stereo_scale
                )
                global_loss = (
                    quality_components.global_rgb / cfg.rgb_scale
                    + cfg.stereo_weight * quality_components.global_warp / cfg.stereo_scale
                )
                outside = (quality_components.outside_rgb - before["outside_rgb"]).clamp_min(
                    0
                ) / cfg.rgb_scale + cfg.stereo_weight * (
                    quality_components.outside_warp - before["outside_warp"]
                ).clamp_min(0) / cfg.stereo_scale
                repair_terms.append(
                    cfg.local_repair_weight * local
                    + cfg.global_repair_weight * global_loss
                    + cfg.collateral_weight * outside
                )
        repair = torch.stack(repair_terms).mean() if repair_terms else rgb * 0
        diag.update(
            geometry_stereo_global_pixels=geometry_stereo_global_pixels,
            geometry_stereo_local_pixels=geometry_stereo_local_pixels,
        )
        timing_rows = [intervention_costs(table, features) for table in tables]
        cost_rows = torch.stack([row[0] for row in timing_rows])
        execution_available = torch.stack([row[1] for row in timing_rows])
        diag["repair_executed_actions"] = int(execution_available[:, 1:].sum())
        diag["action_unavailable_count"] = int((~execution_available[:, 1:]).sum())
        # Persist component labels from the same before-update snapshot, not only scalar utility.
        diag["interventions"] = [
            {
                "region_id": r.index,
                "core_yxyx": list(r.core),
                "write_pixel_count": r.pixel_count,
                "write_flat_indices": list(r.flat_indices) if isinstance(r, MaskRegion) else None,
                "region_protocol": "fixed_grid.v1"
                if learner.mask_config is None
                else learner.mask_config.schema,
                "snapshot_optimizer_step": position,
                "left_support_pixels": int((state.support & r.mask(state.batch.left)).sum()),
                "actions": [
                    {
                        "action": row.action.name,
                        "cost_ms": row.cost_ms,
                        "action_available": row.action_available,
                        "action_executed": row.action_available,
                        "action_unavailable_reason": row.unavailable_reason,
                        "global_gain": row.utility if row.metric_observed else None,
                        "local_gain": row.local_gain if row.metric_observed else None,
                        "harm": row.harmful if row.metric_observed else None,
                        "risk_local": row.risk_local,
                        "risk_global": row.risk_global,
                        "before": row.components_before,
                        "after": row.components_after,
                        "metric_changes": row.metric_changes,
                        "selector_label_available": row.metric_observed,
                        "metric_diagnostics": row.metric_diagnostics,
                    }
                    for row in table
                ],
            }
            for r, table in zip(regions, tables, strict=True)
        ]
        diag["selector_regions"] = len(regions)
        diag["features_raw"] = features.detach().cpu().tolist()
        if selector_training_ready:
            (
                global_predictions,
                local_predictions,
                predicted_harms,
                predicted_events,
                gain_components,
            ) = learner.selector.outputs_with_components(features)
            diag.update(
                predicted_global_gains=global_predictions.detach().cpu().tolist(),
                predicted_local_gains=local_predictions.detach().cpu().tolist(),
                predicted_harms=predicted_harms.detach().cpu().tolist(),
            )
            if cfg.selector_risk_mode == "local_global":
                assert predicted_events is not None
                diag["predicted_risk_events"] = predicted_events.detach().cpu().tolist()
            if gain_components is not None:
                diag["predicted_gain_components"] = gain_components.to_report()
    terms: tuple[torch.Tensor, ...] = (
        rgb / cfg.rgb_scale,
        cfg.stereo_weight * warp / cfg.stereo_scale,
        kd,
        repair,
    )
    if cfg.geometry_supervision_weight:
        local_geometry = torch.stack(geometry_repairs).mean() if geometry_repairs else geometry * 0
        geometry = geometry + local_geometry
        terms += (geometry,)
        diag.update(
            geometry_stereo_local_loss=float(local_geometry.detach()),
            geometry_stereo_action_loss=float(local_geometry.detach()),
            geometry_stereo_observed_regions=len(geometry_repairs),
        )
    pending = None
    initialize_geometry_balance = (
        bool(cfg.geometry_supervision_weight)
        and bool(geometry.detach() > 0)
        and float(learner.objective_balancer.ema[-1]) == 0
    )
    if position % cfg.gradient_diagnostics_every == 0 or initialize_geometry_balance:
        pending, balance_diag = learner.objective_balancer.measure(
            terms, tuple(learner.model.stereo_student.features.parameters())
        )
        diag.update({f"model_{key}": value for key, value in balance_diag.items()})
    weights = learner.objective_balancer.weights if pending is None else pending[1]
    priorities = rgb.new_tensor(
        [
            learner.config.reconstruction_weight,
            learner.config.reconstruction_weight * cfg.stereo_training_priority,
            learner.config.kd_weight,
            learner.config.repair_weight if joint else 0.0,
        ]
        + ([cfg.geometry_supervision_weight] if cfg.geometry_supervision_weight else [])
    )
    loss = learner.objective_balancer.weighted_loss(terms, priorities, pending)
    diag.update(
        stereo_training_priority=cfg.stereo_training_priority,
        model_effective_weight_warp=float(weights[1] * priorities[1] * cfg.stereo_weight),
    )
    if cfg.geometry_supervision_weight:
        diag["model_effective_weight_geometry"] = float(weights[4] * priorities[4])
        diag["geometry"] = float(geometry.detach())
    if not bool(torch.isfinite(loss) & torch.isfinite(selector_loss)):
        raise ValueError("nonfinite balanced objective")
    loss.backward()
    if joint and selector_training_ready:
        # Fast head updates happen BEFORE the student/actions change. Reuse
        # only this sample's detached fresh counterfactuals, never old outcomes.
        loss_before_refinement = float(selector_loss.detach())
        if cfg.selector_gain_only_refinement and cfg.selector_updates_per_snapshot > 1:
            # Report the full objective at both endpoints even though extra
            # gain steps exclude BCE. Risk is trained once on the final step.
            harm_loss = diag["selector_harm_loss"]
            assert isinstance(harm_loss, float)
            loss_before_refinement += cfg.harm_weight * harm_loss
        diag["selector_loss_before_refinement"] = loss_before_refinement
        for refinement in range(cfg.selector_updates_per_snapshot - 1):
            assert selector_pending is not None
            selector_loss.backward()
            learner.commit_gradients(True, selector_only=True)
            learner.selector_balancer.commit(selector_pending)
            learner.selector_optimizer.zero_grad(set_to_none=True)
            selector_loss, selector_pending, selector_diag = selector_objective(
                learner.selector,
                learner.selector_balancer,
                snapshot_features[snapshot_available],
                snapshot_global[snapshot_available],
                snapshot_local[snapshot_available],
                snapshot_harms[snapshot_available],
                risk_components=None
                if snapshot_risk is None
                else snapshot_risk[snapshot_available],
                action_available=snapshot_action_available[snapshot_available],
                train_harm=not cfg.selector_gain_only_refinement
                or refinement == cfg.selector_updates_per_snapshot - 2,
            )
            diag.update(selector_diag)
        selector_loss.backward()
    diag["selector_updates_this_snapshot"] = (
        cfg.selector_updates_per_snapshot if joint and selector_training_ready else 0
    )
    diag["selector_harm_updates_this_snapshot"] = (
        (1 if cfg.selector_gain_only_refinement else cfg.selector_updates_per_snapshot)
        if joint and selector_training_ready
        else 0
    )
    diag.update(
        learner.commit_gradients(
            joint,
            update_selector=selector_training_ready,
            active_repairs=tuple(
                action
                for action in (Action.STEREO, Action.GAUSSIAN)
                if joint and bool(execution_available[:, int(action)].any())
            ),
        )
    )
    if pending is not None:
        learner.objective_balancer.commit(pending)
    if joint:
        # Statistics are collected after the model/action optimizer step.  In
        # burn-in mode this is the only selector state mutation; it enables
        # the next step without backpropagating through the calibration pass.
        if bool(label_available.any()):
            learner.selector.normalization.observe(
                features[label_available],
                global_targets[label_available],
                local_targets[label_available],
                sample_id=sample.sample_id,
                role=sample.data_role,
                action_available=action_available[label_available],
            )
        # Metric availability is independent of measured execution time.  An
        # unknown GT outcome still has a valid timing sample when the action
        # executed successfully. Skipped actions have no timing observation.
        learner.record_action_costs(cost_rows, execution_available)
    if joint and selector_training_ready:
        assert selector_pending is not None
        learner.selector_balancer.commit(selector_pending)
    return {
        "loss": float(loss.detach()),
        "reconstruction": float(rgb.detach()),
        "stereo_consistency": float(warp.detach()),
        "kd": float(kd.detach()),
        "repair": float(repair.detach()),
        "selector": float(selector_loss.detach()),
        **diag,
    }
