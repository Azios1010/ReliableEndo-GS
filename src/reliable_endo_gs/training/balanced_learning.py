"""Joint local actions, global preservation and train-only balanced routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from reliable_endo_gs.actions.repair import Action, eligible
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.regions.grid import grid_regions
from reliable_endo_gs.regions.masks import MaskRegion
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.teachers import AlignedTeacherTarget
from reliable_endo_gs.teachers.quality import screen_target
from reliable_endo_gs.training.balanced_interventions import BalancedHarness
from reliable_endo_gs.training.interventions import reconstruction_loss
from reliable_endo_gs.training.quality_components import measure_quality
from reliable_endo_gs.training.repairability import stereo_consistency_loss

if TYPE_CHECKING:
    from reliable_endo_gs.training.student_learning import StudentLearner


def balanced_step(
    learner: StudentLearner,
    sample: ScaredA00Sample,
    *,
    joint: bool,
    teacher_target: AlignedTeacherTarget | None,
    position: int,
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
    learner.model.train().requires_grad_(True)
    learner.actions.train().requires_grad_(True)
    learner.selector.train()
    for opt in (learner.model_optimizer, learner.action_optimizer, learner.selector_optimizer):
        opt.zero_grad(set_to_none=True)
    state = learner.state(sample)
    h, w = state.batch.left.shape[-2:]
    full = grid_regions((h, w), (h, w), learner.routing.halo)[0]
    base = learner.actions.execute(state, full, Action.STOP, learner.model)
    rgb = reconstruction_loss(state, base)
    warp = stereo_consistency_loss(state, base.disparity)
    kd = rgb.new_zeros(())
    diag: dict[str, object] = {}
    if learner.config.kd_weight:
        assert teacher_target is not None
        learner.model._validate_target_metadata(teacher_target, calibration_metadata=None)
        quality = screen_target(state.batch, teacher_target, cfg)
        target = teacher_target.disparity.detach().to(state.batch.left.device)
        if quality.accepted_count:
            kd = (
                F.smooth_l1_loss(base.disparity[quality.mask], target[quality.mask])
                / cfg.kd_scale_px
            )
        diag.update(
            kd_declared_pixels=float(quality.declared_count),
            kd_accepted_pixels=float(quality.accepted_count),
            kd_below_near_pixels=float(quality.below_near_count),
            kd_outside_right_pixels=float(quality.outside_right_count),
            kd_empty=float(quality.accepted_count == 0),
        )
    repair = rgb.new_zeros(())
    selector_loss = rgb.new_zeros(())
    selector_pending = None
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
                learner.harness.collect(state, r, learner.actions, learner.model) for r in regions
            ]
        finally:
            learner.actions.requires_grad_(True)
        global_targets = features.new_tensor([[row.utility for row in table] for table in tables])
        local_targets = features.new_tensor([[row.local_gain for row in table] for table in tables])
        harms = features.new_tensor([[float(row.harmful) for row in table] for table in tables])
        component_losses, selector_diag = learner.selector.objective(
            features, global_targets, local_targets, harms
        )
        shared = tuple(
            p
            for layer in list(learner.selector.gain_network.children())[:-1]
            for p in layer.parameters()
        )
        selector_pending, balance_diag = learner.selector_balancer.measure(
            tuple(component_losses[k] for k in ("global", "local", "ranking")), shared
        )
        selector_loss = (
            sum(
                selector_pending[1, i] * weight * component_losses[k]
                for i, (k, weight) in enumerate(
                    (
                        ("global", 1.0),
                        ("local", cfg.local_auxiliary_weight),
                        ("ranking", cfg.ranking_weight),
                    )
                )
            )
            + cfg.harm_weight * component_losses["harm"]
        )
        selector_diag.update({f"selector_{k}": v for k, v in balance_diag.items()})
        diag.update(selector_diag)
        repair_terms = []
        for region, table in zip(regions, tables, strict=True):
            before = table[0].components_before
            for action in (Action.STEREO, Action.GAUSSIAN):
                outcome = learner.actions.execute(state, region, action, learner.model)
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
        repair = torch.stack(repair_terms).mean()
        costs = features.new_tensor([[row.cost_ms for row in table] for table in tables]).amax(0)
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
                        "global_gain": row.utility,
                        "local_gain": row.local_gain,
                        "harm": row.harmful,
                        "before": row.components_before,
                        "after": row.components_after,
                    }
                    for row in table
                ],
            }
            for r, table in zip(regions, tables, strict=True)
        ]
        diag["selector_regions"] = len(regions)
        diag["features_raw"] = features.detach().cpu().tolist()
        global_predictions, local_predictions, predicted_harms = learner.selector.outputs(features)
        diag.update(
            predicted_global_gains=global_predictions.detach().cpu().tolist(),
            predicted_local_gains=local_predictions.detach().cpu().tolist(),
            predicted_harms=predicted_harms.detach().cpu().tolist(),
        )
    terms = (rgb / cfg.rgb_scale, cfg.stereo_weight * warp / cfg.stereo_scale, kd, repair)
    pending = None
    if position % cfg.gradient_diagnostics_every == 0:
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
    )
    loss = learner.objective_balancer.weighted_loss(terms, priorities, pending)
    diag.update(
        stereo_training_priority=cfg.stereo_training_priority,
        model_effective_weight_warp=float(weights[1] * priorities[1] * cfg.stereo_weight),
    )
    if not bool(torch.isfinite(loss) & torch.isfinite(selector_loss)):
        raise ValueError("nonfinite balanced objective")
    loss.backward()
    if joint:
        selector_loss.backward()
    diag.update(learner.commit_gradients(joint))
    if pending is not None:
        learner.objective_balancer.commit(pending)
    if joint:
        assert selector_pending is not None
        learner.selector_balancer.commit(selector_pending)
        learner.selector.normalization.observe(
            features,
            global_targets,
            local_targets,
            sample_id=sample.sample_id,
            role=sample.data_role,
        )
        learner.cost_max_ms = torch.maximum(learner.cost_max_ms, costs)
        learner.cost_observations += 1
    return {
        "loss": float(loss.detach()),
        "reconstruction": float(rgb.detach()),
        "stereo_consistency": float(warp.detach()),
        "kd": float(kd.detach()),
        "repair": float(repair.detach()),
        "selector": float(selector_loss.detach()),
        **diag,
    }
