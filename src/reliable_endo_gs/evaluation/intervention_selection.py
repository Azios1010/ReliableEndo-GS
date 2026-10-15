"""Native read-only gain/harm selection benchmark on disjoint development IDs."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.actions.repair import (
    Action,
    ActionResult,
    RepairState,
    action_eligible,
    eligible,
)
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, load_a01_manifest
from reliable_endo_gs.data.target_masks import TargetMaskIndex
from reliable_endo_gs.evaluation.benchmark_geometry import evaluate_geometry
from reliable_endo_gs.evaluation.gain_events import gain_event_metrics
from reliable_endo_gs.evaluation.rendering import psnr, windowed_ssim
from reliable_endo_gs.evaluation.risk_events import risk_event_metrics
from reliable_endo_gs.reconstruction import ReferenceConfig
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.regions.grid import Region, grid_regions
from reliable_endo_gs.regions.masks import MaskRegion
from reliable_endo_gs.rendering.production import ProductionFieldRenderer
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.calibrated_selector import CalibratedSelector
from reliable_endo_gs.routing.config import RoutingConfig
from reliable_endo_gs.routing.cost import load_cost_calibration
from reliable_endo_gs.routing.gain import SignedGainPrediction
from reliable_endo_gs.routing.masks import MaskProposalConfig
from reliable_endo_gs.routing.metric_selection import module_state_sha256
from reliable_endo_gs.routing.repairability import (
    RepairabilitySelector,
    select_repairability,
)
from reliable_endo_gs.routing.selector import Decision
from reliable_endo_gs.routing.tokens import selector_feature_names
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.runtime.identity import source_sha256
from reliable_endo_gs.teachers import load_a01_roles
from reliable_endo_gs.training.harness import capture_rng, restore_rng
from reliable_endo_gs.training.interventions import reconstruction_loss
from reliable_endo_gs.training.metric_quality import metric_action_policy, metric_targets
from reliable_endo_gs.training.quality_components import component_targets, measure_quality
from reliable_endo_gs.training.repairability import (
    outside_loss,
    quality_gain,
    stereo_consistency_loss,
)
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    read_student_checkpoint,
)
from reliable_endo_gs.training.student_config import StudentLearningConfig
from reliable_endo_gs.training.student_learning import StudentLearner, evaluation_mode
from reliable_endo_gs.utils.io import write_json_atomic


def _selection_outputs(
    predictor: BalancedSelector | CalibratedSelector | RepairabilitySelector,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, SignedGainPrediction | None]:
    """Keep complete diagnostics in the same prediction space as the decision."""
    if isinstance(predictor, (BalancedSelector, CalibratedSelector)):
        gains, _, harms, events, components = predictor.outputs_with_components(features)
        return gains, harms, events, components
    gains, harms = predictor.predict(features)
    return gains, harms, None, None


def _action_availability(
    state: RepairState, regions: tuple[Region | MaskRegion, ...]
) -> torch.Tensor:
    """Deployment preconditions [R,3], independent of offline GT availability."""
    return torch.tensor(
        [[action_eligible(state, region, action) for action in Action] for region in regions],
        device=state.batch.left.device,
        dtype=torch.bool,
    ).reshape(-1, 3)


def _weights(learner: StudentLearner) -> str:
    digest = hashlib.sha256()
    for module in (learner.model, learner.actions, learner.selector):
        for name, value in sorted(module.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate_intervention_selection(
    *,
    checkpoint: Path,
    data_root: Path,
    manifest_path: Path,
    training_manifest: Path,
    roles_path: Path,
    output_dir: Path,
    max_samples: int = 112,
    device_name: str = "cuda:0",
    target_mask_index: Path | None = None,
    sample_ids: tuple[str, ...] | None = None,
    metric_config_path: Path | None = None,
    base_only: bool = False,
    cost_calibration_path: Path | None = None,
    expected_checkpoint_sha256: str | None = None,
    selector_refit_path: Path | None = None,
) -> dict[str, object]:
    """Read-only deployment evaluation; restore RNG even on evaluation failure."""
    if not torch.cuda.is_available():
        raise ValueError("native selection evaluation requires CUDA")
    device = torch.device(device_name)
    rng = capture_rng(device)
    try:
        return _evaluate_intervention_selection(
            checkpoint=checkpoint,
            data_root=data_root,
            manifest_path=manifest_path,
            training_manifest=training_manifest,
            roles_path=roles_path,
            output_dir=output_dir,
            max_samples=max_samples,
            device_name=device_name,
            target_mask_index=target_mask_index,
            sample_ids=sample_ids,
            metric_config_path=metric_config_path,
            base_only=base_only,
            cost_calibration_path=cost_calibration_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            selector_refit_path=selector_refit_path,
        )
    finally:
        restore_rng(rng, device)


def _evaluate_intervention_selection(
    *,
    checkpoint: Path,
    data_root: Path,
    manifest_path: Path,
    training_manifest: Path,
    roles_path: Path,
    output_dir: Path,
    max_samples: int,
    device_name: str,
    target_mask_index: Path | None,
    sample_ids: tuple[str, ...] | None,
    metric_config_path: Path | None,
    base_only: bool,
    cost_calibration_path: Path | None = None,
    expected_checkpoint_sha256: str | None = None,
    selector_refit_path: Path | None = None,
) -> dict[str, object]:
    """Evaluate real shared actions, fixed-mask gain and independent GT metrics.

    One intervention per policy. Fixed actions use a cyclic eligible region;
    predicted-region baselines separate region selection from action switching.
    Oracle diagnostic overhead is excluded from estimated policy timing.
    """
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    if not torch.cuda.is_available():
        raise ValueError("native selection evaluation requires CUDA")
    repo = Path(__file__).resolve().parents[3]
    if repo == output_dir.resolve() or repo in output_dir.resolve().parents:
        raise ValueError("evaluation artifacts must be outside repository")
    roles = load_a01_roles(roles_path)
    train = load_a01_manifest(training_manifest, roles=roles)
    manifest = load_a01_manifest(manifest_path, roles=roles)
    train_ids = {entry.sample_id for entry in train.entries}
    ids = [entry.sample_id for entry in manifest.entries if entry.data_role == "final_train"][
        :max_samples
    ]
    if sample_ids is not None:
        known = {entry.sample_id for entry in manifest.entries if entry.data_role == "final_train"}
        if (
            not sample_ids
            or len(set(sample_ids)) != len(sample_ids)
            or not set(sample_ids) <= known
            or len(sample_ids) > max_samples
        ):
            raise ValueError(
                "evaluation worklist must be unique bounded IDs in the development manifest"
            )
        ids = list(sample_ids)
    if not ids or train_ids.intersection(ids):
        raise ValueError("evaluation IDs must be disjoint from training")
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if expected_checkpoint_sha256 is not None and checkpoint_digest != expected_checkpoint_sha256:
        raise ValueError("pinned evaluation checkpoint SHA-256 mismatch")
    raw = read_student_checkpoint(checkpoint)
    if raw.get("intervention_config") is None:
        raise ValueError("requires a gain/harm intervention checkpoint")
    cfg = InterventionConfig(**raw["intervention_config"])
    if (
        (
            raw["identity"]["source_sha256"] != source_sha256(repo)
            and expected_checkpoint_sha256 is None
        )
        or raw["identity"]["manifest_hash"] != train.manifest_hash
        or raw["identity"]["role_hash"] != roles.role_hash
    ):
        raise ValueError("checkpoint source/training/role identity mismatch")
    routing = RoutingConfig(**raw["routing_config"])
    config = StudentLearningConfig(**raw["student_config"])
    construction = ReferenceConfig.from_dict(raw["construction_config"])
    renderer = ProductionFieldRenderer()
    if renderer.identity != raw["identity"]["renderer_identity"]:
        raise ValueError("renderer identity mismatch")
    device = torch.device(device_name)
    model = StudentReconstructionModel(
        renderer=renderer,
        config=construction,
        width=config.width,
        bins=config.bins,
        minimum_effective_px=config.minimum_effective_px,
    ).to(device)
    mask_config = (
        None if raw.get("mask_config") is None else MaskProposalConfig(**raw["mask_config"])
    )
    if (mask_config is None) != (target_mask_index is None):
        raise ValueError("mask checkpoint requires its explicit deployment ROI index")
    masks = None if target_mask_index is None else TargetMaskIndex(target_mask_index)
    if masks is not None:
        if construction.runtime.resolution_hw is None:
            raise ValueError("mask evaluation requires a pinned output grid")
        masks.require_samples(ids, construction.runtime.resolution_hw)
    learner = StudentLearner(
        model, config, routing, intervention_config=cfg, mask_config=mask_config, target_masks=masks
    )
    size = len(train_ids)
    if mask_config is not None and raw["identity"].get("run_kind") == "bounded_mask_validation":
        worklist = raw["identity"].get("training_worklist")
        if (
            not isinstance(worklist, list)
            or not worklist
            or len(set(worklist)) != len(worklist)
            or not set(worklist) <= train_ids
            or raw["identity"].get("samples_per_epoch") != len(worklist)
        ):
            raise ValueError("invalid pinned mask training worklist")
        size = len(worklist)
    # A fresh STOP-only baseline has no fitted routing normalizer. The strict
    # resume validator accepts it on this disposable clone; the public wrapper
    # restores process RNG and no action/selector prediction is executed.
    load_student_checkpoint(checkpoint, learner, raw["identity"], size, resume=base_only)
    if cost_calibration_path is not None and not base_only:
        artifact = load_cost_calibration(cost_calibration_path)
        if artifact.binding.get(
            "training_manifest_hash"
        ) != train.manifest_hash or not artifact.matches_training_identity(raw["identity"]):
            raise ValueError("deployment timing training identity mismatch")
        learner.attach_cost_calibration(artifact)
    if selector_refit_path is not None:
        if base_only:
            raise ValueError("STOP baseline cannot attach a refit selector")
        learner.load_refit_selector(selector_refit_path, source_sha256(repo))
    refit = learner.metric_selector
    if refit is not None and not isinstance(refit, CalibratedSelector):
        raise ValueError("intervention evaluator requires the calibrated selector")
    refit_before = None if refit is None else module_state_sha256(refit)
    if (
        cfg.metric_quality is not None
        and not base_only
        and learner.deployment_cost_calibration is None
    ):
        raise ValueError(
            "physical metric evaluation requires dedicated TRAIN inference cost calibration"
        )
    if (
        not learner.cost_observations
        and not base_only
        and learner.deployment_cost_calibration is None
    ):
        raise ValueError("evaluation needs measured joint-action costs")
    if (
        isinstance(learner.selector, BalancedSelector)
        and not bool(learner.selector.normalization.fitted)
        and not base_only
    ):
        raise ValueError("evaluation requires fitted training normalization")
    for module in (model, learner.actions, learner.selector):
        module.requires_grad_(False)
    assert isinstance(learner.selector, RepairabilitySelector)
    before = _weights(learner)
    metric_config = None
    if metric_config_path is not None:
        from reliable_endo_gs.routing.tradeoff import load_tradeoff_config

        metric_config = load_tradeoff_config(metric_config_path)
    if cfg.metric_quality is not None:
        configured_metric = metric_action_policy(cfg, int(Action.STEREO))
        if metric_config is not None and metric_config.config_hash != configured_metric.config_hash:
            raise ValueError("diagnostic metric policy differs from joint supervision")
        metric_config = configured_metric
    adapter = ScaredA00Adapter(data_root, manifest, roles=roles)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats(device)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with (
        (output_dir / "samples.jsonl").open("x", encoding="utf-8") as stream,
        evaluation_mode(model, learner.actions, learner.selector),
    ):
        warm = learner.state(
            adapter.load(
                ids[0],
                data_role="final_train",
                include_supervision=False,
                output_hw=construction.runtime.resolution_hw,
            )
        )
        warm_regions = learner.candidate_regions(warm, ids[0])
        if not base_only:
            for action in (Action.STEREO, Action.GAUSSIAN):
                warm_region = next(
                    (r for r in warm_regions if action_eligible(warm, r, action)), None
                )
                if warm_region is not None:
                    learner.actions.execute(warm, warm_region, action, model)
        del warm
        for index, sid in enumerate(ids):
            sample = adapter.load(
                sid,
                data_role="final_train",
                include_supervision=False,
                output_hw=construction.runtime.resolution_hw,
            )
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            state = learner.state(sample)
            torch.cuda.synchronize(device)
            base_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            regions = () if base_only else learner.candidate_regions(state, sid)
            valid = torch.tensor(
                [eligible(state, r) for r in regions], device=device, dtype=torch.bool
            )
            executable = _action_availability(state, regions)
            event_predictions = None
            gain_components = None
            if regions:
                features = learner.selection_features(state, regions)
                predictor = refit if refit is not None else learner.selector
                gains, harms, event_predictions, gain_components = _selection_outputs(
                    predictor, features
                )
            else:
                features = (
                    state.batch.left.new_empty((0, len(learner.selector.normalization.center)))
                    if isinstance(learner.selector, BalancedSelector)
                    else state.batch.left.new_empty((0, 26))
                )
                gains = state.batch.left.new_zeros((0, 3))
                harms = torch.zeros_like(gains)
            costs = (
                state.batch.left.new_zeros(3)
                if base_only
                else learner.estimated_action_costs(state)
            ).expand(len(regions), -1)
            torch.cuda.synchronize(device)
            routing_ms = (time.perf_counter() - t0) * 1000
            remaining = max(0.0, cfg.total_budget_ms - base_ms - routing_ms)
            decision = (
                refit.select(features, costs, executable, remaining)
                if refit is not None
                else select_repairability(gains, harms, costs, executable, remaining, cfg)
            )
            torch.cuda.synchronize(device)
            routing_ms = (time.perf_counter() - t0) * 1000
            remaining = max(0.0, cfg.total_budget_ms - base_ms - routing_ms)
            if decision.estimated_ms > remaining:
                decision = Decision(None, Action.STOP, 0.0, 0.0)
            # Decisions/features precede outcomes and separately loaded GT.
            eligible_ids = valid.nonzero().flatten().tolist()
            h, w = state.batch.left.shape[-2:]
            stop_region = regions[0] if regions else grid_regions((h, w), (h, w), routing.halo)[0]
            stop = learner.actions.execute(state, stop_region, Action.STOP, model)
            rgb_before = float(reconstruction_loss(state, stop))
            outcomes: dict[tuple[int | None, int], ActionResult] = {(None, 0): stop}
            observed: dict[tuple[int | None, int], dict[str, Any]] = {
                (None, 0): {"gain": 0.0, "harm": False, "action_ms": 0.0, "rgb_loss": rgb_before}
            }
            actual_gains = torch.zeros_like(gains)
            actual_harms = torch.zeros_like(harms)
            actual_available = executable.clone()
            skipped_actions = []
            for ri in eligible_ids:
                region = regions[ri]
                stereo_before = float(stereo_consistency_loss(state, stop.disparity, region))
                outside_before = float(outside_loss(state, stop, region))
                before_components = (
                    measure_quality(state, stop, region).values()
                    if cfg.schema == "intervention_selection.v2"
                    else None
                )
                for action in (Action.STEREO, Action.GAUSSIAN):
                    if not bool(executable[ri, int(action)]):
                        skipped_actions.append(
                            {
                                "region_id": ri,
                                "action": action.name,
                                "action_available": False,
                                "action_executed": False,
                                "action_unavailable_reason": "action_precondition",
                                "selector_label_available": False,
                                "action_ms": None,
                                "estimated_action_ms": None,
                                "within_estimated_budget": False,
                                "metric_changes": None,
                                "metric_feasible": None,
                            }
                        )
                        continue
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    outcome = learner.actions.execute(state, region, action, model)
                    torch.cuda.synchronize(device)
                    elapsed = (time.perf_counter() - t0) * 1000
                    rgb_after = float(reconstruction_loss(state, outcome))
                    stereo_after = float(stereo_consistency_loss(state, outcome.disparity, region))
                    collateral = float(outside_loss(state, outcome, region))
                    gain, harm = quality_gain(
                        rgb_before,
                        rgb_after,
                        stereo_before,
                        stereo_after,
                        outside_before,
                        collateral,
                        cfg,
                    )
                    component_record: dict[str, object] = {}
                    if before_components is not None:
                        after_components = measure_quality(state, outcome, region).values()
                        gain, local_gain, harm = component_targets(
                            before_components, after_components, cfg
                        )
                        component_record = {
                            "components_before": before_components,
                            "components_after": after_components,
                            "local_gain": local_gain,
                        }
                    key = (ri, int(action))
                    outcomes[key] = outcome
                    observed[key] = {
                        "gain": gain,
                        "harm": harm,
                        "action_ms": elapsed,
                        "rgb_loss": rgb_after,
                        "action_available": True,
                        "action_executed": True,
                        **component_record,
                    }
                    actual_gains[ri, int(action)] = gain
                    actual_harms[ri, int(action)] = float(harm)
            oracle = select_repairability(
                actual_gains,
                actual_harms,
                costs,
                executable,
                remaining,
                cfg,
                risk_available=actual_available,
            )
            selected_key = (decision.region_index, int(decision.action))
            oracle_key = (oracle.region_index, int(oracle.action))
            policies = {"base": (None, 0), "selector": selected_key, "oracle": oracle_key}
            fixed = eligible_ids[index % len(eligible_ids)] if eligible_ids else None
            policies.update(
                fixed_stereo=(fixed, 1)
                if fixed is not None
                and bool(executable[fixed, 1])
                and 0 < float(costs[fixed, 1]) <= remaining
                else (None, 0),
                fixed_gaussian=(fixed, 2)
                if fixed is not None
                and bool(executable[fixed, 2])
                and 0 < float(costs[fixed, 2]) <= remaining
                else (None, 0),
            )
            for name, action_id in (
                ("stereo_predicted_region", 1),
                ("gaussian_predicted_region", 2),
            ):
                allowed = [
                    ri
                    for ri in eligible_ids
                    if bool(executable[ri, action_id])
                    and 0 < float(costs[ri, action_id]) <= remaining
                ]
                ri = max(allowed, key=lambda ri: float(gains[ri, action_id])) if allowed else None
                policies[name] = (ri, action_id) if ri is not None else (None, 0)
            gt = adapter.load(
                sid,
                data_role="final_train",
                include_supervision=True,
                output_hw=construction.runtime.resolution_hw,
            ).supervision
            assert gt is not None
            gt_d, gt_mask = gt.disparity.to(device), gt.valid_mask.to(device)
            metric_cache: dict[tuple[int | None, int], dict[str, Any]] = {}
            metric_outcomes = {}
            if metric_config is not None:
                from reliable_endo_gs.evaluation.mask_metrics import (
                    intervention_metrics,
                    observation_metrics,
                )

                metric_cache[(None, 0)] = observation_metrics(stop, state, gt_d, gt_mask)
                for key, outcome in outcomes.items():
                    if key[0] is None:
                        continue
                    measured = observation_metrics(outcome, state, gt_d, gt_mask)
                    metric_cache[key] = measured
                    metric_outcomes[key] = intervention_metrics(
                        outcome,
                        stop,
                        state,
                        regions[key[0]],
                        gt_d,
                        gt_mask,
                        metric_action_policy(cfg, key[1])
                        if cfg.metric_quality is not None
                        else metric_config,
                        metric_cache[(None, 0)],
                        measured,
                    )
                    if cfg.metric_quality is not None:
                        changes = metric_outcomes[key]["metric_changes"]
                        observed[key]["proxy_gain"] = observed[key]["gain"]
                        observed[key]["proxy_harm"] = observed[key]["harm"]
                        observed[key]["selector_label_available"] = changes is not None
                        if changes is None:
                            actual_available[key[0], key[1]] = False
                            actual_gains[key[0], key[1]] = 0.0
                            actual_harms[key[0], key[1]] = 0.0
                            observed[key].update(gain=0.0, harm=False)
                        else:
                            gain, local_gain, harm = metric_targets(changes, cfg, key[1])
                            actual_gains[key[0], key[1]] = gain
                            actual_harms[key[0], key[1]] = float(harm)
                            observed[key].update(gain=gain, local_gain=local_gain, harm=harm)
            if cfg.metric_quality is not None:
                oracle = select_repairability(
                    actual_gains,
                    actual_harms,
                    costs,
                    executable,
                    remaining,
                    cfg,
                    risk_available=actual_available,
                )
                oracle_key = (oracle.region_index, int(oracle.action))
                policies["oracle"] = oracle_key
            policy_rows = {}
            for name, key in policies.items():
                outcome = outcomes[key]
                if key not in metric_cache:
                    metrics, _ = evaluate_geometry(
                        outcome.disparity,
                        gt_d,
                        gt_mask,
                        state.support & torch.isfinite(outcome.disparity),
                        state.batch.camera,
                        cloud_max_points=1024,
                        cloud_chunk_size=256,
                        gaussian_centres=outcome.field.means3d,
                        gaussian_valid=outcome.field.valid_mask,
                        gaussian_source_index=state.base.prior.source_index,
                        rendered_depth=outcome.left.depth,
                    )
                    for view, render, image in (
                        ("left", outcome.left, state.batch.left),
                        ("right", outcome.right, state.batch.right),
                    ):
                        metrics[view + "_psnr"] = psnr(render, image)
                        metrics[view + "_ssim"] = windowed_ssim(render.image, image)
                    metric_cache[key] = {k: v.to_dict() for k, v in metrics.items()}
                row = dict(observed[key])
                if not row.get("selector_label_available", True):
                    row.update(gain=None, local_gain=None, harm=None)
                actual_ms = (
                    base_ms
                    + row["action_ms"]
                    + (
                        routing_ms
                        if name
                        in ("selector", "stereo_predicted_region", "gaussian_predicted_region")
                        else 0.0
                    )
                )
                policy_rows[name] = {
                    "action": Action(key[1]).name,
                    "region": key[0],
                    **row,
                    "policy_ms_excluding_io": actual_ms,
                    "overrun_ms": max(0.0, actual_ms - cfg.total_budget_ms),
                    "estimated_action_ms": float(costs[key[0], key[1]])
                    if key[0] is not None
                    else 0.0,
                    "metrics": metric_cache[key],
                    **metric_outcomes.get(key, {}),
                }
            record = {
                "sample_id": sid,
                "gt_valid_pixels": int(gt_mask.sum()),
                "base_ms": base_ms,
                "routing_ms": routing_ms,
                "remaining_ms": remaining,
                "predicted_gains": gains.tolist(),
                "predicted_harms": harms.tolist(),
                "prediction_metadata": refit.prediction_metadata()
                if refit is not None
                else {
                    "source": "raw_joint_head",
                    "global_gain": "raw_expected_utility",
                    "risk_any": "raw_probability",
                    "risk_events": "raw_probabilities",
                    "gain_components": "raw_head_sign_and_magnitudes",
                    "gain_components_reconstruct": "predicted_gains",
                },
                "actual_gains": actual_gains.tolist(),
                "actual_harms": actual_harms.tolist(),
                "actual_label_available": actual_available.tolist(),
                "action_available": executable.tolist(),
                "oracle_agreement": selected_key == oracle_key,
                "valid_regions": valid.tolist(),
                "features_raw": features.detach().cpu().tolist(),
                "feature_normalization": (
                    refit.head if refit is not None else learner.selector
                ).normalization.diagnostics(features)
                if isinstance(learner.selector, BalancedSelector)
                else None,
                "safe_gain_regret": None
                if not observed[selected_key].get("selector_label_available", True)
                else observed[oracle_key]["gain"]
                - (
                    min(0.0, observed[selected_key]["gain"])
                    if observed[selected_key]["harm"]
                    else observed[selected_key]["gain"]
                ),
                "policies": policy_rows,
                "mask_candidates": [
                    {
                        "region_id": r.index,
                        "pixel_count": r.pixel_count,
                        "core": list(r.core),
                        "flat_indices": list(r.flat_indices) if isinstance(r, MaskRegion) else None,
                    }
                    for r in regions
                ],
                "counterfactual_metric_outcomes": [
                    {
                        "region_id": key[0],
                        "action": Action(key[1]).name,
                        "action_ms": observed[key]["action_ms"],
                        "action_available": True,
                        "action_executed": True,
                        "selector_label_available": observed[key].get(
                            "selector_label_available", True
                        ),
                        "estimated_action_ms": float(costs[key[0], key[1]]),
                        "within_estimated_budget": 0 < float(costs[key[0], key[1]]) <= remaining,
                        **value,
                    }
                    for key, value in metric_outcomes.items()
                ]
                + skipped_actions,
            }
            if cfg.selector_risk_mode == "local_global":
                record["predicted_risk_events"] = (
                    [] if event_predictions is None else event_predictions.cpu().tolist()
                )
            if cfg.selector_gain_mode == "signed_expectation":
                record["predicted_gain_components"] = (
                    {
                        name: []
                        for name in (
                            "positive_probability",
                            "positive_gain",
                            "negative_cost",
                            "expected_gain",
                        )
                    }
                    if gain_components is None
                    else gain_components.to_report()
                )
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            if index == 0 or (index + 1) % 8 == 0 or index + 1 == len(ids):
                print(
                    json.dumps(
                        {
                            "event": "selection_evaluation_progress",
                            "samples": index + 1,
                            "total": len(ids),
                        }
                    ),
                    flush=True,
                )
            del outcomes, state, metric_cache
    if _weights(learner) != before:
        raise ValueError("evaluation mutated model weights")
    if refit is not None and module_state_sha256(refit) != refit_before:
        raise ValueError("evaluation mutated calibrated selector")
    summaries = {}
    for policy in records[0]["policies"]:
        rows = [r["policies"][policy] for r in records]
        metric_names = rows[0]["metrics"]
        macro_metrics: dict[str, dict[str, object]] = {}
        for key in metric_names:
            values = [
                r["metrics"][key]["value"] for r in rows if r["metrics"][key]["value"] is not None
            ]
            macro_metrics[key] = {
                "macro_mean": statistics.mean(values) if values else None,
                "samples_available": len(values),
            }
        summaries[policy] = {
            "actions": dict(Counter(r["action"] for r in rows)),
            "mean_gain": statistics.mean(r["gain"] for r in rows if r["gain"] is not None)
            if any(r["gain"] is not None for r in rows)
            else None,
            "harmful_interventions": sum(r["harm"] is True for r in rows),
            "budget_overruns": sum(r["overrun_ms"] > 0 for r in rows),
            "metrics": macro_metrics,
            "metric_violations": sum(row.get("metric_feasible") is False for row in rows),
            "unknown_metric_interventions": sum(
                row["action"] != "STOP" and row.get("metric_feasible") is None for row in rows
            ),
        }
    probabilities = [
        p
        for r in records
        for region, valid_region, known in zip(
            r["predicted_harms"], r["valid_regions"], r["actual_label_available"], strict=True
        )
        if valid_region
        for p, available in zip(region[1:], known[1:], strict=True)
        if available
    ]
    labels = [
        y
        for r in records
        for region, valid_region, known in zip(
            r["actual_harms"], r["valid_regions"], r["actual_label_available"], strict=True
        )
        if valid_region
        for y, available in zip(region[1:], known[1:], strict=True)
        if available
    ]
    bins = []
    for b in range(10):
        pairs = [
            (p, y) for p, y in zip(probabilities, labels, strict=True) if min(9, int(p * 10)) == b
        ]
        if pairs:
            bins.append(
                {
                    "bin": b,
                    "count": len(pairs),
                    "mean_prediction": statistics.mean(p for p, _ in pairs),
                    "observed_proxy_harm": statistics.mean(y for _, y in pairs),
                }
            )
    calibration = {
        "label_semantics": "physical_metric_guard_violation"
        if cfg.metric_quality is not None
        else "fixed-mask proxy harm, not metric GT risk",
        "brier": statistics.mean((p - y) ** 2 for p, y in zip(probabilities, labels, strict=True))
        if probabilities
        else None,
        "ECE_10_bins": sum(
            row["count"] * abs(row["mean_prediction"] - row["observed_proxy_harm"]) for row in bins
        )
        / len(probabilities)
        if probabilities
        else None,
        "bins": bins,
    }
    result: dict[str, object] = {
        "status": "native_intervention_evaluation_complete",
        "samples": len(records),
        "samples_sha256": hashlib.sha256((output_dir / "samples.jsonl").read_bytes()).hexdigest(),
        "train_overlap": 0,
        # This exclusion does not establish an unseen evaluation cohort for
        # weights initialized from another training run. Audit that ancestry
        # separately before making generalization or final-test claims.
        "training_overlap_scope": "current_training_manifest_only",
        "initialization_overlap_audited": False,
        "evaluation_scope": "development_diagnostic",
        "manifest_hash": manifest.manifest_hash,
        "training_manifest_hash": train.manifest_hash,
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_source_sha256": raw["identity"]["source_sha256"],
        "checkpoint_import_mode": "explicit_pinned_historical_read_only_evaluation"
        if expected_checkpoint_sha256 is not None
        else "strict_current_source",
        "source_sha256": source_sha256(repo),
        "git_commit": get_git_commit(repo),
        "git_dirty": is_git_dirty(repo),
        "config_hash": cfg.config_hash,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable")),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "renderer_identity": renderer.identity,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": time.perf_counter() - started,
        "oracle_agreement": statistics.mean(r["oracle_agreement"] for r in records),
        "oracle_semantics": "measured physical metric utility and guards"
        if cfg.metric_quality is not None
        else "measured RGB/warp proxy gain and harm; not metric GT utility",
        "cost_calibration": learner.cost_calibration_metadata(),
        "mean_safe_gain_regret": statistics.mean(
            r["safe_gain_regret"] for r in records if r["safe_gain_regret"] is not None
        )
        if any(r["safe_gain_regret"] is not None for r in records)
        else None,
        "harm_calibration": calibration,
        "prediction_metadata": records[0]["prediction_metadata"],
        "risk_event_calibration": risk_event_metrics(records, cfg)
        if cfg.selector_risk_mode == "local_global"
        else None,
        "feature_names": list(selector_feature_names(cfg))
        if isinstance(learner.selector, BalancedSelector)
        else None,
        "weights_unchanged": True,
        "selector_refit_sha256": None
        if selector_refit_path is None
        else hashlib.sha256(selector_refit_path.read_bytes()).hexdigest(),
        "refit_calibration_available": None
        if not isinstance(refit, CalibratedSelector)
        else refit.calibration_available.cpu().tolist(),
        "teacher_inference_executed": False,
        "training_executed": False,
        "summary": summaries,
        "region_protocol": "fixed_grid.v1" if mask_config is None else mask_config.schema,
        "mask_config_hash": None if mask_config is None else mask_config.config_hash,
        "target_mask_index_sha256": None if masks is None else masks.identity,
        "metric_config_hash": None if metric_config is None else metric_config.config_hash,
        "base_only": base_only,
        "limitations": [
            "Same-keyframe diagnostic development, not sequence-disjoint final test.",
            "Warp residual and harm labels are proxies; GT metrics are reported independently.",
            "Single-run policy timings exclude I/O and oracle overhead; not a latency benchmark.",
            "No LPIPS; sampled clouds are diagnostics, not paper-protocol parity.",
        ],
    }
    if metric_config is not None:
        from reliable_endo_gs.evaluation.mask_metrics import summarize_metric_diagnostics

        result["metric_diagnostics"] = summarize_metric_diagnostics(records, metric_config)
    if cfg.selector_gain_mode == "signed_expectation":
        result["gain_event_diagnostics"] = gain_event_metrics(records)
    write_json_atomic(output_dir / "result.json", result)
    return result
