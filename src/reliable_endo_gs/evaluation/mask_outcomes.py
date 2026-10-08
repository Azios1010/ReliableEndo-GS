"""Immutable physical metric labels for frozen mask actions, not deployment GT."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.actions.geometry_repair import GeometryRepair, GeometryRepairConfig
from reliable_endo_gs.actions.repair import Action, ActionResult, RepairState, eligible
from reliable_endo_gs.contracts.interventions import ACTION_NAMES, COMPONENT_NAMES
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.evaluation.mask_metrics import intervention_metrics, observation_metrics
from reliable_endo_gs.regions.grid import Region, grid_regions
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES
from reliable_endo_gs.routing.cost import (
    CostCalibrationPolicy,
    DeploymentCostCalibration,
    capture_runtime_identity,
    save_cost_calibration,
)
from reliable_endo_gs.routing.tradeoff import TradeoffConfig
from reliable_endo_gs.training.student_learning import StudentLearner, evaluation_mode
from reliable_endo_gs.utils.io import write_json_atomic


def mask_candidate_hash(learner: StudentLearner, candidate: GeometryRepairConfig) -> str:
    """Bind proposals/feature ablation and the exploratory action configuration."""
    if learner.mask_config is None:
        raise ValueError("masked metric collection requires an explicit mask config")
    return hashlib.sha256(json.dumps({
        "mask_config_hash": learner.mask_config.config_hash,
        "correspondence_config_hash": candidate.config_hash,
    }, sort_keys=True).encode()).hexdigest()


def collect_mask_metric_table(
    *, learner: StudentLearner, samples: Iterable[ScaredA00Sample],
    gt_provider: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
    collection_role: str, provenance: Mapping[str, object], output_dir: Path,
    tradeoff: TradeoffConfig, candidate_config: GeometryRepairConfig,
    timing_policy: CostCalibrationPolicy,
) -> dict[str, Any]:
    """Freeze one snapshot, execute real actions, then request detached GT.

    Produces the existing metric_outcome_table.v1, reusable by outcome-pilot's
    grouped fit/validation/calibration. The third correspondence action is an
    explicitly named offline comparator; deployment dispatches only the two
    learned repairs. No moving-action labels or proxy risk substitution.
    """
    if collection_role not in ("train", "development") or learner.mask_config is None:
        raise ValueError("requires train/development role and masked learner")
    if learner.intervention_config is None or not tradeoff.schema.endswith(".v2"):
        raise ValueError("metric mask collection requires v2 physical guard policy")
    required = {"training_manifest_hash", "development_manifest_hash", "role_hash",
                "source_sha256", "renderer_identity", "gpu_uuid"}
    if not required <= set(provenance):
        raise ValueError("missing frozen collection provenance")
    repo = Path(__file__).resolve().parents[3]
    if repo == output_dir.resolve() or repo in output_dir.resolve().parents:
        raise ValueError("metric tables must stay outside repository")
    output_dir.mkdir(parents=True, exist_ok=False)
    device = next(learner.model.parameters()).device
    candidate = GeometryRepair(candidate_config)
    identity = learner.frozen_action_identity()
    before_selector = {k: v.detach().clone() for k, v in learner.selector.state_dict().items()}
    ids: list[str] = []
    all_costs: list[torch.Tensor] = []
    shapes: set[tuple[int, int]] = set()
    masks_context: list[dict[str, object]] = []

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # GeometryRepair's frozen-reference guard checks requires_grad. Preserve
    # flags as well as eval modes/RNG, including when GT or an action fails.
    modules = (learner.model, learner.actions, learner.selector)
    flags = [(p, p.requires_grad) for m in modules for p in m.parameters()]
    try:
        for module in modules:
            module.requires_grad_(False)
        with evaluation_mode(*modules), (output_dir / "samples.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            for sample in samples:
                sid = sample.sample_id
                if sample.supervision is not None or sample.data_role != "final_train" or (
                    sid.split("/")[0] not in {f"dataset_{i}" for i in range(1, 8)}
                ) or sid in ids:
                    raise ValueError("collection forbids GT inputs, duplicate IDs and final test")
                ids.append(sid)
                for _ in range(timing_policy.warmup_repeats):
                    learner.state(sample)
                sync()
                started = time.perf_counter()
                state = learner.state(sample)
                sync()
                base_ms = (time.perf_counter() - started) * 1000
                hw = (int(state.batch.left.shape[-2]), int(state.batch.left.shape[-1]))
                shapes.add(hw)
                started = time.perf_counter()
                regions = learner.candidate_regions(state, sid)
                features = learner.selection_features(state, regions) if regions else (
                    state.batch.left.new_empty((0, len(BALANCED_FEATURE_NAMES)))
                )
                sync()
                feature_ms = (time.perf_counter() - started) * 1000
                valid = [eligible(state, r) for r in regions]
                full = grid_regions(hw, hw, learner.routing.halo)[0]
                stop = learner.actions.execute(state, full, Action.STOP, learner.model)
                outcomes: list[tuple[int, str, ActionResult, list[float]]] = []
                for ri, region in enumerate(regions):
                    if not valid[ri]:
                        continue
                    masks_context.append({"sample_id": sid, "region": ri,
                        "pixels": region.pixel_count, "halo_yxyx": list(region.halo)})
                    action_timings: list[list[float]] = []
                    for ai, name in enumerate(ACTION_NAMES):
                        def execute(
                            ai: int = ai, state: RepairState = state, region: Region = region
                        ) -> ActionResult:
                            return candidate.execute(state, region, learner.model) if ai == 0 else (
                                learner.actions.execute(state, region, Action(ai), learner.model)
                            )
                        for _ in range(timing_policy.warmup_repeats):
                            execute()
                        times: list[float] = []
                        for _ in range(timing_policy.measure_repeats):
                            sync()
                            started = time.perf_counter()
                            outcome = execute()
                            sync()
                            times.append((time.perf_counter() - started) * 1000)
                        outcomes.append((ri, name, outcome, times))
                        action_timings.append(times)
                    # Cost artifact enum order: STOP, learned stereo, Gaussian.
                    all_costs.append(torch.tensor([times for times in
                        ([0.0] * timing_policy.measure_repeats, *action_timings[1:])]))
                # GT is requested only after proposals/features/all real repairs.
                gt_d, gt_mask = gt_provider(sid)
                gt_d, gt_mask = gt_d.detach().to(device), gt_mask.detach().to(device)
                baseline = observation_metrics(stop, state, gt_d, gt_mask)
                rows = []
                for ri, name, outcome, times in outcomes:
                    after = observation_metrics(outcome, state, gt_d, gt_mask)
                    measured = intervention_metrics(outcome, stop, state, regions[ri],
                        gt_d, gt_mask, tradeoff, baseline, after)
                    rows.append({"region_id": ri, "action": name,
                        "metric_changes": measured["metric_changes"],
                        "metric_observation_available": measured["complete_metric_observation"],
                        "action_ms_samples": times, "action_ms_max": max(times),
                        "changed_fields": list(outcome.changed_fields),
                        "no_op": not bool(outcome.changed_fields),
                        "guards": measured["component_feasible"]})
                record = {"sample_id": sid, "features_raw": features.cpu().tolist(),
                    "valid_regions": valid, "outcomes": rows, "base_ms": base_ms,
                    # Mask features require shared matching evidence. Keeping
                    # this equal pays the full base in every source ablation.
                    "base_ms_without_matching": base_ms, "feature_ms": feature_ms,
                    "regions": [{"core_yxyx": list(r.core), "halo_yxyx": list(r.halo),
                                 "pixels": r.pixel_count} for r in regions],
                    "gt_valid_pixels": int(gt_mask.sum())}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
    if not ids or learner.frozen_action_identity() != identity or any(
        not torch.equal(v, learner.selector.state_dict()[k]) for k, v in before_selector.items()
    ):
        raise ValueError("empty collection or mutation of frozen model/action/normalization")
    result = {**dict(provenance), "status": "metric_outcome_table_complete",
        "table_schema": "metric_outcome_table.v1", "collection_role": collection_role,
        "samples": len(ids), "sample_ids": ids,
        "sample_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "outcome_sha256": hashlib.sha256((output_dir / "samples.jsonl").read_bytes()).hexdigest(),
        "imported_weights": identity, "candidate_config_hash": mask_candidate_hash(learner, candidate_config),
        "action_names": list(ACTION_NAMES), "component_names": list(COMPONENT_NAMES),
        "feature_names": list(BALANCED_FEATURE_NAMES), "tradeoff_config": asdict(tradeoff),
        "tradeoff_config_hash": tradeoff.config_hash,
        "timing_repeats": timing_policy.measure_repeats, "timing_policy": asdict(timing_policy),
        "mask_config": asdict(learner.mask_config), "candidate_config": asdict(candidate_config),
        "weights_unchanged": True, "teacher_inference_executed": False, "training_executed": False,
        "selector_executed": False, "runtime": capture_runtime_identity(device),
        "target_mask_index_sha256": None if learner.target_masks is None else learner.target_masks.identity,
        "timing_scope": "base/proposal/features/actions measured; excludes I/O/GT/oracle collection",
        "rendered_depth_available": False, "gate_II_certified": False}
    if collection_role == "train" and all_costs and len(shapes) == 1:
        timings = torch.cat(all_costs, dim=1)
        policy = CostCalibrationPolicy(timing_policy.tail_policy, timing_policy.quantile,
            timing_policy.warmup_repeats, int(timings.shape[1]))
        if policy.tail_policy == "max":
            estimate = timings.amax(1)
        else:
            if policy.quantile is None:
                raise ValueError("quantile tail policy requires a quantile")
            estimate = timings.quantile(policy.quantile, dim=1)
        calibration = DeploymentCostCalibration(
            {"collection_role": "train", "frozen_action_identity": identity,
             "training_manifest_hash": provenance["training_manifest_hash"],
             "sample_ids_sha256": result["sample_ids_sha256"], "mask_context": masks_context},
            policy, capture_runtime_identity(device), next(iter(shapes)), timings, estimate,
        )
        save_cost_calibration(output_dir / "action_costs.pt", calibration)
        result["cost_artifact"] = "action_costs.pt"
    write_json_atomic(output_dir / "result.json", result)
    return result
