"""Independent fixed-support metric diagnostics for real sparse interventions."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Any

import torch

from reliable_endo_gs.actions.repair import ActionResult, RepairState
from reliable_endo_gs.evaluation.geometry_native import _geometry_metrics
from reliable_endo_gs.evaluation.rendering import psnr, windowed_ssim
from reliable_endo_gs.evaluation.tradeoff_outcomes import depth_snapshot, metric_changes
from reliable_endo_gs.geometry.rectified import reconstruct_rectified
from reliable_endo_gs.regions.grid import Region
from reliable_endo_gs.routing.tradeoff import (
    COMPONENT_NAMES,
    TradeoffConfig,
    component_feasibility,
    feasible_changes,
)
from reliable_endo_gs.training.quality_components import measure_quality


def observation_metrics(
    outcome: ActionResult, state: RepairState, gt_d: torch.Tensor, gt_mask: torch.Tensor
) -> dict[str, Any]:
    """Existing geometry owner (mm) and RGB owners, without model-state writes."""
    result = _geometry_metrics(outcome, state, gt_d, gt_mask)
    result.update(
        left_psnr=psnr(outcome.left, state.batch.left).to_dict(),
        right_psnr=psnr(outcome.right, state.batch.right).to_dict(),
        left_ssim=windowed_ssim(outcome.left.image, state.batch.left).to_dict(),
        right_ssim=windowed_ssim(outcome.right.image, state.batch.right).to_dict(),
    )
    return result


def intervention_metrics(
    outcome: ActionResult,
    stop: ActionResult,
    state: RepairState,
    region: Region,
    gt_d: torch.Tensor,
    gt_mask: torch.Tensor,
    config: TradeoffConfig,
    baseline: dict[str, Any],
    measured: dict[str, Any],
) -> dict[str, Any]:
    """Fixed base+GT denominators; unknown metrics cannot certify safety.

    Call only AFTER deployment decisions/actions. No GT enters region proposal,
    features or the learned decision. This oracle is bounded by real candidates.
    """
    geometry = reconstruct_rectified(
        gt_d, state.batch.camera, gt_mask, min_effective_disparity=1e-6
    )
    before = depth_snapshot(
        stop, stop, state, geometry.depth.depth, geometry.depth.valid_mask, region
    )
    after = depth_snapshot(
        outcome, stop, state, geometry.depth.depth, geometry.depth.valid_mask, region
    )
    qb = measure_quality(state, stop, region).values()
    qa = measure_quality(state, outcome, region).values()
    changes = metric_changes(
        baseline,
        measured,
        before,
        after,
        qb["outside_rgb"],
        qa["outside_rgb"],
        qb["local_rgb"],
        qa["local_rgb"],
    )
    return {
        "metric_changes": changes,
        "complete_metric_observation": changes is not None,
        "metric_feasible": feasible_changes(changes, config) if changes is not None else None,
        "component_feasible": list(component_feasibility(changes, config))
        if changes is not None
        else None,
        "depth_before_fixed_support": before,
        "depth_after_fixed_support": after,
        "outside_rgb_change": qa["outside_rgb"] - qb["outside_rgb"],
        "local_rgb_change": qa["local_rgb"] - qb["local_rgb"],
    }


def summarize_metric_diagnostics(
    records: Sequence[dict[str, Any]], config: TradeoffConfig
) -> dict[str, Any]:
    """Separate action feasibility from selection errors, without mixing units.

    An opportunity means all raw-unit guards pass and at least one rewarded
    component improves beyond the declared numerical epsilon. It is NOT a
    balanced-utility winner: no scales are fitted on development outcomes.
    Unknown observations never count as safe. Budgeted opportunities use the
    same pre-action cost estimates as deployment, not measured oracle latency.
    """
    counters: dict[str, Counter[str]] = {name: Counter() for name in ("STEREO", "GAUSSIAN")}
    changes: dict[str, list[list[float]]] = {name: [] for name in counters}
    failures: dict[str, Counter[str]] = {name: Counter() for name in counters}
    opportunities = misses = unsafe_selected = unknown_selected = 0
    for record in records:
        available = []
        for row in record["counterfactual_metric_outcomes"]:
            action = row["action"]
            counters[action]["observations"] += 1
            vector = row["metric_changes"]
            if vector is None:
                counters[action]["unknown"] += 1
                continue
            changes[action].append(vector)
            safe = row["metric_feasible"] is True
            counters[action]["feasible" if safe else "violations"] += 1
            for name, passed in zip(COMPONENT_NAMES, row["component_feasible"], strict=True):
                if not passed:
                    failures[action][name] += 1
            improved = safe and any(
                weight > 0 and value > config.comparison_epsilon
                for value, weight in zip(vector, config.score_weights, strict=True)
            )
            counters[action]["feasible_with_some_improvement"] += improved
            if improved and row["within_estimated_budget"]:
                available.append((row["region_id"], action))
                counters[action]["budgeted_opportunities"] += 1
        selected = record["policies"]["selector"]
        key = (selected["region"], selected["action"])
        if available:
            opportunities += 1
            misses += key not in available
        if selected["action"] != "STOP":
            unsafe_selected += selected.get("metric_feasible") is False
            unknown_selected += selected.get("metric_feasible") is None
    return {
        "semantics": "raw-unit feasibility and component improvements; not a normalized utility oracle",
        "samples_with_budgeted_opportunity": opportunities,
        "selector_missed_opportunity_samples": misses,
        "selector_metric_violations": unsafe_selected,
        "selector_unknown_metric_interventions": unknown_selected,
        "actions": {
            name: {
                **dict(counters[name]),
                "failed_guards": dict(failures[name]),
                "mean_changes_raw_units": dict(
                    zip(
                        COMPONENT_NAMES,
                        (statistics.mean(column) for column in zip(*changes[name], strict=True)),
                        strict=True,
                    )
                )
                if changes[name]
                else None,
            }
            for name in counters
        },
        "guard_config_hash": config.config_hash,
    }
