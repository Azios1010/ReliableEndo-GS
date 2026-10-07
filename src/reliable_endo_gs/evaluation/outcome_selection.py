"""Read-only constrained action replay, group intervals and explicit cost accounting."""

from __future__ import annotations

import math
import random
import statistics
import time
from collections import Counter
from typing import Any

import torch

from reliable_endo_gs.data.intervention_table import OutcomeTable
from reliable_endo_gs.routing.tradeoff import (
    ACTION_NAMES,
    COMPONENT_NAMES,
    TradeoffConfig,
    TradeoffDecision,
    component_feasibility,
    feasible_changes,
    outcome_utility,
    select_tradeoff,
)


def constraint_targets(table: OutcomeTable, config: TradeoffConfig) -> torch.Tensor:
    """Unavailable cells are masked out, not supervised as safe/zero gain."""
    # Use the same canonical raw-unit decision as the oracle and deployment.
    # This label-only CPU pass avoids a second tensor-specific inequality rule.
    return table.changes.new_tensor(
        [
            [float(not feasible_changes(values, config)) for values in row]
            for row in table.changes.detach().cpu().tolist()
        ]
    )


def component_constraint_targets(table: OutcomeTable, config: TradeoffConfig) -> torch.Tensor:
    """Per-metric violation targets [R,A,C]; observation masks still apply."""
    return table.changes.new_tensor(
        [
            [[float(not ok) for ok in component_feasibility(values, config)] for values in row]
            for row in table.changes.detach().cpu().tolist()
        ]
    )


def bootstrap_difference(
    values: list[tuple[str, float]], config: TradeoffConfig
) -> dict[str, object]:
    groups: dict[str, list[float]] = {}
    for key, value in values:
        groups.setdefault(key, []).append(value)
    if len(groups) < 2:
        return {"groups": len(groups), "mean": None, "percentile_95": None}
    rng = random.Random(config.seed)
    keys = list(groups)
    samples = [
        statistics.mean(v for key in rng.choices(keys, k=len(keys)) for v in groups[key])
        for _ in range(config.bootstrap_repetitions)
    ]
    samples.sort()
    return {
        "groups": len(groups),
        "mean": statistics.mean(v for _, v in values),
        "percentile_95": [
            samples[int(0.025 * (len(samples) - 1))],
            samples[int(0.975 * (len(samples) - 1))],
        ],
        "unit": "normalized safe utility; unsafe selections count zero, violations also reported",
        "limitation": "cluster-bootstrap development keyframes; not independent sequence validation",
    }


def evaluate_outcome_predictions(
    table: OutcomeTable,
    changes: torch.Tensor,
    risks: torch.Tensor,
    *,
    scales: list[float],
    estimated_costs: torch.Tensor,
    config: TradeoffConfig,
    budgets_ms: tuple[float, ...],
    predictor_ms: tuple[float, ...],
    policy: str,
    action_restriction: str | None = None,
) -> tuple[dict[str, object], list[dict[str, Any]]]:
    """Read-only decisions before consulting measured outcome labels.

    Predictive costs are fitted on TRAIN, not per-development action timings.
    Actual repeated native costs are used only for reporting and offline oracle.
    Fixed policies use cyclic regions and do not pay feature/predictor cost.
    """
    if changes.shape != table.changes.shape or risks.shape != table.observed.shape:
        raise ValueError("predictive outcome shapes must match [R,A,C]/[R,A]")
    for budget in budgets_ms:
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("budgets must be positive finite milliseconds")
    predictions, probabilities = changes.cpu().tolist(), risks.cpu().tolist()
    results = []
    for fi, (record, (start, end)) in enumerate(zip(table.records, table.slices, strict=True)):
        eligible = table.eligible[start:end].tolist()
        estimated = estimated_costs[None].expand(end - start, -1).tolist()
        values = predictions[start:end]
        risk = probabilities[start:end]
        for budget in budgets_ms:
            routing_ms = (
                predictor_ms[fi]
                if policy not in ("constant", "oracle") and not policy.startswith("fixed_")
                else 0.0
            )
            feature_ms = record["feature_ms"] if routing_ms > 0 else 0.0
            base_ms = record["base_ms"] if routing_ms > 0 else record["base_ms_without_matching"]
            overhead = base_ms + feature_ms + routing_ms
            remaining = max(0.0, budget - overhead)
            t0 = time.perf_counter()
            if policy == "oracle":
                # Measured oracle knows outcomes and costs; this upper bound is not deployable.
                decision = select_tradeoff(
                    table.changes[start:end].tolist(),
                    constraint_targets(table, config)[start:end].tolist(),
                    table.costs[start:end].tolist(),
                    table.observed[start:end].tolist(),
                    remaining_ms=remaining,
                    scales=scales,
                    config=config,
                )
            elif policy.startswith("fixed_"):
                name = policy.removeprefix("fixed_")
                ai = ACTION_NAMES.index(name)
                regions = [ri for ri, row in enumerate(eligible) if row[ai]]
                ri = regions[fi % len(regions)] if regions else None
                decision = (
                    TradeoffDecision(ri, name, 0.0, float(estimated_costs[ai]))
                    if ri is not None and 0 < float(estimated_costs[ai]) <= remaining
                    else TradeoffDecision(None, "STOP", 0.0, 0.0)
                )
            else:
                if action_restriction is not None:
                    eligible = [
                        [v and ACTION_NAMES[ai] == action_restriction for ai, v in enumerate(row)]
                        for row in eligible
                    ]
                decision = select_tradeoff(
                    values,
                    risk,
                    estimated,
                    eligible,
                    remaining_ms=remaining,
                    scales=scales,
                    config=config,
                )
            allocation_ms = (time.perf_counter() - t0) * 1000
            # Oracle is an idealized ceiling. All deployable allocation costs count.
            if policy != "oracle":
                overhead += allocation_ms
                if decision.estimated_action_ms > max(0.0, budget - overhead):
                    decision = TradeoffDecision(None, "STOP", 0.0, 0.0)
            # Only after making the decision, look up metric observations/GT labels.
            available = True
            actual_safe = True
            score: float | None = 0.0
            vector = [0.0] * len(COMPONENT_NAMES)
            render_changes: dict[str, float | None] = dict.fromkeys(
                ("left_psnr", "right_psnr", "left_ssim", "right_ssim"), 0.0
            )
            action_ms = 0.0
            if decision.region_index is not None:
                ri = start + decision.region_index
                ai = ACTION_NAMES.index(decision.action)
                available = bool(table.observed[ri, ai])
                action_ms = float(table.costs[ri, ai])
                # Render availability is independent of local GT. Preserve its
                # denominator even when the complete geometry vector is unknown.
                outcome = next(
                    row
                    for row in record["outcomes"]
                    if row["region_id"] == decision.region_index
                    and row["action"] == decision.action
                )
                for name in render_changes:
                    before = record.get("base_metrics", {}).get(name, {}).get("value")
                    after = outcome.get("metrics", {}).get(name, {}).get("value")
                    render_changes[name] = (
                        float(after) - float(before)
                        if before is not None and after is not None
                        else None
                    )
                if available:
                    vector = table.changes[ri, ai].tolist()
                    actual_safe = feasible_changes(vector, config)
                    score = outcome_utility(vector, scales, config)
                else:
                    actual_safe = False
                    score = None
            results.append(
                {
                    "sample_id": record["sample_id"],
                    "policy": policy,
                    "budget_ms": budget,
                    "action": decision.action,
                    "region": decision.region_index,
                    "predicted_utility": decision.utility,
                    "metric_observation_available": available,
                    "actual_constraints_satisfied": actual_safe if available else None,
                    "actual_utility": score,
                    "safe_utility": score
                    if available and actual_safe
                    else (0.0 if available else None),
                    "metric_changes": vector if available else None,
                    "render_changes": render_changes,
                    "base_ms": base_ms,
                    "feature_ms": feature_ms,
                    "predictor_ms": routing_ms,
                    "allocation_ms": allocation_ms,
                    "estimated_action_ms": decision.estimated_action_ms,
                    "actual_action_max_ms": action_ms,
                    "policy_ms_excluding_io": overhead + action_ms,
                    "budget_overrun_ms": max(0.0, overhead + action_ms - budget),
                }
            )
    summary: dict[str, object] = {}
    for budget in budgets_ms:
        selected = [r for r in results if r["budget_ms"] == budget]
        known = [r for r in selected if r["metric_observation_available"]]
        summary[str(budget)] = {
            "actions": dict(Counter(r["action"] for r in selected)),
            "samples": len(selected),
            "metric_samples_available": len(known),
            "unknown_metric_selections": len(selected) - len(known),
            "measured_constraint_violations": sum(
                not r["actual_constraints_satisfied"] for r in known
            ),
            "mean_utility": statistics.mean(r["actual_utility"] for r in known) if known else None,
            "mean_safe_utility": statistics.mean(r["safe_utility"] for r in known)
            if known
            else None,
            "mean_metric_changes": {
                name: statistics.mean(r["metric_changes"][i] for r in known) if known else None
                for i, name in enumerate(COMPONENT_NAMES)
            },
            "render_changes_independent_of_gt": {
                name: {
                    "samples_available": len(
                        values := [
                            r["render_changes"][name]
                            for r in selected
                            if r["render_changes"][name] is not None
                        ]
                    ),
                    "mean_gain": statistics.mean(values) if values else None,
                }
                for name in ("left_psnr", "right_psnr", "left_ssim", "right_ssim")
            },
            "budget_overruns": sum(r["budget_overrun_ms"] > 0 for r in selected),
            "mean_policy_ms_excluding_io": statistics.mean(
                r["policy_ms_excluding_io"] for r in selected
            ),
        }
    return summary, results
