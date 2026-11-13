"""Replay decisions on immutable outcomes, including unknown selected outcomes."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

import torch

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.contracts.interventions import COMPONENT_NAMES
from reliable_endo_gs.evaluation.risk_events import probability_metrics
from reliable_endo_gs.routing.calibrated_selector import CalibratedSelector
from reliable_endo_gs.routing.repairability import select_repairability
from reliable_endo_gs.training.metric_quality import metric_risk_targets
from reliable_endo_gs.training.selector_outcomes import SelectorOutcomes


def keyframe_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row[field] is not None:
            groups[row["sample_id"].rsplit("/", 1)[0]].append(float(row[field]))
    return statistics.mean(statistics.mean(v) for v in groups.values()) if groups else None


@torch.no_grad()
def replay_policy(
    table: SelectorOutcomes,
    config: InterventionConfig,
    policy: str,
    *,
    repair_budget_ms: float,
    model: CalibratedSelector | None = None,
) -> list[dict[str, Any]]:
    """One intervention per frame, same canonical selector and dedicated TRAIN costs.

    Prediction sees all candidates, including those whose offline labels are
    unknown. The oracle alone uses label availability. Fixed repairs use cyclic
    eligible regions, without looking at utility/harm. Budget is repair-only;
    this replay is not a complete latency/FPS benchmark.
    """
    if policy not in ("stop", "fixed_stereo", "fixed_gaussian", "oracle", "selector"):
        raise ValueError("unknown replay policy")
    if policy == "selector" and model is None:
        raise ValueError("selector replay needs a frozen predictor")
    costs_vector = torch.tensor(table.metadata.get("deployment_costs_ms", []), dtype=torch.float32)
    if (
        costs_vector.shape != (3,)
        or not bool(torch.isfinite(costs_vector).all())
        or not bool((costs_vector[1:] > 0).all())
    ):
        raise ValueError("replay requires dedicated TRAIN deployment cost estimates")
    device = torch.device("cpu") if model is None else next(model.parameters()).device
    result = []
    for ordinal, record in enumerate(table.records):
        regions = record["interventions"]
        n = len(regions)
        costs = costs_vector.to(device).expand(n, -1)
        available = torch.tensor(
            [[a.get("action_available", True) for a in r["actions"]] for r in regions],
            dtype=torch.bool,
            device=device,
        ).reshape(n, 3)
        valid = torch.tensor(record["valid_regions"], dtype=torch.bool, device=device)
        available &= valid[:, None]
        ri, ai = None, 0
        if policy.startswith("fixed_") and n:
            eligible = available[:, 0].nonzero().flatten().tolist()
            fixed = eligible[ordinal % len(eligible)] if eligible else None
            action = 1 if policy == "fixed_stereo" else 2
            if (
                fixed is not None
                and bool(available[fixed, action])
                and float(costs[fixed, action]) <= repair_budget_ms
            ):
                ri, ai = fixed, action
        elif policy == "oracle" and n:
            gains, harms = torch.zeros_like(costs), torch.zeros_like(costs)
            known = available.clone()
            for j, region in enumerate(regions):
                for a, row in enumerate(region["actions"]):
                    known[j, a] &= row["selector_label_available"]
                    if row["selector_label_available"]:
                        gains[j, a], harms[j, a] = row["global_gain"], float(row["harm"])
            decision = select_repairability(
                gains, harms, costs, available, repair_budget_ms, config, risk_available=known
            )
            ri, ai = decision.region_index, int(decision.action)
        elif policy == "selector" and n:
            assert model is not None
            features = torch.tensor(record["features_raw"], dtype=torch.float32, device=device)
            decision = model.select(features, costs, available, repair_budget_ms)
            ri, ai = decision.region_index, int(decision.action)
        selected = None if ri is None else regions[ri]["actions"][ai]
        label_known = selected is None or selected["selector_label_available"]
        gain = 0.0 if selected is None else selected["global_gain"]
        cost = 0.0 if selected is None else float(costs_vector[ai])
        changes = [0.0] * len(COMPONENT_NAMES) if selected is None else selected["metric_changes"]
        result.append(
            {
                "sample_id": record["sample_id"],
                "region": ri,
                "action": ("STOP", "STEREO", "GAUSSIAN")[ai],
                "gain": gain,
                "net_gain": None if gain is None else gain - config.cost_weight_per_ms * cost,
                "harm": (False if selected is None else selected["harm"]) if label_known else None,
                "known": bool(label_known),
                "estimated_repair_ms": cost,
                "metric_changes": changes,
            }
        )
    return result


def summarize_replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Equal-keyframe summaries with explicit denominators for unknown selections."""
    selected = [r for r in rows if r["action"] != "STOP"]
    known = [r for r in selected if r["known"]]
    return {
        "frames": len(rows),
        "actions": dict(Counter(r["action"] for r in rows)),
        "known_selected": len(known),
        "unknown_selected": len(selected) - len(known),
        "selected_keyframes": len({r["sample_id"].rsplit("/", 1)[0] for r in known}),
        "mean_gain": keyframe_mean(rows, "gain"),
        "mean_net_gain": keyframe_mean(rows, "net_gain"),
        "selected_harm_rate": keyframe_mean(known, "harm"),
        "mean_estimated_repair_ms": keyframe_mean(rows, "estimated_repair_ms"),
        "metric_delta_means": {
            name: keyframe_mean(
                [
                    {
                        "sample_id": r["sample_id"],
                        "v": None if r["metric_changes"] is None else r["metric_changes"][i],
                    }
                    for r in rows
                ],
                "v",
            )
            for i, name in enumerate(COMPONENT_NAMES)
        },
        "mean_weighting": "equal_keyframe_then_equal_observed_frame",
        "unknown_is_safe": False,
        "timing_scope": "estimated_repair_only_not_FPS",
    }


@torch.no_grad()
def replay_risk_report(
    table: SelectorOutcomes, model: CalibratedSelector, *, repair_budget_ms: float
) -> dict[str, Any]:
    """Per-action/event candidate and selected calibration on the same prediction space."""
    selected = replay_policy(
        table, model.config, "selector", repair_budget_ms=repair_budget_ms, model=model
    )
    pools: dict[tuple[str, str, str], list[tuple[float, bool]]] = defaultdict(list)
    unknown: Counter[Any] = Counter()
    for record, decision in zip(table.records, selected, strict=True):
        if not record["interventions"]:
            continue
        x = torch.tensor(
            record["features_raw"], dtype=torch.float32, device=next(model.parameters()).device
        )
        _, _, any_risk, events, _ = model.outputs_with_components(x)
        for ri, region in enumerate(record["interventions"]):
            for ai, row in enumerate(region["actions"][1:], 1):
                action = row["action"]
                if not row["selector_label_available"]:
                    unknown[action] += 1
                    continue
                labels = metric_risk_targets(row["metric_changes"], model.config, ai)
                indices = (2,) if events is None else (0, 1, 2)
                for ei in indices:
                    event = ("local", "global", "any")[ei]
                    probability = float(any_risk[ri, ai] if events is None else events[ri, ai, ei])
                    pools[(action, "candidates", event)].append((probability, labels[ei]))
                    if decision["action"] == action and decision["region"] == ri:
                        pools[(action, "selected", event)].append((probability, labels[ei]))
    return {
        "prediction_metadata": model.prediction_metadata(),
        "unknown_candidates": dict(unknown),
        "populations": {
            "/".join(k): probability_metrics([p for p, _ in v], [y for _, y in v])
            for k, v in sorted(pools.items())
        },
        "unknown_is_safe": False,
    }
