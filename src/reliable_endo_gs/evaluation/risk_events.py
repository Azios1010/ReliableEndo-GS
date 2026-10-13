"""Read-only physical event calibration, with unknown outcomes excluded."""

from __future__ import annotations

import statistics
from typing import Any

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.contracts.interventions import RISK_EVENT_NAMES
from reliable_endo_gs.training.metric_quality import metric_risk_targets


def risk_event_metrics(records: list[dict[str, Any]], config: InterventionConfig) -> dict[str, Any]:
    """Per-action/event candidate and selected Brier/rates; no model mutation."""
    result: dict[str, Any] = {}
    for action, ai in (("STEREO", 1), ("GAUSSIAN", 2)):
        pairs: list[tuple[list[float], tuple[bool, bool, bool], bool]] = []
        unknown = 0
        for record in records:
            for row in record["counterfactual_metric_outcomes"]:
                ri = row["region_id"]
                if row["action"] != action or not record["valid_regions"][ri]:
                    continue
                changes = row["metric_changes"]
                if changes is None or not record["actual_label_available"][ri][ai]:
                    unknown += 1
                    continue
                chosen = record["policies"]["selector"]
                pairs.append(
                    (
                        record["predicted_risk_events"][ri][ai],
                        metric_risk_targets(changes, config, ai),
                        chosen["action"] == action and chosen["region"] == ri,
                    )
                )
        summary: dict[str, Any] = {"unknown_candidates": unknown}
        for population in ("candidates", "selected"):
            available = [row for row in pairs if population == "candidates" or row[2]]
            summary[population] = {
                name: {
                    "observations": len(available),
                    "brier": statistics.mean((p[j] - y[j]) ** 2 for p, y, _ in available)
                    if available
                    else None,
                    "mean_prediction": statistics.mean(p[j] for p, _, _ in available)
                    if available
                    else None,
                    "harm_rate": statistics.mean(y[j] for _, y, _ in available)
                    if available
                    else None,
                }
                for j, name in enumerate(RISK_EVENT_NAMES)
            }
        result[action] = summary
    return result
