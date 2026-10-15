"""Sign/magnitude and safe-opportunity diagnostics from observed physical outcomes."""

from __future__ import annotations

import statistics
from typing import Any

from reliable_endo_gs.evaluation.risk_events import validate_prediction_rows


def gain_event_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Score complete valid candidates; oracle opportunities are offline diagnostics.

    Positive utility and physical safety are separate labels. Unknown outcomes
    are excluded. No equal-action quota or development-fitted scale is used.
    """
    result: dict[str, Any] = {}
    for record in records:
        components = record.get("predicted_gain_components")
        if not isinstance(components, dict):
            raise ValueError("gain event metrics require recorded raw sign/magnitude predictions")
        for component in ("positive_probability", "positive_gain", "negative_cost"):
            validate_prediction_rows(
                components.get(component),
                len(record["valid_regions"]),
                (3,),
                name=f"predicted_gain_components.{component}",
                probability=component == "positive_probability",
            )
    for name, ai in (("STEREO", 1), ("GAUSSIAN", 2)):
        pairs = []
        opportunities = 0
        captured = 0
        for record in records:
            oracle = record["policies"]["oracle"]
            selected = record["policies"]["selector"]
            if oracle["action"] == name:
                opportunities += 1
                captured += selected["action"] == name and selected["region"] == oracle["region"]
            components = record.get("predicted_gain_components")
            if components is None:
                raise ValueError("gain event metrics require recorded sign/magnitude predictions")
            for ri, valid in enumerate(record["valid_regions"]):
                if not valid or not record["actual_label_available"][ri][ai]:
                    continue
                actual = record["actual_gains"][ri][ai]
                pairs.append(
                    (
                        components["positive_probability"][ri][ai],
                        components["positive_gain"][ri][ai],
                        components["negative_cost"][ri][ai],
                        actual,
                        bool(record["actual_harms"][ri][ai]),
                    )
                )
        positive = [row for row in pairs if row[3] > 0]
        nonpositive = [row for row in pairs if row[3] <= 0]
        result[name] = {
            "prediction_space": "raw_head_sign_and_magnitudes",
            "known_candidates": len(pairs),
            "positive_gain_rate": statistics.mean(row[3] > 0 for row in pairs) if pairs else None,
            "positive_and_safe_rate": statistics.mean(row[3] > 0 and not row[4] for row in pairs)
            if pairs
            else None,
            "positive_probability_brier": statistics.mean(
                (row[0] - (row[3] > 0)) ** 2 for row in pairs
            )
            if pairs
            else None,
            "positive_probability_auroc": statistics.mean(
                (p[0] > n[0]) + 0.5 * (p[0] == n[0]) for p in positive for n in nonpositive
            )
            if positive and nonpositive
            else None,
            "positive_conditional_magnitude_mae": statistics.mean(
                abs(row[1] - row[3]) for row in positive
            )
            if positive
            else None,
            "nonpositive_conditional_magnitude_mae": statistics.mean(
                abs(row[2] + row[3]) for row in nonpositive
            )
            if nonpositive
            else None,
            "safe_oracle_opportunities": opportunities,
            "safe_oracle_opportunities_captured": captured,
            "safe_oracle_opportunity_recall": captured / opportunities if opportunities else None,
        }
    return result
