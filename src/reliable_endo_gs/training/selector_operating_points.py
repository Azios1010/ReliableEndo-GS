"""CAL-only empirical harm budgets; probabilities do not certify safety."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from reliable_endo_gs.config.selector_study import SelectorStudyConfig
from reliable_endo_gs.evaluation.selector_policy import replay_policy, summarize_replay
from reliable_endo_gs.routing.calibrated_selector import CalibratedSelector
from reliable_endo_gs.training.selector_outcomes import SelectorOutcomes


def calibrate_operating_points(
    model: CalibratedSelector, calibration: SelectorOutcomes, study: SelectorStudyConfig
) -> list[dict[str, Any]]:
    """Shared cutoff grid, four budgets, no validation/development cutoff selection.

    Same CAL data calibrates probabilities and empirically chooses cutoffs. This
    is exploratory empirical compliance, with no independent safety guarantee.
    Unknown selected outcomes invalidate a candidate; unsupported heads abstain.
    """
    if calibration.metadata.get("collection_role") != "train":
        raise ValueError("operating points require held-out TRAIN calibration")
    original = model.config
    fixed = {
        name: summarize_replay(
            replay_policy(calibration, original, name, repair_budget_ms=study.repair_budget_ms)
        )
        for name in ("stop", "fixed_stereo", "fixed_gaussian")
    }
    best_fixed = max(
        fixed,
        key=lambda k: (
            fixed[k]["mean_net_gain"] if fixed[k]["mean_net_gain"] is not None else float("-inf")
        ),
    )
    cap = fixed[best_fixed]["selected_harm_rate"]
    # STOP's violation rate is structurally zero. Unknown fixed outcomes cannot
    # establish a permissive reference budget.
    cap = 0.0 if cap is None or fixed[best_fixed]["unknown_selected"] else cap
    candidates = []
    try:
        for cutoff in study.cutoff_candidates:
            model.config = replace(
                original,
                maximum_stereo_harm_probability=cutoff,
                maximum_gaussian_harm_probability=cutoff,
            )
            summary = summarize_replay(
                replay_policy(
                    calibration,
                    model.config,
                    "selector",
                    repair_budget_ms=study.repair_budget_ms,
                    model=model,
                )
            )
            candidates.append({"cutoff": cutoff, **summary})
    finally:
        model.config = original
    result = []
    for budget in study.harm_budgets:
        effective = min(budget, cap)
        eligible = [
            c
            for c in candidates
            if not c["unknown_selected"]
            and c["known_selected"] >= study.minimum_selected_observations
            and c["selected_keyframes"] >= study.minimum_selected_keyframes
            and c["selected_harm_rate"] is not None
            and c["selected_harm_rate"] <= effective
        ]
        selected = (
            max(
                eligible,
                key=lambda c: (c["mean_net_gain"], -c["mean_estimated_repair_ms"], -c["cutoff"]),
            )
            if eligible
            else None
        )
        result.append(
            {
                "schema": "CAL_empirical_harm_budget.v1",
                "budget": budget,
                "effective_harm_budget": effective,
                "best_fixed_reference": best_fixed,
                "status": "selected" if selected is not None else "abstain",
                "selected": selected,
                "candidates": candidates,
                "calibration_outcomes_sha256": calibration.metadata["samples_sha256"],
                "calibration_sample_ids": list(dict.fromkeys(calibration.sample_ids)),
                "minimum_selected_keyframes": study.minimum_selected_keyframes,
                "minimum_selected_observations": study.minimum_selected_observations,
                "development_used": False,
                "safety_guarantee": False,
            }
        )
    return result
