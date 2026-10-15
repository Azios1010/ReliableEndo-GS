"""Read-only physical event calibration, with unknown outcomes excluded."""

from __future__ import annotations

import statistics
from typing import Any

import torch

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.contracts.interventions import RISK_EVENT_NAMES
from reliable_endo_gs.training.metric_quality import metric_risk_targets


def probability_metrics(probabilities: list[float], labels: list[bool]) -> dict[str, Any]:
    """Discrimination and ten fixed-bin calibration on known physical events."""
    pairs = list(zip(probabilities, labels, strict=True))
    positives = [p for p, y in pairs if y]
    negatives = [p for p, y in pairs if not y]
    bins = []
    for index in range(10):
        entries = [(p, y) for p, y in pairs if min(9, int(p * 10)) == index]
        if entries:
            bins.append(
                {
                    "bin": index,
                    "observations": len(entries),
                    "mean_prediction": statistics.mean(p for p, _ in entries),
                    "harm_rate": statistics.mean(y for _, y in entries),
                }
            )
    return {
        "observations": len(pairs),
        "brier": statistics.mean((p - y) ** 2 for p, y in pairs) if pairs else None,
        "mean_prediction": statistics.mean(probabilities) if pairs else None,
        "harm_rate": statistics.mean(labels) if pairs else None,
        "auroc": statistics.mean((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
        if positives and negatives
        else None,
        "ECE_10_bins": sum(
            row["observations"] * abs(row["mean_prediction"] - row["harm_rate"]) for row in bins
        )
        / len(pairs)
        if pairs
        else None,
        "bins": bins,
        "mean_prediction_when_harmful": statistics.mean(positives) if positives else None,
        "mean_prediction_when_safe": statistics.mean(negatives) if negatives else None,
    }


def validate_prediction_rows(
    values: Any, regions: int, trailing_shape: tuple[int, ...], *, name: str, probability: bool
) -> None:
    """Fail explicitly on missing/misaligned outputs before candidate indexing."""
    try:
        tensor = torch.as_tensor(values, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must contain aligned finite prediction rows") from error
    if regions == 0 and tensor.numel() == 0:
        return
    if (
        tensor.shape != (regions, *trailing_shape)
        or not bool(torch.isfinite(tensor).all())
        or (probability and not bool(((tensor >= 0) & (tensor <= 1)).all()))
    ):
        raise ValueError(
            f"{name} must contain finite [R,{','.join(map(str, trailing_shape))}] rows"
        )


def risk_event_metrics(records: list[dict[str, Any]], config: InterventionConfig) -> dict[str, Any]:
    """Per-action/event candidate and selected Brier/rates; no model mutation."""
    result: dict[str, Any] = {}
    for record in records:
        validate_prediction_rows(
            record.get("predicted_risk_events"),
            len(record["valid_regions"]),
            (3, len(RISK_EVENT_NAMES)),
            name="predicted_risk_events",
            probability=True,
        )
    for action, ai in (("STEREO", 1), ("GAUSSIAN", 2)):
        pairs: list[tuple[list[float], tuple[bool, bool, bool], bool]] = []
        unknown = 0
        nonexecutable = 0
        for record in records:
            for row in record["counterfactual_metric_outcomes"]:
                ri = row["region_id"]
                if row["action"] != action or not record["valid_regions"][ri]:
                    continue
                if not row.get("action_available", True):
                    nonexecutable += 1
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
        summary: dict[str, Any] = {
            "unknown_candidates": unknown,
            "nonexecutable_candidates": nonexecutable,
        }
        for population in ("candidates", "selected"):
            available = [row for row in pairs if population == "candidates" or row[2]]
            summary[population] = {
                name: probability_metrics(
                    [p[j] for p, _, _ in available], [y[j] for _, y, _ in available]
                )
                for j, name in enumerate(RISK_EVENT_NAMES)
            }
            summary[population]["joint_probability_inconsistencies"] = sum(
                p[2] < max(p[:2]) - 1e-6 or p[2] > min(1.0, sum(p[:2])) + 1e-6
                for p, _, _ in available
            )
        result[action] = summary
    return result
