"""Grouped checkpoint selection from TRAIN-derived validation only."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from reliable_endo_gs.data.intervention_table import OutcomeTable
from reliable_endo_gs.evaluation.outcome_selection import constraint_targets
from reliable_endo_gs.routing.outcome_predictor import OutcomePredictor
from reliable_endo_gs.routing.tradeoff import TradeoffConfig


def validate_training_views(fit: OutcomeTable, validation: OutcomeTable) -> None:
    """Reject external development, source mixing and shared frame/keyframe labels."""
    if any(t.metadata.get("collection_role") != "train" for t in (fit, validation)):
        raise ValueError("checkpoint selection must use TRAIN-derived views")
    if fit.metadata.get("outcome_sha256") != validation.metadata.get("outcome_sha256"):
        raise ValueError("fit/validation must originate from the same immutable training table")
    groups = [set(r["sample_id"].rsplit("/", 1)[0] for r in t.records) for t in (fit, validation)]
    if not all(groups) or groups[0] & groups[1]:
        raise ValueError("fit/validation keyframes must be nonempty and disjoint")


@torch.no_grad()
def validation_score(
    model: OutcomePredictor, table: OutcomeTable, config: TradeoffConfig
) -> dict[str, float]:
    """Equal-keyframe mean(normalized SmoothL1 + proper joint risk log loss).

    Feature/metric statistics remain fit-only. Unknown outcomes are masked;
    no class rebalance, action-frequency goal or development metric tuning.
    Mode is restored and no model buffers are updated by this scoring pass.
    """
    if table.metadata.get("collection_role") != "train":
        raise ValueError("validation scoring requires a TRAIN-derived view")
    mode = model.training
    model.eval()
    try:
        device = model.metric_scale.device
        prediction, logits = model(table.features.to(device))
        target = table.changes.to(device) / model.metric_scale
        risk = constraint_targets(table, config).to(device)
        observed = table.observed.to(device)
        regression = F.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
        log_loss = F.binary_cross_entropy_with_logits(logits, risk, reduction="none")
        groups: dict[str, list[int]] = {}
        for record, (start, end) in zip(table.records, table.slices, strict=True):
            groups.setdefault(record["sample_id"].rsplit("/", 1)[0], []).extend(range(start, end))
        values = []
        for indices in groups.values():
            index = torch.tensor(indices, device=device, dtype=torch.long)
            mask = observed[index]
            if bool(mask.any()):
                values.append(
                    torch.stack((regression[index][mask].mean(), log_loss[index][mask].mean()))
                )
        if not values:
            raise ValueError("validation has no observed metric outcomes")
        mean = torch.stack(values).mean(0)
        if not bool(torch.isfinite(mean).all()):
            raise ValueError("nonfinite grouped validation score")
        return {
            "score": float(mean.sum()),
            "regression": float(mean[0]),
            "risk_log_loss": float(mean[1]),
            "known_keyframes": float(len(values)),
        }
    finally:
        model.train(mode)
