"""Training-only grouped risk calibration; immutable native outcomes stay intact."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Any

import torch
from torch.nn import functional as F

from reliable_endo_gs.config.outcome_training import OutcomeTrainingConfig
from reliable_endo_gs.data.intervention_table import OutcomeTable
from reliable_endo_gs.evaluation.outcome_selection import constraint_targets
from reliable_endo_gs.routing.outcome_predictor import OutcomePredictor
from reliable_endo_gs.routing.tradeoff import ACTION_NAMES, TradeoffConfig


def _frame_view(table: OutcomeTable, frames: list[int]) -> OutcomeTable:
    """In-memory indexed view, not a rewritten complete native table artifact."""
    indices: list[int] = []
    slices = []
    for frame in frames:
        start, end = table.slices[frame]
        offset = len(indices)
        indices.extend(range(start, end))
        slices.append((offset, len(indices)))
    index = torch.tensor(indices, dtype=torch.long)
    records = [table.records[i] for i in frames]
    return replace(
        table,
        metadata={
            **table.metadata,
            "in_memory_frame_view": True,
            "sample_ids": [r["sample_id"] for r in records],
            "samples": len(records),
        },
        records=records,
        features=table.features[index],
        changes=table.changes[index],
        observed=table.observed[index],
        eligible=table.eligible[index],
        costs=table.costs[index],
        slices=tuple(slices),
    )


def split_training_calibration(
    table: OutcomeTable, config: TradeoffConfig
) -> tuple[OutcomeTable, OutcomeTable, dict[str, Any]]:
    """One hashed keyframe/fold held out solely from the existing TRAIN table.

    Native train/development/final-test roles and parent manifests are unchanged.
    Fit/calibration share neither frames nor keyframes. Both are recorded as
    derived views bound to the original immutable table checksum.
    """
    if table.metadata["collection_role"] != "train":
        raise ValueError("risk calibration partition must originate in training data")
    groups: dict[str, set[str]] = {}
    for record in table.records:
        sid = record["sample_id"]
        fold, keyframe = sid.split("/")[:2]
        groups.setdefault(fold, set()).add(f"{fold}/{keyframe}")
    held_out = set()
    for fold, keys in groups.items():
        if len(keys) < 2:
            raise ValueError(f"need at least two training keyframes in {fold} for calibration")
        held_out.add(
            min(keys, key=lambda key: hashlib.sha256(f"{config.seed}:{key}".encode()).hexdigest())
        )
    cal = [i for i, r in enumerate(table.records) if r["sample_id"].rsplit("/", 1)[0] in held_out]
    fit = [i for i in range(len(table.records)) if i not in cal]
    report = {
        "source_training_table_sha256": table.metadata["outcome_sha256"],
        "fit_sample_ids": [table.records[i]["sample_id"] for i in fit],
        "calibration_sample_ids": [table.records[i]["sample_id"] for i in cal],
        "calibration_keyframes": sorted(held_out),
        "seed": config.seed,
        "policy": "one_sha256_selected_training_keyframe_per_fold",
        "development_used_for_fitting": False,
        "limitation": "small train-only calibration; shared-keyframe development is not independent validation",
    }
    report["partition_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode()
    ).hexdigest()
    return _frame_view(table, fit), _frame_view(table, cal), report


def split_training_validation_calibration(
    table: OutcomeTable, config: TradeoffConfig
) -> tuple[OutcomeTable, OutcomeTable, OutcomeTable, dict[str, Any]]:
    """Three disjoint TRAIN-derived keyframe views; calibration groups preserved.

    One additional seed-hashed keyframe per fold chooses checkpoints; that data
    is never used to fit risk calibration. At least three keyframes/fold are
    required. Insufficient group diversity is explicit, not a frame-level fallback.
    """
    fit_pool, calibration, old = split_training_calibration(table, config)
    groups: dict[str, set[str]] = {}
    for record in fit_pool.records:
        key = record["sample_id"].rsplit("/", 1)[0]
        groups.setdefault(key.split("/")[0], set()).add(key)
    validation_groups = set()
    for fold, keys in groups.items():
        if len(keys) < 2:
            raise ValueError(
                f"need at least three training keyframes in {fold} for independent validation/calibration"
            )
        validation_groups.add(
            min(
                keys,
                key=lambda key: hashlib.sha256(
                    f"validation:{config.seed}:{key}".encode()
                ).hexdigest(),
            )
        )
    val_frames = [
        i
        for i, r in enumerate(fit_pool.records)
        if r["sample_id"].rsplit("/", 1)[0] in validation_groups
    ]
    fit_frames = [i for i in range(len(fit_pool.records)) if i not in val_frames]
    fit, validation = _frame_view(fit_pool, fit_frames), _frame_view(fit_pool, val_frames)
    report = {
        **{k: v for k, v in old.items() if k != "partition_sha256"},
        "fit_sample_ids": [r["sample_id"] for r in fit.records],
        "validation_sample_ids": [r["sample_id"] for r in validation.records],
        "validation_keyframes": sorted(validation_groups),
        "policy": "one_calibration_and_one_validation_training_keyframe_per_fold",
        "validation_used_for_calibration": False,
    }
    report["partition_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode()
    ).hexdigest()
    return fit, validation, calibration, report


@torch.no_grad()
def calibrate_joint_risk(
    model: OutcomePredictor,
    calibration: OutcomeTable,
    config: TradeoffConfig,
    training_policy: OutcomeTrainingConfig | None = None,
) -> dict[str, Any]:
    """Fit temperature/intercept on disjoint TRAIN keyframes using proper log loss.

    Identity (T=1,b=0) is included. No cutoff is selected using development.
    Per-action support needs both classes across enough distinct keyframes;
    unsupported sources remain explicitly unavailable for predictive decisions.
    Calibration is not a statistical safety guarantee.
    """
    if not model.component_risk or calibration.metadata["collection_role"] != "train":
        raise ValueError("component-risk calibration requires training-only v2 outcomes")
    device = model.metric_scale.device
    mode = model.training
    model.eval()
    try:
        _, logits = model(calibration.features.to(device))
    finally:
        model.train(mode)
    target = constraint_targets(calibration, config).to(device)
    model.risk_temperature.fill_(1)
    model.risk_bias.zero_()
    model.risk_calibrated.zero_()
    temperatures = (1.0, 0.5, 2.0)
    biases = (0.0, -2.0, -1.0, 1.0, 2.0)
    method = training_policy.calibration_method if training_policy is not None else "grid"
    per_action = {}
    for ai, action in enumerate(ACTION_NAMES):
        known = calibration.observed[:, ai].to(device)
        classes: dict[int, set[str]] = {0: set(), 1: set()}
        for record, (start, end) in zip(calibration.records, calibration.slices, strict=True):
            keyframe = record["sample_id"].rsplit("/", 1)[0]
            for label in (0, 1):
                if bool((known[start:end] & target[start:end, ai].eq(label)).any()):
                    classes[label].add(keyframe)
        x, y = logits[:, ai][known], target[:, ai][known]
        supported = all(
            len(classes[label]) >= config.calibration_min_class_groups for label in (0, 1)
        )
        report: dict[str, Any] = {
            "observations": len(y),
            "safe_keyframes": len(classes[0]),
            "unsafe_keyframes": len(classes[1]),
            "supported": supported,
            "status": "insufficient_class_group_support",
        }
        if supported:
            if method == "platt":
                assert training_policy is not None
                # Every observed keyframe receives equal mass, without rebalancing
                # safe/unsafe classes or inspecting development outcomes.
                weights = torch.zeros_like(target[:, ai])
                groups: dict[str, list[int]] = {}
                for record, (start, end) in zip(
                    calibration.records, calibration.slices, strict=True
                ):
                    groups.setdefault(record["sample_id"].rsplit("/", 1)[0], []).extend(
                        range(start, end)
                    )
                for indices in groups.values():
                    index = torch.tensor(indices, device=device)
                    valid = index[known[index]]
                    if valid.numel():
                        weights[valid] = 1 / valid.numel()
                weight = weights[known]
                weight /= weight.sum()
                temperature, bias = fit_platt_risk(
                    x, y, weight, regularization=training_policy.calibration_regularization
                )
                objective_weights = weight
            else:
                best = (float("inf"), 1.0, 0.0)
                for temperature in temperatures:
                    for bias in biases:
                        loss = float(F.binary_cross_entropy_with_logits(x / temperature + bias, y))
                        if loss < best[0]:
                            best = (loss, temperature, bias)
                _, temperature, bias = best
                objective_weights = torch.ones_like(y) / len(y)
            model.risk_temperature[ai] = temperature
            model.risk_bias[ai] = bias
            model.risk_calibrated[ai] = True
            report.update(
                status="train_group_temperature_intercept_fitted",
                temperature=float(model.risk_temperature[ai]),
                bias=float(model.risk_bias[ai]),
                raw_log_loss=float(
                    (
                        F.binary_cross_entropy_with_logits(x, y, reduction="none")
                        * objective_weights
                    ).sum()
                ),
                calibrated_log_loss=float(
                    (
                        F.binary_cross_entropy_with_logits(
                            x / model.risk_temperature[ai] + model.risk_bias[ai],
                            y,
                            reduction="none",
                        )
                        * objective_weights
                    ).sum()
                ),
            )
        per_action[action] = report
    return {
        "actions": per_action,
        "temperature_grid": list(temperatures) if method == "grid" else None,
        "bias_grid": list(biases) if method == "grid" else None,
        "platt_bounds": {"log_slope": [-2, 2], "bias": [-4, 4], "max_iter": 80}
        if method == "platt"
        else None,
        "cutoff": config.maximum_predicted_violation_probability,
        "cutoff_tuned": False,
        "calibration_guarantees_safety": False,
        "method": method,
        "objective_population": "equal_keyframe" if method == "platt" else "observed_regions",
        "regularization": training_policy.calibration_regularization
        if method == "platt" and training_policy is not None
        else None,
    }


def fit_platt_risk(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, *, regularization: float
) -> tuple[float, float]:
    """Monotone bounded Platt transform of detached known outcomes [N].

    Minimize weighted proper log loss + ridge on log-slope/intercept. Fixed
    bounds log-slope[-2,2], intercept[-4,4] and80 LBFGS iterations limit small-
    sample extrapolation. No cutoff fitting. Identity is the fallback if the
    penalized objective is not improved; failed/nonfinite optimization is an error.
    Returns temperature=1/slope and bias in the original tensor precision.
    """
    if (
        logits.ndim != 1
        or labels.shape != logits.shape
        or weights.shape != logits.shape
        or not logits.numel()
        or not bool(
            torch.isfinite(logits).all()
            & torch.isfinite(labels).all()
            & torch.isfinite(weights).all()
        )
        or not bool(((labels == 0) | (labels == 1)).all())
        or not bool((weights > 0).all())
        or not math.isfinite(regularization)
        or regularization <= 0
    ):
        raise ValueError("invalid detached Platt calibration observations")
    x, y = logits.detach().double(), labels.detach().double()
    w = weights.detach().double() / weights.detach().double().sum()
    with torch.enable_grad():
        parameters = torch.zeros(2, device=x.device, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([parameters], max_iter=80, line_search_fn="strong_wolfe")

        def objective() -> torch.Tensor:
            log_slope, intercept = parameters[0].clamp(-2, 2), parameters[1].clamp(-4, 4)
            log_loss = (
                F.binary_cross_entropy_with_logits(
                    x * log_slope.exp() + intercept, y, reduction="none"
                )
                * w
            ).sum()
            return log_loss + regularization * (log_slope.square() + intercept.square())

        baseline = float(objective().detach())

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = objective()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite risk calibration objective")
            loss.backward()
            return loss

        optimizer.step(closure)
        score = float(objective().detach())
        if not math.isfinite(score):
            raise ValueError("nonfinite risk calibration result")
        if score >= baseline:
            return 1.0, 0.0
        temperature = parameters[0].detach().clamp(-2, 2).neg().exp().to(logits.dtype)
        bias = parameters[1].detach().clamp(-4, 4).to(logits.dtype)
        return float(temperature), float(bias)
