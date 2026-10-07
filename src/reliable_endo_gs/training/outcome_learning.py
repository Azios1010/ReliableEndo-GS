"""Bounded frozen-intervention predictor pilot and matched-candidate evaluation.

GT metric outcomes are offline labels only. Train/development tables remain
immutable and disjoint by frame ID; shared keyframes are disclosed. This pilot
neither certifies Gate II nor changes the joint model or default router.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from reliable_endo_gs.config.outcome_training import (
    OutcomeTrainingConfig,
    load_outcome_training_config,
)
from reliable_endo_gs.data.intervention_table import OutcomeTable as OutcomeTable
from reliable_endo_gs.data.intervention_table import load_outcome_table as _load_outcome_table
from reliable_endo_gs.evaluation.outcome_selection import (
    bootstrap_difference,
    component_constraint_targets,
    constraint_targets,
)
from reliable_endo_gs.evaluation.outcome_selection import (
    evaluate_outcome_predictions as evaluate_outcome_predictions,
)
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES
from reliable_endo_gs.routing.outcome_predictor import OutcomePredictor
from reliable_endo_gs.routing.tradeoff import (
    ACTION_NAMES,
    COMPONENT_NAMES,
    TradeoffConfig,
    constraint_margins_tensor,
    feasible_changes,
    load_tradeoff_config,
    outcome_utility_tensor,
)
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.runtime.identity import source_sha256
from reliable_endo_gs.training.outcome_calibration import (
    calibrate_joint_risk,
    split_training_calibration,
    split_training_validation_calibration,
)
from reliable_endo_gs.training.outcome_risk import refit_conditional_risk, risk_ranking_loss
from reliable_endo_gs.training.outcome_validation import validate_training_views, validation_score
from reliable_endo_gs.utils.io import write_json_atomic


def load_outcome_table(directory: Path, expected_role: str) -> OutcomeTable:
    """The experiment owns its feature schema; the data reader stays independent."""
    return _load_outcome_table(directory, expected_role, feature_names=BALANCED_FEATURE_NAMES)


def fit_outcome_predictor(
    table: OutcomeTable,
    config: TradeoffConfig,
    *,
    linear: bool = False,
    device: torch.device | None = None,
) -> tuple[OutcomePredictor, list[dict[str, float]]]:
    model, logs, _ = _fit_outcome_predictor(table, config, linear=linear, device=device)
    return model, logs


def fit_regularized_outcome_predictor(
    table: OutcomeTable,
    validation: OutcomeTable,
    config: TradeoffConfig,
    training_policy: OutcomeTrainingConfig,
    *,
    linear: bool = False,
    device: torch.device | None = None,
) -> tuple[OutcomePredictor, list[dict[str, float]], dict[str, Any]]:
    """Select one checkpoint using disjoint TRAIN keyframes, then restore it."""
    if not config.schema.endswith(".v2"):
        raise ValueError("regularized grouped fitting requires v2 metric policy")
    validate_training_views(table, validation)
    return _fit_outcome_predictor(
        table,
        config,
        linear=linear,
        device=device,
        validation=validation,
        training_policy=training_policy,
    )


def _fit_outcome_predictor(
    table: OutcomeTable,
    config: TradeoffConfig,
    *,
    linear: bool,
    device: torch.device | None,
    validation: OutcomeTable | None = None,
    training_policy: OutcomeTrainingConfig | None = None,
) -> tuple[OutcomePredictor, list[dict[str, float]], dict[str, Any]]:
    """Full-batch bounded pilot. Only a table explicitly marked train may fit."""
    device = torch.device("cpu") if device is None else device
    if table.metadata["collection_role"] != "train":
        raise ValueError("development outcomes may not fit predictor/normalization")
    if any(not bool(table.observed[:, ai].any()) for ai in range(len(ACTION_NAMES))):
        raise ValueError("training lacks observed outcomes for an action")
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        model = OutcomePredictor(
            config.width,
            linear=linear,
            component_risk=config.schema.endswith(".v2"),
            dropout=training_policy.dropout if training_policy is not None else 0.0,
            risk_context_config=config
            if training_policy is not None
            and training_policy.risk_feature_mode == "predicted_margins"
            else None,
        ).to(device)
        x, y, mask = table.features.to(device), table.changes.to(device), table.observed.to(device)
        model.fit_normalization(x, y, mask)
        target = y / model.metric_scale
        risk_target = constraint_targets(table, config).to(device)
        positives = (risk_target * mask).sum(0)
        negatives = ((1 - risk_target) * mask).sum(0)
        positive_weight = torch.where(positives > 0, negatives / positives.clamp_min(1), 1).clamp(
            0.25, 4
        )
        if model.component_risk:
            # Proper probabilities: v2 uses unweighted BCE and train-group
            # calibration, not class-weight-shifted odds for an aggregate event.
            positive_weight = torch.ones_like(positive_weight)
        model.risk_positive_weight.copy_(positive_weight)
        optimizer = (
            torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            if training_policy is None
            else torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=training_policy.weight_decay,
            )
        )
        scales = model.metric_scale.tolist()
        actual_score = outcome_utility_tensor(y, scales, config)
        component_risk_target = component_constraint_targets(table, config).to(device)
        component_mask = mask[..., None] & model.structural.bool()[None]
        safe = mask & ~risk_target.bool()
        # STOP=0 participates. Only safe observed repairs contribute ranking.
        rank_target = torch.cat((torch.zeros(len(x), 1, device=device), actual_score), -1)
        rank_known = torch.cat((torch.ones(len(x), 1, dtype=torch.bool, device=device), safe), -1)
        logs = []
        maximum = config.epochs if training_policy is None else training_policy.max_steps
        best_state = None
        best_score = float("inf")
        best_step = 0
        stale = 0
        encoder_states: dict[str, dict[str, torch.Tensor]] = {}
        encoder_scores = {"changes": float("inf"), "risk": float("inf")}
        encoder_steps = {"changes": 0, "risk": 0}
        encoder_stale = {"changes": 0, "risk": 0}
        independent = training_policy is not None and training_policy.independent_encoder_stopping
        stop_reason = "max_steps"
        for epoch in range(maximum):
            optimizer.zero_grad(set_to_none=True)
            prediction, logits, component_logits = model.forward_details(x)
            regression = F.smooth_l1_loss(prediction, target, reduction="none")[mask].mean()
            risk = F.binary_cross_entropy_with_logits(
                logits, risk_target, pos_weight=positive_weight, reduction="none"
            )[mask].mean()
            component_risk = (
                F.binary_cross_entropy_with_logits(
                    component_logits, component_risk_target, reduction="none"
                )[component_mask].mean()
                if component_logits is not None
                else risk * 0
            )
            constraint_margin = (
                F.binary_cross_entropy_with_logits(
                    -constraint_margins_tensor(prediction * model.metric_scale, scales, config),
                    component_risk_target,
                    reduction="none",
                )[component_mask].mean()
                if model.component_risk
                else regression * 0
            )
            score = outcome_utility_tensor(prediction * model.metric_scale, scales, config)
            score = torch.cat((torch.zeros(len(x), 1, device=device), score), -1)
            terms = []
            for a in range(4):
                for b in range(a + 1, 4):
                    margin = rank_target[:, a] - rank_target[:, b]
                    informative = (
                        rank_known[:, a]
                        & rank_known[:, b]
                        & (margin.abs() > config.ranking_epsilon)
                    )
                    if bool(informative.any()):
                        # Relative observed margins, no fixed raw gain threshold.
                        terms.append(
                            (
                                F.softplus(-margin.sign() * (score[:, a] - score[:, b]))
                                * margin.abs().clamp(max=5)
                            )[informative].mean()
                        )
            ranking = torch.stack(terms).mean() if terms else prediction.sum() * 0
            loss = regression + risk + config.ranking_weight * ranking
            risk_ranking = (
                risk_ranking_loss(logits, risk_target, mask)
                if training_policy is not None and training_policy.risk_ranking_weight > 0
                else risk * 0
            )
            if training_policy is not None:
                loss = loss + training_policy.risk_ranking_weight * risk_ranking
            if model.component_risk:
                loss = loss + config.component_risk_weight * component_risk
                loss = loss + config.constraint_margin_weight * constraint_margin
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite outcome predictor objective")
            loss.backward()
            if any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
            ):
                raise ValueError("nonfinite predictor gradients")
            # Independent clipping prevents risk gradients from scaling the
            # change encoder's update through a shared total-norm multiplier.
            for encoder in (model.change_network, model.risk_network):
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == maximum:
                logs.append(
                    {
                        "epoch": float(epoch + 1),
                        "loss": float(loss.detach()),
                        "components": float(regression.detach()),
                        "risk": float(risk.detach()),
                        "ranking": float(ranking.detach()),
                        "component_risk": float(component_risk.detach()),
                        "constraint_margin": float(constraint_margin.detach()),
                        "risk_ranking": float(risk_ranking.detach()),
                    }
                )
            step = epoch + 1
            if (
                training_policy is not None
                and step >= training_policy.min_steps
                and (step % training_policy.validate_every == 0 or step == maximum)
            ):
                assert validation is not None
                scored = validation_score(model, validation, config)
                entry = {
                    "epoch": float(step),
                    "validation_score": scored["score"],
                    "validation_regression": scored["regression"],
                    "validation_risk_log_loss": scored["risk_log_loss"],
                }
                if logs and logs[-1]["epoch"] == float(step):
                    logs[-1].update(entry)
                else:
                    logs.append(entry)
                if best_state is None or scored["score"] < best_score - training_policy.min_delta:
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    best_score, best_step, stale = scored["score"], step, 0
                else:
                    stale += 1
                if independent:
                    for name, encoder, metric in (
                        ("changes", model.change_network, "regression"),
                        ("risk", model.risk_network, "risk_log_loss"),
                    ):
                        value = scored[metric]
                        if (
                            name not in encoder_states
                            or value < encoder_scores[name] - training_policy.min_delta
                        ):
                            encoder_states[name] = {
                                k: v.detach().clone() for k, v in encoder.state_dict().items()
                            }
                            encoder_scores[name], encoder_steps[name], encoder_stale[name] = (
                                value,
                                step,
                                0,
                            )
                        else:
                            encoder_stale[name] += 1
                done = (
                    all(s >= training_policy.patience for s in encoder_stale.values())
                    if independent
                    else stale >= training_policy.patience
                )
                if done:
                    stop_reason = "validation_patience"
                    break
        selection: dict[str, Any] = {"enabled": False}
        if training_policy is not None:
            assert best_state is not None and validation is not None
            if independent:
                model.change_network.load_state_dict(encoder_states["changes"], strict=True)
                model.risk_network.load_state_dict(encoder_states["risk"], strict=True)
                best_score = validation_score(model, validation, config)["score"]
                best_step = max(encoder_steps.values())
            else:
                model.load_state_dict(best_state, strict=True)
            selection = {
                "enabled": True,
                "best_step": best_step,
                "completed_steps": epoch + 1,
                "best_validation_score": best_score,
                "stop_reason": stop_reason,
                "best_state_restored": True,
                "development_used_for_selection": False,
                "validation_sample_ids": [r["sample_id"] for r in validation.records],
                "fit_sample_ids": [r["sample_id"] for r in table.records],
                "selection_metric": "independent_keyframe_regression_and_joint_log_loss"
                if independent
                else "keyframe_mean_normalized_regression_plus_joint_log_loss",
                "training_policy_hash": training_policy.config_hash,
            }
            if independent:
                selection["encoder_selection"] = {
                    name: {
                        "best_step": encoder_steps[name],
                        "best_score": encoder_scores[name],
                        "best_state_restored": True,
                    }
                    for name in encoder_steps
                }
            if training_policy.risk_feature_mode == "predicted_margins":
                risk_logs, refit = refit_conditional_risk(
                    model, table, validation, config, training_policy
                )
                logs.extend(risk_logs)
                selection["first_stage_completed_steps"] = epoch + 1
                selection["risk_refit"] = refit
                selection["completed_steps"] = epoch + 1 + int(refit["completed_steps"])
                selection["best_step"] = epoch + 1 + int(refit["best_step"])
                selection["best_validation_score"] = validation_score(model, validation, config)[
                    "score"
                ]
                selection["encoder_selection"]["risk"] = {
                    "best_step": selection["best_step"],
                    "best_score": refit["best_score"],
                    "best_state_restored": True,
                }
                selection["stop_reason"] = (
                    "validation_patience" if refit["validation_patience"] else "max_steps"
                )
        model.eval().requires_grad_(False)
    return model, logs, selection


def _constant_predictions(
    table: OutcomeTable, rows: int, config: TradeoffConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    means = torch.stack([table.changes[:, a][table.observed[:, a]].mean(0) for a in range(3)])
    risk = constraint_targets(table, config)
    prevalence = torch.stack([risk[:, a][table.observed[:, a]].mean() for a in range(3)])
    return means[None].expand(rows, -1, -1), prevalence[None].expand(rows, -1)


def _state_hash(model: OutcomePredictor) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run_outcome_pilot(
    *,
    training_table: Path,
    development_table: Path,
    config_path: Path,
    output_dir: Path,
    device_name: str = "cpu",
    budgets_ms: tuple[float, ...] = (250.0, 500.0, 1000.0),
    relabel_policy: bool = False,
    training_policy_path: Path | None = None,
) -> dict[str, object]:
    """Fit constant/linear/MLP on training outcomes, then report development only.

    No test fold access, upstream update or automatic threshold/weight tuning.
    GPU mode measures tiny predictor timing on the same device as native tables.
    CPU mode is a functional/learning diagnostic, not a matched runtime benchmark.
    """
    repo = Path(__file__).resolve().parents[3]
    if output_dir.resolve() == repo or repo in output_dir.resolve().parents:
        raise ValueError("pilot outputs must remain outside repository")
    if output_dir.exists():
        raise FileExistsError("outcome pilot output exists")
    config = load_tradeoff_config(config_path)
    training_policy = (
        load_outcome_training_config(training_policy_path)
        if training_policy_path is not None
        else None
    )
    if training_policy is not None and not config.schema.endswith(".v2"):
        raise ValueError("grouped training policy requires v2 outcomes")
    train = load_outcome_table(training_table, "train")
    dev = load_outcome_table(development_table, "development")
    if set(train.metadata["sample_ids"]) & set(dev.metadata["sample_ids"]):
        raise ValueError("train/development frame overlap")
    for key in (
        "source_sha256",
        "training_manifest_hash",
        "development_manifest_hash",
        "role_hash",
        "candidate_config_hash",
        "renderer_identity",
        "gpu_uuid",
        "action_names",
        "component_names",
        "feature_names",
        "tradeoff_config_hash",
        "imported_weights",
        "timing_repeats",
    ):
        if train.metadata.get(key) != dev.metadata.get(key):
            raise ValueError(f"incompatible outcome tables: {key}")
    if relabel_policy and not config.schema.endswith(".v2"):
        raise ValueError("explicit policy relabel requires the v2 outcome contract")
    if relabel_policy:
        for table in (train, dev):
            collected = TradeoffConfig(**table.metadata["tradeoff_config"])
            if collected.config_hash != table.metadata["tradeoff_config_hash"]:
                raise ValueError("collection policy provenance mismatch")
    elif train.metadata["tradeoff_config_hash"] != config.config_hash:
        raise ValueError("policy contract does not match collected tables")
    validation: OutcomeTable | None = None
    calibration_table: OutcomeTable | None
    partition: dict[str, Any] | None
    if training_policy is not None:
        fit, validation, calibration_table, partition = split_training_validation_calibration(
            train, config
        )
    else:
        fit, calibration_table, partition = (
            split_training_calibration(train, config)
            if config.schema.endswith(".v2")
            else (train, None, None)
        )
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("outcome pilot supports cpu/cuda only")
    if device.type == "cuda":
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or device.index not in (None, 0)
        ):
            raise ValueError("native outcome pilot needs one visible GPU")
        if (
            str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable"))
            != dev.metadata["gpu_uuid"]
        ):
            raise ValueError("predictor/table hardware identity mismatch")
    if (
        not budgets_ms
        or len(set(budgets_ms)) != len(budgets_ms)
        or any(not math.isfinite(b) or b <= 0 for b in budgets_ms)
    ):
        raise ValueError("need unique finite positive budgets")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    models = {}
    logs = {}
    timings = {}
    predictions = {}
    training_calibration = {}
    selections = {}
    for name in ("linear", "mlp"):
        if training_policy is not None:
            assert validation is not None
            model, log, selection = fit_regularized_outcome_predictor(
                fit, validation, config, training_policy, linear=name == "linear", device=device
            )
            selections[name] = selection
        else:
            model, log = fit_outcome_predictor(fit, config, linear=name == "linear", device=device)
        if calibration_table is not None:
            training_calibration[name] = calibrate_joint_risk(
                model, calibration_table, config, training_policy
            )
        models[name] = model
        logs[name] = log
        state_before = _state_hash(model)
        features = dev.features.to(device)
        # Warm predictor, then time each frame repeatedly on precomputed features.
        model.predict(features[: dev.slices[0][1]])
        frame_times = []
        for start, end in dev.slices:
            repeats = []
            for _ in range(5):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                pred, risk = model.predict(features[start:end])
                # Include the host materialization needed by this pilot allocator.
                pred.cpu().tolist()
                risk.cpu().tolist()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                repeats.append((time.perf_counter() - t0) * 1000)
            frame_times.append(statistics.median(repeats))
        predictions[name] = tuple(t.cpu() for t in model.predict(features))
        timings[name] = tuple(frame_times)
        if _state_hash(model) != state_before:
            raise ValueError("outcome evaluation mutated predictor")
        torch.save(
            {
                "schema": "metric_outcome_predictor.v2"
                if config.schema.endswith(".v2")
                else "metric_outcome_predictor.v1",
                "kind": name,
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "config_hash": config.config_hash,
                "feature_names": list(BALANCED_FEATURE_NAMES),
                "risk_feature_names": list(model.risk_feature_names),
                "component_names": list(COMPONENT_NAMES),
                "action_names": list(ACTION_NAMES),
                "training_table_sha256": train.metadata["outcome_sha256"],
                "frozen_action_identity": train.metadata["imported_weights"],
                "candidate_config_hash": train.metadata["candidate_config_hash"],
                "source_sha256": source_sha256(repo),
                "collection_config_hash": train.metadata["tradeoff_config_hash"],
                "policy_relabel_explicit": relabel_policy,
                "training_partition": partition,
                "training_risk_calibration": training_calibration.get(name),
                "training_policy": asdict(training_policy) if training_policy is not None else None,
                "training_policy_hash": training_policy.config_hash
                if training_policy is not None
                else None,
                "training_selection": selections.get(name),
            },
            output_dir / f"{name}.pt",
        )
    scales = models["mlp"].metric_scale.cpu().tolist()
    predictions["constant"] = _constant_predictions(fit, len(dev.features), config)
    costs = fit.costs.amax(0)
    summaries = {}
    policy_records = []
    for name in (
        "constant",
        "linear",
        "mlp",
        "oracle",
        *(f"fixed_{a}" for a in ACTION_NAMES),
        *(f"mlp_{a}_only" for a in ACTION_NAMES),
    ):
        source = name if name in predictions else "mlp"
        restriction = next((a for a in ACTION_NAMES if name == f"mlp_{a}_only"), None)
        summary, rows = evaluate_outcome_predictions(
            dev,
            *predictions[source],
            scales=scales,
            estimated_costs=costs,
            config=config,
            budgets_ms=budgets_ms,
            predictor_ms=timings.get(source, tuple(0.0 for _ in dev.records)),
            policy=name,
            action_restriction=restriction,
        )
        summaries[name] = summary
        policy_records.extend(rows)
    calibration = {}
    target = constraint_targets(dev, config)
    component_target = component_constraint_targets(dev, config)
    for name in ("constant", "linear", "mlp"):
        changes, risks = predictions[name]
        component_probabilities = (
            models[name].predict_component_risks(dev.features.to(device)).cpu()
            if name in models and models[name].component_risk
            else None
        )
        per_action = {}
        for ai, action in enumerate(ACTION_NAMES):
            mask = dev.observed[:, ai]
            risk_available = not (name in models and models[name].component_risk) or bool(
                models[name].risk_calibrated[ai]
            )
            p, y = risks[:, ai][mask], target[:, ai][mask]
            bins = []
            for b in range(10):
                selected = (p >= b / 10) & (p < ((b + 1) / 10) if b < 9 else p <= 1)
                if bool(selected.any()):
                    bins.append(
                        {
                            "count": int(selected.sum()),
                            "prediction": float(p[selected].mean()),
                            "observed_violation": float(y[selected].mean()),
                        }
                    )
            per_action[action] = {
                "observations": int(mask.sum()),
                "risk_probability_available": risk_available,
                "Brier": float((p - y).square().mean()) if len(p) and risk_available else None,
                "ECE_10_bins": sum(
                    row["count"] * abs(row["prediction"] - row["observed_violation"])
                    for row in bins
                )
                / len(p)
                if len(p) and risk_available
                else None,
                "bins": bins if risk_available else [],
                "false_safe_at_cutoff": int(
                    ((p <= config.maximum_predicted_violation_probability) & (y == 1)).sum()
                ),
                "false_unsafe_at_cutoff": int(
                    ((p > config.maximum_predicted_violation_probability) & (y == 0)).sum()
                )
                if risk_available
                else None,
                "safe_accepted_at_cutoff": int(
                    ((p <= config.maximum_predicted_violation_probability) & (y == 0)).sum()
                )
                if risk_available
                else None,
                "true_safe_observations": int((y == 0).sum()),
                "safe_recall_at_cutoff": float(
                    ((p <= config.maximum_predicted_violation_probability) & (y == 0)).sum()
                    / (y == 0).sum()
                )
                if risk_available and bool((y == 0).any())
                else None,
                "accepted_violation_rate": float(
                    y[p <= config.maximum_predicted_violation_probability].mean()
                )
                if risk_available
                and bool((p <= config.maximum_predicted_violation_probability).any())
                else None,
                "component_MAE_raw_units": {
                    key: float((changes[:, ai, j][mask] - dev.changes[:, ai, j][mask]).abs().mean())
                    if bool(mask.any())
                    else None
                    for j, key in enumerate(COMPONENT_NAMES)
                },
                "predicted_components_feasible": int(
                    sum(feasible_changes(row, config) for row in changes[:, ai][mask].tolist())
                ),
                "measured_components_feasible": int((y == 0).sum()),
                "component_risk_Brier": {
                    key: float(
                        (component_probabilities[:, ai, j][mask] - component_target[:, ai, j][mask])
                        .square()
                        .mean()
                    )
                    if bool(mask.any())
                    else None
                    for j, key in enumerate(COMPONENT_NAMES)
                }
                if component_probabilities is not None
                else None,
            }
        calibration[name] = per_action
    comparisons = {}
    for budget in budgets_ms:
        lookup = {
            (r["sample_id"], r["policy"]): r for r in policy_records if r["budget_ms"] == budget
        }
        for baseline in ("fixed_gaussian", "mlp_gaussian_only", "oracle"):
            diffs = []
            for record in dev.records:
                proposed = lookup[(record["sample_id"], "mlp")]
                reference = lookup[(record["sample_id"], baseline)]
                if proposed["safe_utility"] is not None and reference["safe_utility"] is not None:
                    diffs.append(
                        (
                            record["sample_id"].rsplit("/", 1)[0],
                            proposed["safe_utility"] - reference["safe_utility"],
                        )
                    )
            comparisons[f"{budget}:mlp_minus_{baseline}"] = bootstrap_difference(diffs, config)
    with (output_dir / "decisions.jsonl").open("x") as stream:
        for row in policy_records:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    write_json_atomic(output_dir / "training.json", logs)
    shared = set(s.rsplit("/", 1)[0] for s in train.metadata["sample_ids"]) & set(
        s.rsplit("/", 1)[0] for s in dev.metadata["sample_ids"]
    )
    result = {
        "status": "metric_outcome_pilot_complete",
        "Gate_II_certified": False,
        "upstream_training_executed": False,
        "selector_training_executed": True,
        "teacher_inference_executed": False,
        "config": asdict(config),
        "config_hash": config.config_hash,
        "collection_config_hash": train.metadata["tradeoff_config_hash"],
        "policy_relabel_explicit": relabel_policy,
        "training_partition": partition,
        "training_risk_calibration": training_calibration,
        "training_policy": asdict(training_policy) if training_policy is not None else None,
        "training_policy_hash": training_policy.config_hash
        if training_policy is not None
        else None,
        "training_selection": selections,
        "training_table_sha256": train.metadata["outcome_sha256"],
        "development_table_sha256": dev.metadata["outcome_sha256"],
        "frozen_action_identity": train.metadata["imported_weights"],
        "candidate_config_hash": train.metadata["candidate_config_hash"],
        "source_sha256": source_sha256(repo),
        "git_commit": get_git_commit(repo),
        "git_dirty": is_git_dirty(repo),
        "collector_source_sha256": train.metadata["source_sha256"],
        "table_gpu_uuid": train.metadata["gpu_uuid"],
        "training_manifest_hash": train.metadata["training_manifest_hash"],
        "development_manifest_hash": train.metadata["development_manifest_hash"],
        "roles_hash": train.metadata["role_hash"],
        "torch": str(torch.__version__),
        "device": str(device),
        "native_predictor_timing": device.type == "cuda",
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "training_frames": len(train.records),
        "development_frames": len(dev.records),
        "training_regions": len(train.features),
        "fit_frames": len(fit.records),
        "fit_regions": len(fit.features),
        "development_regions": len(dev.features),
        "training_unknown_metric_cells": int((train.eligible & ~train.observed).sum()),
        "development_unknown_metric_cells": int((dev.eligible & ~dev.observed).sum()),
        "shared_keyframes": len(shared),
        "train_fitted_metric_scales": dict(zip(COMPONENT_NAMES, scales, strict=True)),
        "train_fitted_action_cost_max_ms": dict(zip(ACTION_NAMES, costs.tolist(), strict=True)),
        "policies": summaries,
        "calibration": calibration,
        "group_bootstrap": comparisons,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Experimental frozen-action outcome prediction; not default deployment or joint resume.",
            "Predictive safety is not guaranteed; violation probabilities require held-out calibration.",
            "Geometry targets measure stereo triangulation, not native rendered depth.",
            "Shared-keyframe development is not independent sequence validation.",
            "Small single-seed pilot; no final-test access or automatic hyperparameter tuning.",
            "CPU timings cannot be combined with native GPU costs for runtime claims.",
            "Native table costs use few repeated action timings; base/features are single-pass.",
            "Oracle uses measured outcomes/costs as an unattainable upper bound.",
            "No repair action is executed by replay evaluation; costs are measured counterfactual replay.",
        ],
    }
    write_json_atomic(output_dir / "result.json", result)
    return result
