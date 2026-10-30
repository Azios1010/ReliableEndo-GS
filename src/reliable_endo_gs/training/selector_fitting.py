"""Stationary TRAIN outcomes, independent head stopping and held-out calibration."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.config.selector_refit import SelectorRefitConfig
from reliable_endo_gs.contracts.interventions import RISK_EVENT_NAMES
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.calibrated_selector import CalibratedSelector
from reliable_endo_gs.training.outcome_calibration import fit_platt_risk
from reliable_endo_gs.training.selector_optimization import (
    make_selector_balancer,
    selector_objective,
)
from reliable_endo_gs.training.selector_outcomes import SelectorOutcomes


def joint_checkpoint_improves(
    components: Mapping[str, float],
    accepted_floors: Mapping[str, float],
    joint_score: float,
    best_joint_score: float,
    tolerance: float,
) -> bool:
    """Whole-head selection: improve joint VAL objective without masked regression.

    Floors are minima of accepted snapshots, not all rejected candidates, and
    prevent repeated small regressions from drifting away from an accepted head.
    Tolerance is the existing declared refit min_delta in normalized loss units.
    """
    if (
        not components
        or set(components) != set(accepted_floors)
        or not all(
            math.isfinite(v)
            for v in (
                *components.values(),
                *accepted_floors.values(),
                joint_score,
                best_joint_score,
                tolerance,
            )
        )
        or tolerance < 0
    ):
        raise ValueError("invalid joint checkpoint comparison")
    return joint_score < best_joint_score - tolerance and all(
        value <= accepted_floors[key] + tolerance for key, value in components.items()
    )


def keyframe_weights(table: SelectorOutcomes, device: torch.device) -> torch.Tensor:
    """Equal keyframe mass; safe/unsafe class odds are never reweighted."""
    keys = [s.rsplit("/", 1)[0] for s in table.sample_ids]
    counts = Counter(keys)
    return torch.tensor([1 / counts[k] / len(counts) for k in keys], device=device)


def validation_guard_components(
    components: Mapping[str, float], config: InterventionConfig, policy: SelectorRefitConfig
) -> dict[str, float]:
    """Guard comparable action objectives while retaining risk/ranking protection.

    These are normalized fitting losses, not physical utility or safety labels.
    The opt-in grouped mode permits global/local regression tradeoffs within an
    action using the declared objective weights. Neither action's total, either
    action's harm BCE, nor cross-action ranking may drift above accepted floors.
    Historical component-minimum selection remains the default.
    """
    if policy.validation_guard_mode == "component_minima":
        return dict(components)
    if config.selector_balance_mode != "per_action":
        raise ValueError("action-objective guards require per-action selector balance")
    return {
        **{
            f"{action}_objective": components[f"{action}_global"]
            + config.local_auxiliary_weight * components[f"{action}_local"]
            + config.harm_weight * components[f"{action}_harm"]
            + (
                config.gain_probability_weight * components[f"{action}_gain_probability"]
                if config.selector_gain_mode == "signed_expectation"
                else 0.0
            )
            for action in ("stereo", "gaussian")
        },
        "stereo_harm": components["stereo_harm"],
        "gaussian_harm": components["gaussian_harm"],
        "ranking": components["ranking"],
        **(
            {
                f"{action}_gain_probability": components[f"{action}_gain_probability"]
                for action in ("stereo", "gaussian")
            }
            if config.selector_gain_mode == "signed_expectation"
            else {}
        ),
    }


def fit_selector_heads(
    fit: SelectorOutcomes,
    validation: SelectorOutcomes,
    config: InterventionConfig,
    policy: SelectorRefitConfig,
    width: int,
    device: torch.device,
) -> tuple[BalancedSelector, dict[str, Any]]:
    """Only fit data estimates normalization; validation selects each encoder."""
    if (
        policy.validation_guard_mode == "action_objectives"
        and config.selector_balance_mode != "per_action"
    ):
        raise ValueError("action-objective guards require per-action selector balance")
    for table in (fit, validation):
        if table.metadata.get("collection_role") != "train":
            raise ValueError("selector fitting is TRAIN-only")
    if set(s.rsplit("/", 1)[0] for s in fit.sample_ids) & set(
        s.rsplit("/", 1)[0] for s in validation.sample_ids
    ):
        raise ValueError("fit/validation keyframes overlap")
    torch.manual_seed(policy.resolved_fit_seed)
    fit_ids = tuple(dict.fromkeys(fit.sample_ids))
    head_config = replace(
        config,
        normalization_frames=len(fit_ids),
        normalization_rows=max(config.normalization_rows, len(fit.sample_ids)),
    )
    head = BalancedSelector(width, head_config).to(device)
    x, y, local, h = (
        t.to(device) for t in (fit.features, fit.global_gain, fit.local_gain, fit.harms)
    )
    positions: dict[str, list[int]] = defaultdict(list)
    for i, sid in enumerate(fit.sample_ids):
        positions[sid].append(i)
    for sid in fit_ids:
        index = torch.tensor(positions[sid], device=device, dtype=torch.long)
        head.normalization.observe(
            x[index], y[index], local[index], sample_id=sid, role="final_train"
        )
    head.validate_state()
    if not bool(head.normalization.fitted):
        raise ValueError("fit role lacks enough known regions for normalization")
    weight = keyframe_weights(fit, device)
    val_weight = keyframe_weights(validation, device)
    vx, vy, vlocal, vh = (
        t.to(device)
        for t in (
            validation.features,
            validation.global_gain,
            validation.local_gain,
            validation.harms,
        )
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=policy.learning_rate, weight_decay=policy.weight_decay
    )
    independent = config.selector_balance_mode == "per_action"
    balancer = make_selector_balancer(head_config).to(device) if independent else None
    joint_state: dict[str, torch.Tensor] | None = None
    joint_floors: dict[str, float] = {}
    joint_best = float("inf")
    joint_stale = 0
    best = {"gain": float("inf"), "risk": float("inf")}
    states = {}
    steps = {}
    stale = {"gain": 0, "risk": 0}
    logs = []
    for step in range(1, policy.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        pending = None
        if balancer is not None:
            loss, pending, _ = selector_objective(
                head,
                balancer,
                x,
                y,
                local,
                h,
                row_weights=weight,
                risk_components=None
                if fit.risk_components is None
                else fit.risk_components.to(device),
            )
        else:
            losses, _ = head.objective(
                x,
                y,
                local,
                h,
                row_weights=weight,
                risk_components=None
                if fit.risk_components is None
                else fit.risk_components.to(device),
            )
            loss = (
                losses["global"]
                + config.local_auxiliary_weight * losses["local"]
                + config.ranking_weight * losses["ranking"]
                + config.harm_weight * losses["harm"]
            )
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite stationary selector loss")
        loss.backward()
        for action_network in head.optimization_networks():
            torch.nn.utils.clip_grad_norm_(action_network.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if balancer is not None and pending is not None:
            balancer.commit(pending)
        if step == 1 or step % policy.validate_every == 0 or step == policy.max_steps:
            with torch.no_grad():
                vl, validation_diagnostics = head.objective(
                    vx,
                    vy,
                    vlocal,
                    vh,
                    row_weights=val_weight,
                    risk_components=None
                    if validation.risk_components is None
                    else validation.risk_components.to(device),
                )
            scores = {
                "gain": float(
                    vl["global"]
                    + config.local_auxiliary_weight * vl["local"]
                    + config.ranking_weight * vl["ranking"]
                ),
                "risk": float(vl["harm"]),
            }
            if config.selector_gain_mode == "signed_expectation":
                scores["gain"] += config.gain_probability_weight * float(vl["gain_probability"])
            if not all(math.isfinite(v) for v in scores.values()):
                raise ValueError("nonfinite selector validation loss")
            components = {
                key: float(vl[key])
                for key in (
                    "stereo_global",
                    "stereo_local",
                    "stereo_harm",
                    "gaussian_global",
                    "gaussian_local",
                    "gaussian_harm",
                    "ranking",
                )
            }
            if config.selector_gain_mode == "signed_expectation":
                components.update(
                    {
                        f"{action}_gain_probability": float(vl[f"{action}_gain_probability"])
                        for action in ("stereo", "gaussian")
                    }
                )
            accepted = False
            guarded = validation_guard_components(components, config, policy)
            if independent:
                joint_score = scores["gain"] + config.harm_weight * scores["risk"]
                accepted = joint_state is None or joint_checkpoint_improves(
                    guarded, joint_floors, joint_score, joint_best, policy.min_delta
                )
                if accepted:
                    joint_state = copy.deepcopy(head.state_dict())
                    joint_floors = {
                        key: min(joint_floors.get(key, value), value)
                        for key, value in guarded.items()
                    }
                    joint_best, joint_stale = joint_score, 0
                    best, steps = scores.copy(), {"gain": step, "risk": step}
                elif step >= policy.min_steps:
                    joint_stale += 1
            else:
                for name, network in (("gain", head.gain_network), ("risk", head.harm_network)):
                    if scores[name] < best[name] - policy.min_delta:
                        best[name] = scores[name]
                        states[name] = copy.deepcopy(network.state_dict())
                        steps[name] = step
                        stale[name] = 0
                    elif step >= policy.min_steps:
                        stale[name] += 1
            logs.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "validation_gain": scores["gain"],
                    "validation_risk": scores["risk"],
                    "validation_components": components,
                    "validation_guard_components": guarded,
                    "validation_action_metrics": {
                        key: value
                        for key, value in validation_diagnostics.items()
                        if key.startswith(("selector_stereo_", "selector_gaussian_"))
                    },
                    "joint_checkpoint_accepted": accepted if independent else None,
                }
            )
            if step == 1 or step % 100 == 0 or step == policy.max_steps:
                print(
                    json.dumps(
                        {"event": "selector_refit_head_progress", **logs[-1], "best_steps": steps}
                    ),
                    flush=True,
                )
            stopped = (
                joint_stale >= policy.patience
                if independent
                else all(n >= policy.patience for n in stale.values())
            )
            if step >= policy.min_steps and stopped:
                break
    if independent:
        assert joint_state is not None
        head.load_state_dict(joint_state, strict=True)
    else:
        head.gain_network.load_state_dict(states["gain"])
        head.harm_network.load_state_dict(states["risk"])
    return head.eval().requires_grad_(False), {
        "selected_steps": steps,
        "validation_scores": best,
        "completed_steps": step,
        "checkpoint_selection_protocol": (
            "whole_head_action_objectives_risk_and_ranking.v1"
            if policy.validation_guard_mode == "action_objectives"
            else "whole_head_per_action_guard_and_joint_ranking.v1"
        )
        if independent
        else "historical_independent_gain_risk",
        "validation_guard_mode": policy.validation_guard_mode,
        "accepted_component_floors": joint_floors if independent else None,
        "logs": logs,
        "normalization_fit_sample_ids": list(fit_ids),
        "development_used_for_fitting": False,
        "label_observations": {
            "fit": fit.observation_summary(),
            "validation": validation.observation_summary(),
        },
    }


def fit_gain_affine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    regularization: float,
) -> tuple[float, float]:
    """Monotone ridge correction in utility units, bounded slope [0,2]."""
    if (
        prediction.ndim != 1
        or target.shape != prediction.shape
        or weight.shape != prediction.shape
        or not prediction.numel()
        or scale.numel() != 1
        or not math.isfinite(regularization)
        or regularization <= 0
        or not bool(
            torch.isfinite(prediction).all()
            & torch.isfinite(target).all()
            & torch.isfinite(weight).all()
            & (weight > 0).all()
            & torch.isfinite(scale).all()
            & (scale > 0).all()
        )
    ):
        raise ValueError(
            "affine gain calibration requires finite aligned vectors and positive scale/weights"
        )
    x = prediction.detach().double() / scale
    y = target.detach().double() / scale
    w = weight.double() / weight.sum()
    design = torch.stack((x, torch.ones_like(x)), 1)
    identity = design.new_tensor([1.0, 0.0])
    matrix = design.T @ (w[:, None] * design) + regularization * torch.eye(
        2, device=x.device, dtype=x.dtype
    )
    rhs = design.T @ (w * y) + regularization * identity
    parameters = torch.linalg.solve(matrix, rhs)
    a = parameters[0].clamp(0, 2)
    b = parameters[1].clamp(-1, 1)

    def score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (w * (a * x + b - y).square()).sum() + regularization * (
            (a - 1).square() + b.square()
        )

    if score(a, b) >= score(identity[0], identity[1]):
        return 1.0, 0.0
    return float(a), float(b * scale)


def calibrate_selector(
    model: CalibratedSelector,
    table: SelectorOutcomes,
    policy: SelectorRefitConfig,
    *,
    excluded_keyframes: set[str],
) -> dict[str, Any]:
    """Separate TRAIN calibration; signed expectation and each physical risk event.

    Gain affine calibration acts on E[utility], preserving raw sign/magnitude
    diagnostics. Each event needs independent safe/harm keyframe support; only
    supported any-event calibration permits action selection.
    """
    if table.metadata.get("collection_role") != "train":
        raise ValueError("calibration requires TRAIN outcomes")
    if not excluded_keyframes or excluded_keyframes & {
        s.rsplit("/", 1)[0] for s in table.sample_ids
    }:
        raise ValueError("calibration must be disjoint from fit/validation keyframes")
    model.calibration_available.zero_()
    model.gain_slope.fill_(1)
    model.gain_bias.zero_()
    model.risk_temperature.fill_(1)
    model.risk_bias.zero_()
    grouped = model.config.selector_risk_mode == "local_global"
    if grouped:
        model.event_temperature.fill_(1)
        model.event_bias.zero_()
        model.event_calibration_available.zero_()
    device = next(model.parameters()).device
    x, y, h = (t.to(device) for t in (table.features, table.global_gain, table.harms))
    weight = keyframe_weights(table, device)
    with torch.no_grad():
        gain, _, logits, _ = model.head.outputs_with_logits(x)
    if grouped:
        components = table.risk_components
        if (
            components is None
            or components.shape != (*h.shape, 2)
            or not bool(torch.isfinite(components).all())
            or not bool(((components == 0) | (components == 1)).all())
            or not torch.equal(components.bool().any(-1).to(h), h)
        ):
            raise ValueError(
                "event calibration requires local/global labels consistent with any harm"
            )
        labels = torch.cat((components.to(device), h[..., None]), -1)
    else:
        labels, logits = h[..., None], logits[..., None]
    report = {}
    for ai, name in ((1, "STEREO"), (2, "GAUSSIAN")):
        event_report = {}
        for j, event in enumerate(RISK_EVENT_NAMES if grouped else (RISK_EVENT_NAMES[-1],)):
            target, prediction = labels[:, ai, j], logits[:, ai - 1, j]
            groups = {
                label: {
                    s.rsplit("/", 1)[0]
                    for i, s in enumerate(table.sample_ids)
                    if int(target[i]) == label
                }
                for label in (0, 1)
            }
            supported = (
                len(table.sample_ids) >= policy.calibration_min_observations
                and min(map(len, groups.values())) >= policy.calibration_min_class_groups
            )
            row: dict[str, Any] = {
                "observations": len(table.sample_ids),
                "safe_keyframes": len(groups[0]),
                "unsafe_keyframes": len(groups[1]),
                "supported": supported,
                "output": "platt_probability" if supported else "raw_identity_transform",
            }
            if supported:
                t, bias = fit_platt_risk(
                    prediction, target, weight, regularization=policy.calibration_regularization
                )
                if grouped and j < 2:
                    model.event_temperature[ai - 1, j] = t
                    model.event_bias[ai - 1, j] = bias
                    model.event_calibration_available[ai - 1, j] = True
                else:
                    model.risk_temperature[ai - 1] = t
                    model.risk_bias[ai - 1] = bias
                    model.calibration_available[ai - 1] = True
                row.update(
                    risk_temperature=t,
                    risk_bias=bias,
                    raw_brier=float(((prediction.sigmoid() - target).square() * weight).sum()),
                    calibrated_brier=float(
                        (((prediction / t + bias).sigmoid() - target).square() * weight).sum()
                    ),
                )
            event_report[event] = row
        row = dict(event_report[RISK_EVENT_NAMES[-1]])
        if bool(model.calibration_available[ai - 1]):
            a, b = fit_gain_affine(
                gain[:, ai],
                y[:, ai],
                weight,
                model.head.normalization.gain_scales[0],
                policy.gain_regularization,
            )
            model.gain_slope[ai - 1] = a
            model.gain_bias[ai - 1] = b
            row.update(gain_slope=a, gain_bias=b)
        if grouped:
            row["events"] = event_report
        report[name] = row
    model.validate_state()
    return {
        "actions": report,
        "cutoffs_tuned": False,
        "development_used_for_calibration": False,
        "label_observations": table.observation_summary(),
        "safety_guarantee": False,
        "prediction_metadata": model.prediction_metadata(),
        "event_calibration": "independent_marginals" if grouped else "any_event_only",
        "joint_event_consistency_guaranteed": False,
    }
