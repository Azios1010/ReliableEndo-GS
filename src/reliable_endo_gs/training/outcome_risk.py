"""Offline risk discrimination from canonical safety labels; no inference inputs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from reliable_endo_gs.config.outcome_training import OutcomeTrainingConfig
from reliable_endo_gs.data.intervention_table import OutcomeTable
from reliable_endo_gs.evaluation.outcome_selection import (
    component_constraint_targets,
    constraint_targets,
)
from reliable_endo_gs.routing.outcome_predictor import OutcomePredictor
from reliable_endo_gs.routing.tradeoff import TradeoffConfig
from reliable_endo_gs.training.outcome_validation import validate_training_views, validation_score


def risk_ranking_loss(
    logits: torch.Tensor, labels: torch.Tensor, observed: torch.Tensor
) -> torch.Tensor:
    """Mean per-action softplus(safe logit - unsafe logit), shapes [R,A].

    Compare the two classes explicitly so abundant unsafe outcomes cannot hide
    failure to discriminate the minority safe class. Joint BCE remains the
    unweighted probability objective; this auxiliary term changes score fitting,
    so probabilities require independent TRAIN-group calibration. No preferred
    action frequency or utility/threshold modification is encoded here.
    Only known outcomes participate; a one-class action contributes no pair.
    """
    if (
        logits.ndim != 2
        or labels.shape != logits.shape
        or observed.shape != logits.shape
        or observed.dtype != torch.bool
        or not bool(torch.isfinite(logits).all() & torch.isfinite(labels).all())
        or not bool(((labels == 0) | (labels == 1)).all())
    ):
        raise ValueError("risk ranking requires finite logits/binary labels/bool mask [R,A]")
    terms = []
    for action in range(logits.shape[1]):
        safe = logits[:, action][observed[:, action] & labels[:, action].eq(0)]
        unsafe = logits[:, action][observed[:, action] & labels[:, action].eq(1)]
        if safe.numel() and unsafe.numel():
            terms.append(F.softplus(safe[:, None] - unsafe[None, :]).mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def refit_conditional_risk(
    model: OutcomePredictor,
    fit: OutcomeTable,
    validation: OutcomeTable,
    config: TradeoffConfig,
    policy: OutcomeTrainingConfig,
) -> tuple[list[dict[str, float]], dict[str, int | float | bool]]:
    """Fit only risk parameters against one fixed selected gain model.

    Conditional margins are prediction-derived at training and inference, never
    replaced by known outcomes. Inference mode fixes gain dropout during this
    bounded second stage. Validation groups select risk; calibration stays unseen.
    Normalization/gain tensors are checked unchanged before return.
    """
    validate_training_views(fit, validation)
    if model.risk_context_config != config or policy.risk_feature_mode != "predicted_margins":
        raise ValueError("conditional risk refit requires matching prediction context policy")
    device = model.metric_scale.device
    gain_state = {k: v.detach().clone() for k, v in model.change_network.state_dict().items()}
    normalization = [
        v.clone() for v in (model.feature_center, model.feature_scale, model.metric_scale)
    ]
    model.change_network.requires_grad_(False).eval()
    model.risk_network.requires_grad_(True).train()
    torch.manual_seed(config.seed)
    for layer in model.risk_network:
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()
    last = model.risk_network[-1]
    assert isinstance(last, nn.Linear)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    optimizer = torch.optim.AdamW(
        model.risk_network.parameters(), lr=config.learning_rate, weight_decay=policy.weight_decay
    )
    features = fit.features.to(device)
    observed = fit.observed.to(device)
    labels = constraint_targets(fit, config).to(device)
    components = component_constraint_targets(fit, config).to(device)
    mask = observed[..., None] & model.structural.bool()[None]
    best_state = None
    best_score, best_step, stale = float("inf"), 0, 0
    logs = []
    stop_reason = 0
    for step in range(1, policy.max_steps + 1):
        model.change_network.eval()
        model.risk_network.train()
        optimizer.zero_grad(set_to_none=True)
        _, logits, component_logits = model.forward_details(features)
        assert component_logits is not None
        joint = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")[
            observed
        ].mean()
        auxiliary = F.binary_cross_entropy_with_logits(
            component_logits, components, reduction="none"
        )[mask].mean()
        ranking = risk_ranking_loss(logits, labels, observed)
        loss = (
            joint + config.component_risk_weight * auxiliary + policy.risk_ranking_weight * ranking
        )
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite conditional risk refit loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.risk_network.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 10 == 0:
            logs.append(
                {
                    "epoch": float(step),
                    "loss": float(loss.detach()),
                    "risk": float(joint.detach()),
                    "risk_ranking": float(ranking.detach()),
                    "risk_refit": 1.0,
                }
            )
        if step >= policy.min_steps and (
            step % policy.validate_every == 0 or step == policy.max_steps
        ):
            score = validation_score(model, validation, config)["risk_log_loss"]
            logs.append(
                {"epoch": float(step), "validation_risk_log_loss": score, "risk_refit": 1.0}
            )
            if best_state is None or score < best_score - policy.min_delta:
                best_state = {
                    k: v.detach().clone() for k, v in model.risk_network.state_dict().items()
                }
                best_score, best_step, stale = score, step, 0
            else:
                stale += 1
            if stale >= policy.patience:
                stop_reason = 1
                break
    assert best_state is not None
    model.risk_network.load_state_dict(best_state, strict=True)
    if any(
        not torch.equal(v, gain_state[k]) for k, v in model.change_network.state_dict().items()
    ) or any(
        not torch.equal(v, old)
        for v, old in zip(
            (model.feature_center, model.feature_scale, model.metric_scale),
            normalization,
            strict=True,
        )
    ):
        raise ValueError("risk refit mutated selected gain/normalization")
    return logs, {
        "best_step": best_step,
        "completed_steps": step,
        "best_score": best_score,
        "validation_patience": bool(stop_reason),
        "best_state_restored": True,
        "gain_unchanged": True,
        "seed": config.seed,
    }
