"""Balanced optimization on one detached, freshly measured TRAIN snapshot."""

from __future__ import annotations

import torch

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.training.gradient_balance import GradientBalance


def _balance_layout(
    config: InterventionConfig,
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    if config.selector_balance_mode == "per_action":
        return (
            ("stereo_global", "stereo_local", "gaussian_global", "gaussian_local"),
            ((0, 1), (2, 3)),
        )
    return ("global", "local", "ranking"), ((0, 1, 2),)


def make_selector_balancer(config: InterventionConfig) -> GradientBalance:
    """Versioned loss groups; ranking is fixed-weight in per-action mode."""
    names, groups = _balance_layout(config)
    return GradientBalance(names, config, groups=groups)


def selector_objective(
    selector: BalancedSelector,
    balancer: GradientBalance,
    features: torch.Tensor,
    global_targets: torch.Tensor,
    local_targets: torch.Tensor,
    harms: torch.Tensor,
    *,
    train_harm: bool = True,
    row_weights: torch.Tensor | None = None,
    risk_components: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Prepare [R,F] -> [R,3] supervision; commit EMA after optimizer success.

    Every input is detached. Repeated calls use the same current model/action
    snapshot; they cannot send selector gradients into those parameter owners.
    """
    losses, diagnostics = selector.objective(
        features.detach(),
        global_targets.detach(),
        local_targets.detach(),
        harms.detach(),
        row_weights=row_weights,
        risk_components=None if risk_components is None else risk_components.detach(),
    )
    names, groups = _balance_layout(selector.config)
    if balancer.names != names or balancer.groups != groups:
        raise ValueError("selector balancer does not match configured loss ownership")
    shared = selector.gain_shared_parameters()
    pending, balance_diagnostics = balancer.measure(
        tuple(losses[k] for k in balancer.names), shared
    )
    cfg = selector.config
    if cfg.selector_balance_mode == "per_action":
        # Average action losses as before. Each action's global/local weights
        # depend only on that action's gradients/EMA. Cross-action ranking is
        # the explicit, fixed-coefficient coupling and stays outside the EMA.
        priorities = features.new_tensor([1.0, cfg.local_auxiliary_weight] * 2)
        loss = (
            balancer.weighted_loss(tuple(losses[k] for k in balancer.names), priorities, pending)
            / 2
            + cfg.ranking_weight * losses["ranking"]
        )
        if cfg.selector_gain_mode == "signed_expectation":
            loss = loss + cfg.gain_probability_weight * losses["gain_probability"]
        if train_harm:
            loss = loss + cfg.harm_weight * losses["harm"]
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite selector objective")
        diagnostics.update({f"selector_{k}": v for k, v in balance_diagnostics.items()})
        diagnostics["selector_harm_in_objective"] = train_harm
        diagnostics["selector_ranking_weight"] = cfg.ranking_weight
        return loss, pending, diagnostics
    weighted_terms = [
        pending[1, i] * weight * losses[k]
        for i, (k, weight) in enumerate(
            (
                ("global", 1.0),
                ("local", cfg.local_auxiliary_weight),
                ("ranking", cfg.ranking_weight),
            )
        )
    ]
    loss = weighted_terms[0]
    for term in weighted_terms[1:]:
        loss = loss + term
    if train_harm:
        loss = loss + cfg.harm_weight * losses["harm"]
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite selector objective")
    diagnostics.update({f"selector_{k}": v for k, v in balance_diagnostics.items()})
    diagnostics["selector_harm_in_objective"] = train_harm
    return loss, pending, diagnostics
