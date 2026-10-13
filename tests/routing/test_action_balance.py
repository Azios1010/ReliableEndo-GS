"""Action-local balancing, joint ranking and guarded whole-head selection."""

import copy
from dataclasses import replace

import pytest
import torch

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.config.selector_refit import SelectorRefitConfig
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.calibrated_selector import CalibratedSelector
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES
from reliable_endo_gs.training.gradient_balance import GradientBalance
from reliable_endo_gs.training.selector_fitting import fit_selector_heads, joint_checkpoint_improves
from reliable_endo_gs.training.selector_optimization import (
    make_selector_balancer,
    selector_objective,
)
from tests.routing.test_selector_refit import table


def _config() -> InterventionConfig:
    return InterventionConfig(
        schema="intervention_selection.v2",
        selector_architecture="separate_action_heads",
        selector_balance_mode="per_action",
        ranking_weight=0.0,
        ranking_margin=10.0,
    )


def _observable_head(config: InterventionConfig) -> BalancedSelector:
    head = BalancedSelector(8, config)
    with torch.no_grad():
        for net in head.optimization_networks()[:2]:
            for layer in (net[0], net[2]):
                layer.weight.fill_(0.05)
                layer.bias.fill_(0.01)
            net[-1].weight.fill_(0.1)
            net[-1].bias.zero_()
        head.optimization_networks()[1][-1].weight[1].zero_()
    return head


@pytest.mark.parametrize("primed", [False, True])
def test_gaussian_targets_cannot_change_stereo_balance_or_gradients_without_ranking(primed):
    config = _config()
    head = _observable_head(config)
    x = torch.zeros(4, len(BALANCED_FEATURE_NAMES), requires_grad=True)
    local = torch.tensor([[0.0, 0.04, 0.0]]).expand(4, -1)
    balances, gradients = [], []
    for gaussian_target in (0.005, 0.08):
        current = copy.deepcopy(head)
        balance = make_selector_balancer(config)
        if primed:
            balance.ema.copy_(torch.tensor([0.1, 0.2, 10.0, 20.0]))
            balance.observations.fill_(3)
        targets = torch.tensor([[0.0, 0.01, gaussian_target]]).expand(4, -1).requires_grad_()
        loss, pending, _ = selector_objective(
            current, balance, x, targets, local, torch.zeros_like(targets)
        )
        loss.backward()
        balances.append(pending[:, :2])
        gradients.append(
            torch.cat([p.grad.flatten() for p in current.optimization_networks()[0].parameters()])
        )
        assert x.grad is None and targets.grad is None
        assert pending.shape == (2, 4)
    torch.testing.assert_close(balances[0], balances[1], atol=0, rtol=0)
    torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)


def test_cross_action_ranking_remains_an_explicit_coupling():
    config = replace(_config(), ranking_weight=1.0, ranking_margin=0.02)
    base = _observable_head(config)
    x = torch.zeros(4, len(BALANCED_FEATURE_NAMES))
    grads, ranking = [], []
    for gaussian_target in (-0.04, 0.08):
        head = copy.deepcopy(base)
        targets = torch.tensor([[0.0, 0.01, gaussian_target]]).expand(4, -1)
        loss, _, diag = selector_objective(
            head,
            make_selector_balancer(config),
            x,
            targets,
            torch.zeros_like(targets),
            torch.zeros_like(targets),
        )
        loss.backward()
        ranking.append(diag["selector_ranking_loss"])
        grads.append(
            torch.cat([p.grad.flatten() for p in head.optimization_networks()[0].parameters()])
        )
    assert all(value > 0 for value in ranking)
    assert not torch.equal(grads[0], grads[1])


def test_loss_components_keep_equal_action_mass_and_common_units():
    head = _observable_head(_config())
    x = torch.zeros(3, len(BALANCED_FEATURE_NAMES))
    labels = torch.tensor([[0.0, 0.1, -0.2]]).expand(3, -1)
    harms = torch.tensor([[0.0, 1.0, 0.0]]).expand(3, -1)
    losses, diag = head.objective(x, labels, labels, harms)
    for name in ("global", "local", "harm"):
        torch.testing.assert_close(
            losses[name], (losses[f"stereo_{name}"] + losses[f"gaussian_{name}"]) / 2
        )
    assert diag["selector_stereo_harm_rate"] == 1
    assert diag["selector_gaussian_harm_rate"] == 0
    assert "selector_stereo_global_gain_mae" in diag


def test_checkpoint_guard_rejects_masked_action_or_ranking_regression():
    accepted = {
        "stereo_global": 0.2,
        "gaussian_global": 0.2,
        "stereo_harm": 0.4,
        "gaussian_harm": 0.4,
        "ranking": 0.1,
    }
    good = {k: v / 2 for k, v in accepted.items()}
    assert joint_checkpoint_improves(good, accepted, 0.5, 1.0, 0.001)
    for component in ("stereo_global", "gaussian_global", "stereo_harm", "ranking"):
        bad = {**good, component: accepted[component] + 0.01}
        assert not joint_checkpoint_improves(bad, accepted, 0.5, 1.0, 0.001)
    assert not joint_checkpoint_improves(good, accepted, 1.0, 1.0, 0.001)
    with pytest.raises(ValueError):
        joint_checkpoint_improves({"unknown": 1.0}, accepted, 0.5, 1.0, 0.001)


def test_stationary_refit_selects_one_whole_snapshot_and_logs_both_actions():
    fit, validation = table((1, 2)), table((3, 4))
    config = replace(_config(), ranking_weight=1.0, ranking_margin=0.02)
    policy = SelectorRefitConfig(max_steps=4, min_steps=1, validate_every=1, patience=3)
    head, report = fit_selector_heads(fit, validation, config, policy, 8, torch.device("cpu"))
    assert report["selected_steps"]["gain"] == report["selected_steps"]["risk"]
    assert (
        report["checkpoint_selection_protocol"]
        == "whole_head_per_action_guard_and_joint_ranking.v1"
    )
    assert "ranking" in report["accepted_component_floors"]
    assert all("validation_action_metrics" in row for row in report["logs"])
    assert all(not p.requires_grad for p in head.parameters())
    assert bool(head.normalization.fitted)


@pytest.mark.parametrize("groups", [((0,),), ((0, 0),), ((), (0, 1)), ((False,), (1,))])
def test_balance_groups_reject_incomplete_or_overlapping_ownership(groups):
    with pytest.raises(ValueError, match="partition"):
        GradientBalance(("a", "b"), _config(), groups=groups)


def test_balance_mode_identity_rejects_shared_heads_or_wrong_balancer():
    with pytest.raises(ValueError, match="separate"):
        InterventionConfig(schema="intervention_selection.v2", selector_balance_mode="per_action")
    shared = replace(_config(), selector_balance_mode="shared")
    assert "selector_balance_mode" not in shared.to_dict()
    assert _config().config_hash != shared.config_hash
    head = _observable_head(_config())
    x = torch.zeros(1, len(BALANCED_FEATURE_NAMES))
    labels = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="ownership"):
        selector_objective(head, make_selector_balancer(shared), x, labels, labels, labels)


def test_calibrated_selector_rejects_different_balance_identity():
    config = replace(_config(), normalization_frames=1)
    head = _observable_head(config)
    x = torch.randn(2, len(BALANCED_FEATURE_NAMES))
    labels = torch.tensor([[0.0, -0.2, 0.3], [0.0, 0.1, 0.2]])
    head.normalization.observe(
        x, labels, labels, sample_id="dataset_1/keyframe_1/a", role="final_train"
    )
    calibrated = CalibratedSelector(head, config, {}, "synthetic")
    calibrated.validate_state()
    wrong = CalibratedSelector(
        head, replace(config, selector_balance_mode="shared"), {}, "synthetic"
    )
    with pytest.raises(ValueError, match="differs from deployment"):
        wrong.validate_state()
