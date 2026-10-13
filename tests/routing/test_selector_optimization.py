"""Fresh head updates, independent ownership, historical identities and resume."""

import copy
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.config.intervention import InterventionConfig, load_intervention_config
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES
from reliable_endo_gs.training.gradient_balance import GradientBalance
from reliable_endo_gs.training.selector_optimization import selector_objective
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_learning import StudentLearner
from tests.routing.test_balanced import setup as balanced_fixture
from tests.routing.test_metric_joint import metric_learner

setup = balanced_fixture


def test_neutral_defaults_preserve_historical_config_hashes():
    old = load_intervention_config(Path("configs/routing/intervention_metric_stereo_gain.yaml"))
    assert old.config_hash == "e5713af6ce98d538d7c0841989532ef27c129f225afd4b7bbe5e37b35fa8a214"
    assert (
        not {"selector_updates_per_snapshot", "selector_weight_decay", "ranking_gap_fraction"}
        & old.to_dict().keys()
    )
    new = load_intervention_config(Path("configs/routing/intervention_metric_selector_fit.yaml"))
    assert InterventionConfig(**new.to_dict()) == new
    assert new.metric_quality == old.metric_quality
    assert new.metric_scales == old.metric_scales
    assert new.stereo_utility_multiplier == old.stereo_utility_multiplier
    assert new.maximum_stereo_harm_probability == old.maximum_stereo_harm_probability
    assert new.maximum_gaussian_harm_probability == old.maximum_gaussian_harm_probability
    assert new.config_hash != old.config_hash


@pytest.mark.parametrize(
    "kwargs",
    [
        {"selector_updates_per_snapshot": 0},
        {"selector_updates_per_snapshot": 17},
        {"selector_updates_per_snapshot": True},
        {"selector_updates_per_snapshot": 2.0},
        {"selector_weight_decay": -1},
        {"selector_weight_decay": float("nan")},
        {"ranking_gap_fraction": -0.1},
        {"ranking_gap_fraction": 1.1},
        {"ranking_gap_fraction": float("nan")},
    ],
)
def test_invalid_optimization_settings(kwargs):
    with pytest.raises(ValueError):
        InterventionConfig(schema="intervention_selection.v2", **kwargs)


def test_v1_rejects_new_optimization_semantics():
    with pytest.raises(ValueError, match="v2"):
        InterventionConfig(selector_updates_per_snapshot=4)


def test_relative_ranking_margin_depends_on_verified_gap_and_preserves_harm_mask():
    cfg = InterventionConfig(schema="intervention_selection.v2", ranking_margin_units="train_scale")
    fixed = BalancedSelector(8, cfg)
    relative = BalancedSelector(8, replace(cfg, ranking_gap_fraction=0.5))
    relative.load_state_dict(fixed.state_dict())
    features = torch.zeros(2, len(BALANCED_FEATURE_NAMES))
    targets = torch.tensor([[0.0, 0.4, 0.1], [0.0, -0.2, 0.3]])
    harms = torch.zeros_like(targets)
    old, _ = fixed.objective(features, targets, targets, harms)
    new, _ = relative.objective(features, targets, targets, harms)
    assert new["ranking"] > old["ranking"]
    assert torch.equal(new["global"], old["global"])
    assert torch.equal(new["harm"], old["harm"])
    physical = load_intervention_config(
        Path("configs/routing/intervention_metric_selector_fit.yaml")
    )
    head = BalancedSelector(8, physical)
    harms[:, 1:] = 1
    losses, _ = head.objective(features, targets, targets, harms)
    assert losses["ranking"] == 0


def _on_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value):
        return replace(
            value, **{f.name: _on_device(getattr(value, f.name), device) for f in fields(value)}
        )
    return value


@pytest.mark.parametrize("unknown_position", [None, 0, 3])
@pytest.mark.parametrize("gain_only", [False, True])
@pytest.mark.parametrize(
    "architecture,balance_mode",
    [
        ("shared_actions", "shared"),
        ("separate_action_heads", "shared"),
        ("separate_action_heads", "per_action"),
    ],
)
def test_fresh_joint_updates_and_resume_own_each_optimizer_step(
    setup, tmp_path, unknown_position, gain_only, architecture, balance_mode
):
    old, sample, target, truth = metric_learner(setup)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = replace(
        old.intervention_config,
        selector_updates_per_snapshot=4,
        selector_weight_decay=0.01,
        ranking_gap_fraction=0.5,
        selector_gain_only_refinement=gain_only,
        selector_architecture=architecture,
        selector_balance_mode=balance_mode,
    )
    config = replace(old.config, joint_epochs=5)
    model = old.model.to(device)
    learner = StudentLearner(model, config, old.routing, intervention_config=cfg)
    sample, target, truth = (_on_device(x, device) for x in (sample, target, truth))
    unknown = replace(truth, valid_mask=torch.zeros_like(truth.valid_mask))
    original_commit = learner.commit_gradients
    frozen_owners_at_head_updates = []
    risk_versions_at_head_updates = []

    def track_commit(joint, **kwargs):
        if kwargs.get("selector_only"):
            frozen_owners_at_head_updates.append(
                tuple(p._version for m in (learner.model, learner.actions) for p in m.parameters())
            )
            risk_versions_at_head_updates.append(
                tuple(p._version for p in learner.selector.harm_network.parameters())
            )
        return original_commit(joint, **kwargs)

    learner.commit_gradients = track_commit
    for position in range(4):
        result = learner.step(
            sample,
            joint=True,
            teacher_target=target,
            metric_truth=unknown if position == unknown_position else truth,
            position=position,
        )
        assert result["selector_updates_this_snapshot"] in (0, 4)
        if result["selector_updates_this_snapshot"] and architecture == "separate_action_heads":
            for action in ("stereo", "gaussian"):
                assert result[f"selector_{action}_harm_brier"] >= 0
                for task in ("gain", "risk"):
                    assert result[f"optimizer_update_norm_selector_{action}_{task}"] > 0
        assert result["selector_harm_updates_this_snapshot"] == (
            (1 if gain_only else 4) if result["selector_updates_this_snapshot"] else 0
        )
        if position == unknown_position:
            assert result["selector_updates_this_snapshot"] == 0
    expected = 8 if unknown_position is None else 4
    assert learner.optimizer_update_counts == {"model": 4, "actions": 4, "selector": expected}
    assert int(learner.selector_balancer.observations) == expected
    assert len(frozen_owners_at_head_updates) == expected // 4 * 3
    for start in range(0, len(frozen_owners_at_head_updates), 3):
        assert len(set(frozen_owners_at_head_updates[start : start + 3])) == 1
        if gain_only:
            assert len(set(risk_versions_at_head_updates[start : start + 3])) == 1
    assert learner.selector_optimizer.param_groups[0]["weight_decay"] == 0.01
    for group in learner.selector_optimizer.param_groups:
        steps = expected // 4 if group.get("name") == "harm" else expected
        assert all(
            int(learner.selector_optimizer.state[p]["step"]) == steps for p in group["params"]
        )
    path = tmp_path / "joint.pt"
    identity = dict(
        source_sha256="source",
        manifest_hash="train",
        role_hash="roles",
        renderer_identity="synthetic",
    )
    cursor = dict(epoch=4, next_sample_index=0, samples_seen=4, optimizer_steps=4)
    save_student_checkpoint(path, learner, identity, cursor)
    restored = StudentLearner(copy.deepcopy(model), config, old.routing, intervention_config=cfg)
    assert load_student_checkpoint(path, restored, identity, 1) == cursor
    assert restored.optimizer_update_counts == learner.optimizer_update_counts
    restored.step(sample, joint=True, teacher_target=target, metric_truth=truth, position=4)
    assert restored.optimizer_update_counts == {"model": 5, "actions": 5, "selector": expected + 4}
    raw = torch.load(path, weights_only=True)
    raw["optimizer_update_counts"]["selector"] += 1
    invalid = tmp_path / "bad.pt"
    torch.save(raw, invalid)
    before = copy.deepcopy(restored.selector.state_dict())
    with pytest.raises(ValueError, match="update counts"):
        load_student_checkpoint(invalid, restored, identity, 1)
    assert all(torch.equal(v, before[k]) for k, v in restored.selector.state_dict().items())
    raw["optimizer_update_counts"]["selector"] = expected + 4
    wrong_adam = tmp_path / "bad-adam.pt"
    torch.save(raw, wrong_adam)
    with pytest.raises(ValueError, match="balance update count|Adam step|burn-in"):
        load_student_checkpoint(wrong_adam, restored, identity, 1)
    raw = torch.load(path, weights_only=True)
    raw["selector_gradient_balance"]["observations"].add_(1)
    wrong_balance = tmp_path / "bad-balance.pt"
    torch.save(raw, wrong_balance)
    with pytest.raises(ValueError, match="balance update count|cursor mismatch"):
        load_student_checkpoint(wrong_balance, restored, identity, 1)


def test_selector_objective_detaches_all_inputs():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = InterventionConfig(schema="intervention_selection.v2")
    head = BalancedSelector(8, cfg).to(device)
    balance = GradientBalance(("global", "local", "ranking"), cfg).to(device)
    features = torch.randn(3, len(BALANCED_FEATURE_NAMES), device=device, requires_grad=True)
    target = torch.tensor([[0.0, 0.2, -0.1]] * 3, device=device, requires_grad=True)
    loss, _, _ = selector_objective(
        head, balance, features, target, target, torch.zeros_like(target)
    )
    loss.backward()
    assert features.grad is None and target.grad is None
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
