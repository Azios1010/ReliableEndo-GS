"""Independent action encoders, common utility units and strict checkpoint identity."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.config.intervention import InterventionConfig, load_intervention_config
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.calibrated_selector import CalibratedSelector
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES
from reliable_endo_gs.training.gradient_balance import GradientBalance
from reliable_endo_gs.training.selector_optimization import selector_objective
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_learning import StudentLearner
from tests.routing.test_balanced import setup as balanced_fixture

setup = balanced_fixture


def _head() -> BalancedSelector:
    return BalancedSelector(
        8,
        InterventionConfig(
            schema="intervention_selection.v2", selector_architecture="separate_action_heads"
        ),
    )


def test_action_architecture_is_explicit_and_common_policy_is_unchanged():
    common = load_intervention_config(
        Path("configs/routing/intervention_metric_common_utility.yaml")
    )
    split = load_intervention_config(Path("configs/routing/intervention_metric_action_heads.yaml"))
    assert split == replace(
        common, selector_architecture="separate_action_heads", selector_balance_mode="per_action"
    )
    assert split.config_hash != common.config_hash
    assert "selector_architecture" not in common.to_dict()
    assert InterventionConfig(**split.to_dict()) == split
    with pytest.raises(ValueError, match="v2"):
        InterventionConfig(selector_architecture="separate_action_heads")
    with pytest.raises(ValueError, match="architecture"):
        InterventionConfig(schema="intervention_selection.v2", selector_architecture="three_heads")


def test_action_networks_have_disjoint_parameters_and_no_stop_parameters():
    head = _head()
    networks = head.optimization_networks()
    assert len(networks) == 4
    owners = [{id(p) for p in net.parameters()} for net in networks]
    for i, owner in enumerate(owners):
        for other in owners[i + 1 :]:
            assert owner.isdisjoint(other)
    assert set().union(*owners) == {id(p) for p in head.parameters()}
    expected_trunks = {
        id(p)
        for net in networks[:2]
        for layer in list(net.children())[:-1]
        for p in layer.parameters()
    }
    shared = head.gain_shared_parameters()
    assert {id(p) for p in shared} == expected_trunks
    assert len(shared) == len(expected_trunks)
    assert not any("stop" in key.lower() for key in head.state_dict())


def test_action_output_order_common_scale_and_signed_stop_reference():
    head = _head()
    stereo_gain, gaussian_gain, stereo_risk, gaussian_risk = head.optimization_networks()
    with torch.no_grad():
        head.normalization.gain_scales.copy_(torch.tensor([0.1, 0.2]))
        stereo_gain[-1].bias.copy_(torch.tensor([-2.0, 3.0]))
        gaussian_gain[-1].bias.copy_(torch.tensor([4.0, -5.0]))
        stereo_risk[-1].bias.fill_(-1.0)
        gaussian_risk[-1].bias.fill_(2.0)
    x = torch.zeros(2, len(BALANCED_FEATURE_NAMES))
    global_gain, local_gain, risk = head.outputs(x)
    torch.testing.assert_close(global_gain, torch.tensor([[0.0, -0.2, 0.4]]).expand(2, -1))
    torch.testing.assert_close(local_gain, torch.tensor([[0.0, 0.6, -1.0]]).expand(2, -1))
    torch.testing.assert_close(risk[:, 1:], torch.tensor([[-1.0, 2.0]]).sigmoid().expand(2, -1))
    assert torch.equal(risk[:, 0], torch.zeros(2))


@pytest.mark.parametrize("action", [0, 1])
@pytest.mark.parametrize("task", ["gain", "risk"])
def test_action_gradient_does_not_reach_other_action_or_task(action, task):
    head = _head()
    networks = head.optimization_networks()
    # Nonzero outputs make trunk isolation observable, beyond zero initialization.
    with torch.no_grad():
        for net in networks:
            net[-1].weight.fill_(0.1)
    x = torch.randn(3, len(BALANCED_FEATURE_NAMES), requires_grad=True)
    gains, _, logits = head.utility_and_risk_logits(x)
    value = gains[:, action + 1].sum() if task == "gain" else logits[:, action].sum()
    value.backward()
    owner_index = action + (0 if task == "gain" else 2)
    assert x.grad is None
    for i, net in enumerate(networks):
        grads = [p.grad for p in net.parameters()]
        if i == owner_index:
            assert any(g is not None and bool((g != 0).any()) for g in grads)
        else:
            assert all(g is None or bool((g == 0).all()) for g in grads)


def test_both_action_gain_and_risk_heads_learn_and_state_roundtrips():
    torch.manual_seed(3)
    head = _head()
    before = copy.deepcopy(head.state_dict())
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    balancer = GradientBalance(("global", "local", "ranking"), head.config)
    x = torch.randn(4, len(BALANCED_FEATURE_NAMES), requires_grad=True)
    labels = torch.tensor([[0.0, -0.2, 0.3]]).expand(4, -1)
    harms = torch.tensor([[0.0, 1.0, 0.0]]).expand(4, -1)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, pending, _ = selector_objective(head, balancer, x, labels, labels, harms)
        loss.backward()
        assert all(
            p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()
        )
        for net in head.optimization_networks():
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        balancer.commit(pending)
    for task in ("gain", "harm"):
        for action in ("stereo", "gaussian"):
            key = f"{task}_network.branches.{action}.4.bias"
            assert not torch.equal(head.state_dict()[key], before[key])
    clone = _head()
    clone.load_state_dict(head.state_dict(), strict=True)
    for result, restored in zip(head.outputs(x), clone.outputs(x), strict=True):
        torch.testing.assert_close(result, restored)
    assert x.grad is None


def test_calibrated_selector_accepts_action_heads_and_rejects_architecture_mismatch():
    head = BalancedSelector(8, replace(_head().config, normalization_frames=1))
    x = torch.randn(2, len(BALANCED_FEATURE_NAMES))
    labels = torch.tensor([[0.0, -0.2, 0.3], [0.0, 0.1, 0.2]])
    head.normalization.observe(
        x, labels, labels, sample_id="dataset_1/keyframe_1/a", role="final_train"
    )
    calibrated = CalibratedSelector(head, head.config, {}, "synthetic")
    calibrated.validate_state()
    gain, risk = calibrated.predict(x)
    assert gain.shape == risk.shape == (2, 3)
    assert torch.equal(gain[:, 0], risk[:, 0]) and bool((gain[:, 0] == 0).all())
    wrong = CalibratedSelector(
        head, replace(head.config, selector_architecture="shared_actions"), {}, "synthetic"
    )
    with pytest.raises(ValueError, match="architecture"):
        wrong.validate_state()


@pytest.mark.parametrize(
    "saved_architecture,saved_balance,other_architecture,other_balance",
    [
        ("shared_actions", "shared", "separate_action_heads", "shared"),
        ("separate_action_heads", "shared", "shared_actions", "shared"),
        ("separate_action_heads", "shared", "separate_action_heads", "per_action"),
        ("separate_action_heads", "per_action", "separate_action_heads", "shared"),
    ],
)
def test_checkpoint_rejects_other_architecture_before_mutation(
    setup, tmp_path, saved_architecture, saved_balance, other_architecture, other_balance
):
    old, _, _ = setup
    saved = StudentLearner(
        old.model,
        old.config,
        old.routing,
        intervention_config=replace(
            old.intervention_config,
            selector_architecture=saved_architecture,
            selector_balance_mode=saved_balance,
        ),
    )
    restored = StudentLearner(
        copy.deepcopy(old.model),
        old.config,
        old.routing,
        intervention_config=replace(
            old.intervention_config,
            selector_architecture=other_architecture,
            selector_balance_mode=other_balance,
        ),
    )
    checkpoint = tmp_path / "initial.pt"
    save_student_checkpoint(
        checkpoint,
        saved,
        {},
        {"epoch": 0, "next_sample_index": 0, "optimizer_steps": 0, "samples_seen": 0},
    )
    before = copy.deepcopy(restored.selector.state_dict())
    with pytest.raises(ValueError, match="protocol/architecture"):
        load_student_checkpoint(checkpoint, restored, {}, 1)
    assert all(
        torch.equal(value, before[key]) for key, value in restored.selector.state_dict().items()
    )
