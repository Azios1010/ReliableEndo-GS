"""Rare beneficial actions, signed expectation and joint ownership contracts."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from reliable_endo_gs.config.intervention import load_intervention_config
from reliable_endo_gs.config.selector_refit import SelectorRefitConfig
from reliable_endo_gs.evaluation.gain_events import gain_event_metrics
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.calibrated_selector import CalibratedSelector
from reliable_endo_gs.routing.gain import (
    SignedGainPrediction,
    decode_signed_gain,
    signed_gain_losses,
)
from reliable_endo_gs.training.metric_quality import metric_targets
from reliable_endo_gs.training.selector_fitting import fit_selector_heads
from reliable_endo_gs.training.selector_optimization import (
    make_selector_balancer,
    selector_objective,
)
from reliable_endo_gs.training.selector_outcomes import SelectorOutcomes
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_learning import StudentLearner
from tests.routing.test_balanced import setup as balanced_fixture
from tests.routing.test_mask_tokens_risk import learner_inputs
from tests.training.test_student_freeze_ablation import assert_state_equal

setup = balanced_fixture


def config():
    return load_intervention_config(Path("configs/routing/intervention_metric_signed_gain.yaml"))


def test_opt_in_identity_and_physical_utility_are_preserved():
    old = load_intervention_config(Path("configs/routing/intervention_metric_mask_tokens.yaml"))
    assert old.config_hash == "02dca54c842feb0820a949f3414d75c14249373f1f4d007d9640c1909de2b7f8"
    assert "selector_gain_mode" not in old.to_dict()
    assert "gain_probability_weight" not in old.to_dict()
    new = config()
    assert type(new)(**new.to_dict()) == new
    assert replace(new, selector_gain_mode="regression") == old
    for action in (1, 2):
        changes = [0.1] * 12
        assert metric_targets(changes, old, action) == metric_targets(changes, new, action)
    frozen = load_intervention_config(
        Path("configs/routing/intervention_metric_signed_gain_frozen_student.yaml")
    )
    assert replace(frozen, student_update_mode="joint") == new
    for kwargs in (
        {"selector_gain_mode": "unknown"},
        {"selector_balance_mode": "shared"},
        {"gain_probability_weight": 0},
        {"gain_probability_weight": float("nan")},
        {"gain_probability_weight": float("inf")},
        {"gain_probability_weight": True},
        {"gain_probability_weight": "bad"},
        {"metric_quality": None},
    ):
        with pytest.raises(ValueError):
            replace(new, **kwargs)


def test_signed_expected_gain_units_negative_cost_and_component_packing():
    probability = torch.tensor([[0.2, 0.8]])
    positive = torch.tensor([[1.0, 0.2]])
    negative = torch.tensor([[0.1, 0.4]])
    scale = torch.tensor(0.1)
    raw = torch.stack(
        (
            torch.logit(probability),
            torch.log(torch.expm1(positive / scale)),
            torch.log(torch.expm1(negative / scale)),
        ),
        -1,
    )
    pred = decode_signed_gain(raw, scale)
    torch.testing.assert_close(pred.expected_gain, torch.tensor([[0.12, 0.08]]))
    head = BalancedSelector(8, config())
    head.normalization.gain_scales.copy_(torch.tensor([0.1, 0.2]))
    with torch.no_grad():
        for ai, action in enumerate(("stereo", "gaussian")):
            head.gain_network.branches[action][-1].bias.copy_(
                torch.cat((raw[0, ai], torch.tensor([ai + 1.0])))
            )
    x = torch.randn(3, 157, requires_grad=True)
    gain, local, risk, events, parts = head.outputs_with_components(x)
    torch.testing.assert_close(gain[:, 1:], pred.expected_gain.expand(3, -1))
    torch.testing.assert_close(local[:, 1:], torch.tensor([[0.2, 0.4]]).expand(3, -1))
    assert parts is not None and events is not None
    assert torch.equal(gain[:, 0], torch.zeros(3)) and torch.equal(risk[:, 0], torch.zeros(3))
    assert torch.equal(events[:, 0], torch.zeros(3, 3))
    gain.sum().backward()
    assert x.grad is None
    assert all(p.grad is None for p in head.harm_network.parameters())
    calibrated = CalibratedSelector(head, head.config, {}, "mask")
    assert calibrated.head.config.selector_gain_mode == "signed_expectation"


def test_gain_readout_expansion_preserves_risk_trunks_and_rng_stream():
    cfg = config()
    torch.manual_seed(16)
    old = BalancedSelector(8, replace(cfg, selector_gain_mode="regression"))
    old_rng = torch.random.get_rng_state()
    torch.manual_seed(16)
    new = BalancedSelector(8, cfg)
    assert torch.equal(old_rng, torch.random.get_rng_state())
    assert_state_equal(old.harm_network.state_dict(), new.harm_network.state_dict())
    for action in ("stereo", "gaussian"):
        assert_state_equal(
            old.gain_network.branches[action][:-1].state_dict(),
            new.gain_network.branches[action][:-1].state_dict(),
        )


def test_conditional_magnitudes_ignore_other_sign_frequency_and_preserve_weight_scale():
    targets = torch.tensor([[1.0, -0.2], [-0.1, 0.8], [-0.3, 0.0]])
    pred = SignedGainPrediction(
        torch.zeros_like(targets), torch.full_like(targets, 0.4), torch.full_like(targets, 0.5)
    )
    loss = signed_gain_losses(pred, targets, torch.tensor(1.0))
    torch.testing.assert_close(loss["positive"], torch.tensor([0.36, 0.16]))
    torch.testing.assert_close(loss["negative"], torch.tensor([0.10, 0.17]))
    assert loss["probability"][0] == pytest.approx(
        float(F.binary_cross_entropy_with_logits(torch.zeros(3), (targets[:, 0] > 0).float()))
    )
    duplicate = torch.cat((targets, targets[1:2].repeat(20, 1)))
    duplicate_pred = SignedGainPrediction(
        torch.zeros_like(duplicate),
        torch.full_like(duplicate, 0.4),
        torch.full_like(duplicate, 0.5),
    )
    repeated = signed_gain_losses(duplicate_pred, duplicate, torch.tensor(1.0))
    torch.testing.assert_close(repeated["positive"], loss["positive"])
    for weights in (torch.ones(3), torch.ones(3) * 1e-4):
        weighted = signed_gain_losses(pred, targets, torch.tensor(1.0), weights)
        for key in ("positive", "negative", "probability"):
            torch.testing.assert_close(weighted[key], loss[key])


def test_missing_class_has_zero_magnitude_gradient_and_zero_gain_is_nonpositive():
    positive = torch.ones(3, 2, requires_grad=True)
    negative = torch.ones(3, 2, requires_grad=True)
    pred = SignedGainPrediction(torch.zeros(3, 2, requires_grad=True), positive, negative)
    losses = signed_gain_losses(pred, torch.ones(3, 2), torch.tensor(1.0))
    losses["negative"].sum().backward()
    assert torch.equal(negative.grad, torch.zeros_like(negative))
    zeros = signed_gain_losses(pred, torch.zeros(3, 2), torch.tensor(1.0))
    assert torch.equal(zeros["positive_count"], torch.zeros(2, dtype=torch.long))
    assert torch.equal(zeros["negative_count"], torch.full((2,), 3, dtype=torch.long))
    for targets in (torch.full((3, 2), float("nan")), torch.ones(0, 2), torch.ones(3, 3)):
        with pytest.raises(ValueError):
            signed_gain_losses(pred, targets, torch.tensor(1.0))


def test_probability_learns_true_rare_frequency_without_class_bonus():
    logits = torch.nn.Parameter(torch.zeros(100, 2))
    # Shared constant logit: no evidence distinguishes these draws.
    bias = torch.nn.Parameter(torch.zeros(1, 2))
    targets = torch.full((100, 2), -0.1)
    targets[:10, 0] = 1
    targets[:80, 1] = 0.2
    optimizer = torch.optim.Adam([bias], lr=0.1)
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        pred = SignedGainPrediction(
            bias.expand_as(logits), targets.new_ones(targets.shape), targets.new_ones(targets.shape)
        )
        loss = signed_gain_losses(pred, targets, torch.tensor(1.0))["probability"].mean()
        loss.backward()
        optimizer.step()
    torch.testing.assert_close(bias.sigmoid()[0], torch.tensor([0.1, 0.8]), atol=0.003, rtol=0)


def test_gain_head_learns_rare_large_benefit_and_common_small_benefit():
    torch.manual_seed(7)
    cfg = replace(
        config(), selector_feature_mode="statistics", selector_token_channels=0, ranking_weight=0.0
    )
    head = BalancedSelector(16, cfg)
    head.normalization.gain_scales.fill_(0.1)
    x = torch.zeros(64, 61)
    x[:, 0] = -1
    x[:8, 0] = 1
    targets = torch.zeros(64, 3)
    targets[:, 1] = -0.1
    targets[:8, 1] = 0.8
    targets[:, 2] = 0.1
    optimizer = torch.optim.Adam(head.gain_network.parameters(), lr=0.01)
    for _ in range(250):
        optimizer.zero_grad(set_to_none=True)
        losses, _ = head.objective(
            x, targets, targets * 0, targets * 0, risk_components=torch.zeros(64, 3, 2)
        )
        loss = losses["global"] + losses["gain_probability"]
        loss.backward()
        optimizer.step()
    gain, _, _, _, parts = head.outputs_with_components(x)
    assert parts is not None
    assert bool((gain[:8, 1] > gain[:8, 2]).all())
    assert bool((gain[8:, 1] < 0).all())
    assert bool((parts.positive_probability[:8, 0] > 0.9).all())
    assert float((gain[:, 1:] - targets[:, 1:]).abs().mean()) < 0.03


@pytest.mark.parametrize("frozen", [False, True])
def test_joint_harness_ownership_resume_and_topology_rejection(setup, tmp_path, frozen):
    old, sample, teacher, truth = learner_inputs(setup, tmp_path / "roi", frozen)
    learner = StudentLearner(
        old.model,
        old.config,
        old.routing,
        intervention_config=replace(
            old.intervention_config, selector_gain_mode="signed_expectation"
        ),
        mask_config=old.mask_config,
        target_masks=old.target_masks,
    )
    before = copy.deepcopy(learner.model.state_dict())
    for step in range(3):
        result = learner.step(
            sample, joint=True, teacher_target=teacher, metric_truth=truth, position=step
        )
    assert "predicted_gain_components" in result
    assert result["selector_updates_this_snapshot"] == 4
    assert "selector_stereo_gain_positive_brier" in result
    if frozen:
        assert_state_equal(before, learner.model.state_dict())
    else:
        assert any(not torch.equal(v, learner.model.state_dict()[k]) for k, v in before.items())
    checkpoint = tmp_path / "signed.pt"
    cursor = {"epoch": 0, "next_sample_index": 3, "optimizer_steps": 3, "samples_seen": 3}
    save_student_checkpoint(checkpoint, learner, {}, cursor)
    assert load_student_checkpoint(checkpoint, learner, {}, 10) == cursor
    with pytest.raises(ValueError):
        load_student_checkpoint(checkpoint, old, {}, 10)
    counts = dict(learner.optimizer_update_counts)
    unknown = replace(truth, valid_mask=torch.zeros_like(truth.valid_mask))
    result = learner.step(
        sample, joint=True, teacher_target=teacher, metric_truth=unknown, position=3
    )
    assert result["selector_updates_this_snapshot"] == 0
    assert learner.optimizer_update_counts["selector"] == counts["selector"]


def test_probability_loss_is_an_explicit_optimization_term():
    cfg = config()
    head = BalancedSelector(8, cfg)
    x = torch.zeros(8, 157)
    targets = torch.zeros(8, 3)
    targets[:, 1:] = 0.2
    balancer = make_selector_balancer(cfg)
    objective, _, diag = selector_objective(
        head, balancer, x, targets, targets * 0, targets * 0, risk_components=torch.zeros(8, 3, 2)
    )
    assert diag["selector_gain_probability_weight"] == 1
    objective.backward()
    for branch in head.gain_network.branches.values():
        assert branch[-1].bias.grad[0] != 0


def test_event_metrics_exclude_unknown_and_count_safe_oracle_opportunities():
    record = {
        "valid_regions": [True, True],
        "actual_label_available": [[True, True, True], [True, False, False]],
        "actual_gains": [[0, 0.8, -0.1], [0, 100, 100]],
        "actual_harms": [[False, False, True], [False, False, False]],
        "predicted_gain_components": {
            "positive_probability": [[0, 0.9, 0.1], [0, 0.9, 0.9]],
            "positive_gain": [[0, 0.7, 0.2], [0, 100, 100]],
            "negative_cost": [[0, 0.1, 0.1], [0, 0.1, 0.1]],
        },
        "policies": {
            "selector": {"action": "STEREO", "region": 0},
            "oracle": {"action": "STEREO", "region": 0},
        },
    }
    result = gain_event_metrics([record])
    assert result["STEREO"]["known_candidates"] == 1
    assert result["STEREO"]["safe_oracle_opportunity_recall"] == 1
    assert result["STEREO"]["positive_conditional_magnitude_mae"] == pytest.approx(0.1)
    assert result["GAUSSIAN"]["positive_gain_rate"] == 0


def test_stationary_fit_scores_and_guards_sign_probability():
    torch.manual_seed(9)
    cfg = config()
    features = torch.randn(16, 157)
    targets = torch.zeros(16, 3)
    targets[:, 1:] = features[:, :2] * 0.1
    components = torch.zeros(16, 3, 2)

    def table(fold):
        return SelectorOutcomes(
            {"collection_role": "train"},
            [],
            features,
            targets,
            targets * 0,
            targets * 0,
            tuple(f"dataset_{fold}/keyframe_1/frame_data{i:06d}" for i in range(16)),
            components,
        )

    head, report = fit_selector_heads(
        table(1),
        table(2),
        cfg,
        SelectorRefitConfig(
            max_steps=3, min_steps=1, validate_every=1, validation_guard_mode="action_objectives"
        ),
        8,
        torch.device("cpu"),
    )
    assert head.outputs_with_components(features)[4] is not None
    for log in report["logs"]:
        for action in ("stereo", "gaussian"):
            assert f"{action}_gain_probability" in log["validation_components"]
            assert f"{action}_gain_probability" in log["validation_guard_components"]
