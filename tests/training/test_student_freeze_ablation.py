"""Matched joint/frozen ablation: fixed weights, live repairs and strict resume."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.config.intervention import load_intervention_config
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_learning import StudentLearner
from tests.routing.test_balanced import setup as balanced_fixture
from tests.routing.test_metric_joint import metric_learner

setup = balanced_fixture


def learner_inputs(setup, *, frozen):
    old, sample, teacher, truth = metric_learner(setup)
    cfg = replace(
        old.intervention_config,
        geometry_supervision_weight=1,
        student_update_mode="frozen" if frozen else "joint",
    )
    learner = StudentLearner(old.model, old.config, old.routing, intervention_config=cfg)
    return learner, sample, teacher, replace(truth, disparity=truth.disparity * 1.1)


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_state_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_state_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("frozen", [False, True])
def test_optimizer_ownership_and_geometry_loss_remain_real(setup, frozen, tmp_path):
    learner, sample, teacher, truth = learner_inputs(setup, frozen=frozen)
    model = copy.deepcopy(learner.model.state_dict())
    adam = copy.deepcopy(learner.model_optimizer.state_dict())
    actions = copy.deepcopy(learner.actions.state_dict())
    result = learner.step(
        sample, joint=True, teacher_target=teacher, metric_truth=truth, position=0
    )
    assert result["model_update_enabled"] is not frozen
    assert result["geometry_base_loss"] > 0
    assert result["grad_norm_stereo_repair"] > 0 and result["grad_norm_gaussian_repair"] > 0
    assert any(not torch.equal(v, learner.actions.state_dict()[k]) for k, v in actions.items())
    if frozen:
        assert_state_equal(model, learner.model.state_dict())
        assert_state_equal(adam, learner.model_optimizer.state_dict())
        assert all(p.grad is None for p in learner.model.parameters())
        assert not learner.model.training
        assert learner.optimizer_update_counts == {"model": 0, "actions": 1, "selector": 0}
        # The same virtual model probe remains present; frozen reference/input
        # Jacobians do not accidentally prevent local stereo learning.
        assert result["model_shared_grad_geometry"] > 0
    else:
        assert any(not torch.equal(v, learner.model.state_dict()[k]) for k, v in model.items())
        assert learner.optimizer_update_counts["model"] == 1
    path = tmp_path / "checkpoint.pt"
    cursor = {"epoch": 0, "next_sample_index": 1, "samples_seen": 1, "optimizer_steps": 1}
    save_student_checkpoint(path, learner, {}, cursor)
    assert load_student_checkpoint(path, learner, {}, 2) == cursor


def test_frozen_continuation_preserves_existing_model_adam_and_anchor(setup, tmp_path):
    parent, sample, teacher, truth = learner_inputs(setup, frozen=False)
    parent.step(sample, joint=True, teacher_target=teacher, metric_truth=truth)
    frozen = StudentLearner(
        copy.deepcopy(parent.model),
        parent.config,
        parent.routing,
        intervention_config=replace(parent.intervention_config, student_update_mode="frozen"),
    )
    for name in ("actions", "selector", "objective_balancer", "selector_balancer"):
        getattr(frozen, name).load_state_dict(getattr(parent, name).state_dict())
    for name in ("model_optimizer", "action_optimizer", "selector_optimizer"):
        getattr(frozen, name).load_state_dict(getattr(parent, name).state_dict())
    frozen.optimizer_update_counts = dict(parent.optimizer_update_counts)
    frozen.cost_observations = parent.cost_observations
    frozen.cost_max_ms.copy_(parent.cost_max_ms)
    frozen.cost_history.copy_(parent.cost_history)
    frozen.cost_history_observations = parent.cost_history_observations
    frozen.frozen_student_start_step = 1
    model = copy.deepcopy(frozen.model.state_dict())
    adam = copy.deepcopy(frozen.model_optimizer.state_dict())
    for step in (1, 2):
        frozen.step(sample, joint=True, teacher_target=teacher, metric_truth=truth, position=step)
    assert_state_equal(model, frozen.model.state_dict())
    assert_state_equal(adam, frozen.model_optimizer.state_dict())
    assert frozen.optimizer_update_counts == {"model": 1, "actions": 3, "selector": 1}
    path = tmp_path / "frozen.pt"
    cursor = {"epoch": 0, "next_sample_index": 3, "samples_seen": 3, "optimizer_steps": 3}
    save_student_checkpoint(path, frozen, {}, cursor)
    assert load_student_checkpoint(path, frozen, {}, 4) == cursor
    assert frozen.frozen_student_start_step == 1
    raw = torch.load(path, weights_only=True)
    raw["frozen_student_start_step"] = 2
    bad = tmp_path / "invalid.pt"
    torch.save(raw, bad)
    with pytest.raises(ValueError, match="optimizer update counts"):
        load_student_checkpoint(bad, frozen, {}, 4)


def test_freeze_opt_in_keeps_historical_hash_and_rejects_incompatible_schedule(setup):
    cfg = load_intervention_config(Path("configs/routing/intervention_metric_action_heads.yaml"))
    assert cfg.config_hash == "8fb6905c83f04b6bfbefef65a1b7ec4273cf0f5de6fe4dfa16ac2bf5fb530eec"
    assert "student_update_mode" not in cfg.to_dict()
    old, _, _ = setup
    with pytest.raises(ValueError, match="warmup"):
        StudentLearner(
            old.model,
            replace(old.config, student_epochs=1),
            old.routing,
            intervention_config=replace(cfg, student_update_mode="frozen"),
        )
    with pytest.raises(ValueError, match="mode"):
        replace(cfg, student_update_mode="depth_only")
