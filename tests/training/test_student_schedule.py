"""Epoch LR boundaries, strict checkpoint validation and exact split-run recovery."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    read_student_checkpoint,
)
from reliable_endo_gs.training.student_config import StudentLearningConfig, load_student_config
from reliable_endo_gs.training.student_learning import StudentLearner
from reliable_endo_gs.training.student_run import run_student_training
from reliable_endo_gs.training.student_schedule import (
    apply_model_learning_rate,
    checkpoint_model_learning_rate,
    model_learning_rate,
)
from tests.training.test_student_joint import setup as joint_fixture

setup = joint_fixture


def test_constant_schedule_keeps_historical_identity():
    config = StudentLearningConfig()
    assert config.config_hash == "6c1b50685777e7db3c6605cd1abaa8823ebcb91c320f31f15cbc9ffb7229388d"
    assert not any(k.startswith("model_lr_") for k in config.to_dict())
    assert StudentLearningConfig(**config.to_dict()) == config
    assert all(model_learning_rate(config, epoch) == config.learning_rate for epoch in (0, 9, 99))


def test_cosine_endpoints_prefix_and_config_roundtrip(tmp_path):
    config = StudentLearningConfig(model_lr_schedule="cosine_epochs", student_epochs=10)
    rates = [model_learning_rate(config, epoch) for epoch in range(12)]
    assert rates[0] == config.learning_rate
    assert rates[9] == pytest.approx(config.learning_rate * 0.1)
    assert rates[10] == rates[11] == rates[9]
    assert all(a > b for a, b in zip(rates[:9], rates[1:10], strict=True))
    shorter = replace(config, student_epochs=3)
    assert [model_learning_rate(shorter, i) for i in range(3)] == rates[:3]
    assert config.config_hash != StudentLearningConfig(student_epochs=10).config_hash
    path = tmp_path / "schedule.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    assert load_student_config(path) == config


@pytest.mark.parametrize(
    "options",
    [
        {"model_lr_schedule": "unknown"},
        {"model_lr_decay_epochs": True},
        {"model_lr_decay_epochs": 1},
        {"model_lr_min_ratio": True},
        {"model_lr_min_ratio": 0},
        {"model_lr_min_ratio": 1.1},
        {"model_lr_min_ratio": float("nan")},
        {"model_lr_min_ratio": float("inf")},
        {"model_lr_schedule": "constant", "model_lr_decay_epochs": 20},
    ],
)
def test_invalid_schedule_rejected(options):
    with pytest.raises(ValueError):
        StudentLearningConfig(**options)


def test_boundary_lr_refers_to_last_committed_update():
    config = StudentLearningConfig(model_lr_schedule="cosine_epochs")
    assert checkpoint_model_learning_rate(config, 0, 5) == model_learning_rate(config, 0)
    assert checkpoint_model_learning_rate(config, 5, 5) == model_learning_rate(config, 0)
    assert checkpoint_model_learning_rate(config, 6, 5) == model_learning_rate(config, 1)
    assert checkpoint_model_learning_rate(config, 10, 5) == model_learning_rate(config, 1)
    with pytest.raises(ValueError):
        model_learning_rate(config, -1)
    with pytest.raises(ValueError):
        checkpoint_model_learning_rate(config, 0, 0)


def test_setting_lr_retains_adam_moments_and_steps():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    parameter.square().backward()
    optimizer.step()
    before = copy.deepcopy(optimizer.state_dict())
    config = StudentLearningConfig(model_lr_schedule="cosine_epochs")
    apply_model_learning_rate(optimizer, config, 9)
    after = optimizer.state_dict()
    assert after["param_groups"][0]["lr"] == pytest.approx(0.0001)
    assert all(
        torch.equal(value, after["state"][key][name])
        for key, state in before["state"].items()
        for name, value in state.items()
    )


def test_scheduled_boundary_resume_matches_uninterrupted_updates(setup, tmp_path):
    original, sample, _ = setup
    config = replace(
        original.config,
        student_epochs=3,
        joint_epochs=0,
        kd_weight=0,
        checkpoint_every=1,
        model_lr_schedule="cosine_epochs",
        model_lr_decay_epochs=3,
    )

    def fresh():
        return StudentLearner(copy.deepcopy(original.model), config, original.routing)

    uninterrupted, partial = fresh(), fresh()
    run_student_training([sample], uninterrupted, {}, tmp_path / "full")
    first = run_student_training([sample], partial, {}, tmp_path / "partial", max_steps=2)
    payload = read_student_checkpoint(Path(first["checkpoint"]))
    assert payload["cursor"]["epoch"] == 2
    assert payload["model_optimizer"]["param_groups"][0]["lr"] == model_learning_rate(config, 1)
    restored = fresh()
    result = run_student_training(
        [sample], restored, {}, tmp_path / "resumed", resume=Path(first["checkpoint"])
    )
    assert result["schedule_complete"]
    assert all(
        torch.equal(value, restored.model.state_dict()[key])
        for key, value in uninterrupted.model.state_dict().items()
    )
    expected, actual = (
        uninterrupted.model_optimizer.state_dict(),
        restored.model_optimizer.state_dict(),
    )
    assert expected["param_groups"] == actual["param_groups"]
    assert all(
        torch.equal(value, actual["state"][key][name])
        for key, state in expected["state"].items()
        for name, value in state.items()
    )
    assert restored.optimizer_update_counts == {"model": 3, "actions": 0, "selector": 0}

    payload["model_optimizer"]["param_groups"][0]["lr"] = config.learning_rate
    bad = tmp_path / "corrupt-lr.pt"
    torch.save(payload, bad)
    untouched = fresh()
    before = copy.deepcopy(untouched.model.state_dict())
    with pytest.raises(ValueError, match="optimizer groups/config mismatch"):
        load_student_checkpoint(bad, untouched, {}, 1)
    assert all(torch.equal(v, untouched.model.state_dict()[k]) for k, v in before.items())


def test_frozen_mode_rejects_model_schedule(setup):
    learner, _, _ = setup
    config = replace(
        learner.config, student_epochs=0, joint_epochs=1, model_lr_schedule="cosine_epochs"
    )
    with pytest.raises(ValueError, match="constant model Adam"):
        StudentLearner(
            learner.model,
            config,
            learner.routing,
            intervention_config=InterventionConfig(
                schema="intervention_selection.v2", student_update_mode="frozen"
            ),
        )
