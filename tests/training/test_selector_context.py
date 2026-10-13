"""Fresh current-actor supervision: isolation, optimizer ownership and resume."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.config.intervention import InterventionConfig, load_intervention_config
from reliable_endo_gs.data.scared_a00 import ScaredSupervision
from reliable_endo_gs.training import balanced_learning
from reliable_endo_gs.training.student_learning import StudentLearner
from reliable_endo_gs.training.student_run import run_student_training
from tests.routing.test_balanced import setup as balanced_fixture
from tests.routing.test_metric_joint import metric_learner

setup = balanced_fixture


def context_learner(setup, *, fitted=True):
    old, sample, teacher, truth = metric_learner(setup)
    cfg = replace(
        old.intervention_config,
        selector_context_frames=3,
        selector_updates_per_snapshot=2,
        geometry_supervision_weight=1,
    )
    learner = StudentLearner(old.model, old.config, old.routing, intervention_config=cfg)
    with torch.no_grad():
        state = learner.state(sample)
        features = learner.selection_features(
            state, learner.candidate_regions(state, sample.sample_id)
        )
    labels = features.new_ones((features.shape[0], 3))
    labels[:, 0] = 0
    for _ in range(2 if fitted else 0):
        learner.selector.normalization.observe(
            features, labels, labels, sample_id=sample.sample_id, role="final_train"
        )
    context = replace(sample, sample_id="dataset_2/keyframe_1/frame_data000001")
    context_truth = replace(truth, sample_id=context.sample_id)
    return learner, sample, teacher, truth, context, context_truth


def test_context_labels_use_current_actor_without_gradients_and_restore_owners(setup, monkeypatch):
    learner, sample, teacher, truth, context, context_truth = context_learner(setup)
    original = learner.harness.collect
    calls = []

    def collect(state, region, actions, reference, *, metric_truth):
        assert all(not p.requires_grad for p in actions.parameters())
        assert reference is learner.model and actions is learner.actions
        assert metric_truth.disparity.grad is None
        calls.append(
            (metric_truth.sample_id, any(p.grad is not None for p in actions.parameters()))
        )
        return original(state, region, actions, reference, metric_truth=metric_truth)

    monkeypatch.setattr(learner.harness, "collect", collect)
    before = {
        name: copy.deepcopy(getattr(learner, name).state_dict())
        for name in ("model", "actions", "selector")
    }
    result = learner.step(
        sample,
        joint=True,
        teacher_target=teacher,
        metric_truth=truth,
        selector_context=((context, context_truth),),
        position=9,
    )
    assert result["selector_snapshot_frames"] == 2
    assert {identity for identity, _ in calls} == {sample.sample_id, context.sample_id}
    assert all(not has_grad for _, has_grad in calls)
    extra = result["selector_snapshot_context"][0]
    assert extra["sample_id"] == context.sample_id and extra["snapshot_optimizer_step"] == 9
    assert learner.optimizer_update_counts == {"model": 1, "actions": 1, "selector": 2}
    for name, state in before.items():
        owner = getattr(learner, name)
        assert any(not torch.equal(value, owner.state_dict()[key]) for key, value in state.items())
        assert owner.training and all(p.requires_grad for p in owner.parameters())
    assert sample.supervision is None and context.supervision is None
    assert teacher.disparity.grad is None


def test_unknown_current_truth_does_not_discard_known_context(setup):
    learner, sample, teacher, truth, context, context_truth = context_learner(setup)
    truth = replace(truth, valid_mask=torch.zeros_like(truth.valid_mask))
    result = learner.step(
        sample,
        joint=True,
        teacher_target=teacher,
        metric_truth=truth,
        selector_context=((context, context_truth),),
    )
    assert result["selector_observed_regions"] == 0
    assert result["selector_snapshot_known_regions"] > 0
    assert result["selector_updates_this_snapshot"] == 2


@pytest.mark.parametrize("change", ["duplicate", "identity", "test", "gt", "count", "warmup"])
def test_context_rejects_unverified_or_forbidden_inputs_before_update(setup, change):
    learner, sample, teacher, truth, context, context_truth = context_learner(setup)
    contexts = ((context, context_truth),)
    joint = True
    if change == "duplicate":
        contexts = ((sample, truth),)
    elif change == "identity":
        contexts = ((context, truth),)
    elif change == "test":
        contexts = ((replace(context, data_role="final_test"), context_truth),)
    elif change == "gt":
        contexts = (
            (
                replace(
                    context,
                    supervision=ScaredSupervision(
                        context_truth.disparity, context_truth.valid_mask
                    ),
                ),
                context_truth,
            ),
        )
    elif change == "count":
        contexts *= 3
    else:
        joint = False
    with pytest.raises(ValueError):
        learner.step(
            sample,
            joint=joint,
            teacher_target=teacher,
            metric_truth=truth,
            selector_context=contexts,
        )
    assert learner.optimizer_update_counts == {"model": 0, "actions": 0, "selector": 0}


def test_exception_during_context_collection_restores_train_and_gradient_flags(setup, monkeypatch):
    learner, sample, teacher, truth, context, context_truth = context_learner(setup)
    original = learner.harness.collect

    def collect(*args, **kwargs):
        if kwargs["metric_truth"].sample_id == context.sample_id:
            raise RuntimeError("injected context failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(learner.harness, "collect", collect)
    with pytest.raises(RuntimeError, match="injected context"):
        learner.step(
            sample,
            joint=True,
            teacher_target=teacher,
            metric_truth=truth,
            selector_context=((context, context_truth),),
        )
    assert learner.model.training and learner.actions.training
    assert all(p.requires_grad for p in learner.actions.parameters())
    assert learner.optimizer_update_counts == {"model": 0, "actions": 0, "selector": 0}


def test_fresh_context_inputs_are_deterministic_across_resume(setup, tmp_path, monkeypatch):
    learner, sample, teacher, truth, _, _ = context_learner(setup, fitted=False)
    learner.config = replace(learner.config, joint_epochs=1, checkpoint_every=1)
    samples = [
        replace(sample, sample_id=f"dataset_1/keyframe_1/frame_data{i:06d}") for i in range(3)
    ]
    observed = []
    original = balanced_learning.balanced_step

    def step(*args, **kwargs):
        observed.append(
            (args[1].sample_id, tuple(s.sample_id for s, _ in kwargs.get("selector_context", ())))
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(balanced_learning, "balanced_step", step)

    def offline(sample):
        return replace(truth, sample_id=sample.sample_id)

    first = run_student_training(
        samples,
        learner,
        {},
        tmp_path / "first",
        max_steps=2,
        target_provider=lambda _: teacher,
        metric_truth_provider=offline,
    )
    result = run_student_training(
        samples,
        learner,
        {},
        tmp_path / "resume",
        resume=Path(first["checkpoint"]),
        target_provider=lambda _: teacher,
        metric_truth_provider=offline,
    )
    assert result["schedule_complete"]
    assert observed == [
        (samples[i].sample_id, tuple(s.sample_id for s in samples[max(0, i - 2) : i]))
        for i in range(3)
    ]
    assert learner.optimizer_update_counts == {"model": 3, "actions": 3, "selector": 2}


def test_context_config_defaults_preserve_historical_identity():
    old = load_intervention_config(Path("configs/routing/intervention_metric_action_heads.yaml"))
    assert old.config_hash == "8fb6905c83f04b6bfbefef65a1b7ec4273cf0f5de6fe4dfa16ac2bf5fb530eec"
    assert "selector_context_frames" not in old.to_dict()
    cfg = replace(old, selector_context_frames=8)
    assert InterventionConfig(**cfg.to_dict()) == cfg and cfg.config_hash != old.config_hash
    for count in (0, 17, True, 1.5):
        with pytest.raises(ValueError):
            replace(old, selector_context_frames=count)
