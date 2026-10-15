"""Execution preconditions, offline labels, repair gradients and timing support."""

import copy
from dataclasses import replace

import pytest
import torch

from reliable_endo_gs.actions import repair as repair_module
from reliable_endo_gs.actions.repair import Action
from reliable_endo_gs.regions.grid import grid_regions
from reliable_endo_gs.training.balanced_interventions import BalancedHarness
from reliable_endo_gs.training.interventions import InterventionHarness, intervention_costs
from reliable_endo_gs.training.repairability import RepairabilityHarness
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_learning import StudentLearner
from tests.routing.test_balanced import setup as balanced_fixture
from tests.routing.test_metric_joint import metric_learner
from tests.training.test_selector_context import context_learner

setup = balanced_fixture


def deny_actions(monkeypatch, denied, unavailable_states=None):
    original = repair_module.action_eligible

    def available(state, region, action):
        denied_here = action in denied and (
            unavailable_states is None or id(state) in unavailable_states
        )
        return not denied_here and original(state, region, action)

    for module in (
        "training.interventions",
        "training.repairability",
        "training.balanced_interventions",
        "training.balanced_learning",
        "training.student_learning",
        "training.action_learning",
        "actions.repair",
    ):
        monkeypatch.setattr(f"reliable_endo_gs.{module}.action_eligible", available)


def assert_state_equal(before, after):
    assert set(before) == set(after)
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, after[key])
        else:
            assert value == after[key]


@pytest.mark.parametrize("kind", ["photometric", "repairability", "balanced"])
@pytest.mark.parametrize("denied", [Action.STEREO, Action.GAUSSIAN])
def test_harness_skips_unavailable_actions_without_manufacturing_timing(
    setup, monkeypatch, kind, denied
):
    learner, sample, _ = setup
    harness = {
        "photometric": InterventionHarness(0.0),
        "repairability": RepairabilityHarness(learner.intervention_config),
        "balanced": BalancedHarness(learner.intervention_config),
    }[kind]
    deny_actions(monkeypatch, {denied})
    execute = learner.actions.execute
    seen = []

    def guarded(state, region, action, model):
        assert action != denied
        seen.append(action)
        return execute(state, region, action, model)

    monkeypatch.setattr(learner.actions, "execute", guarded)
    learner.actions.requires_grad_(False)
    with torch.no_grad():
        state = learner.state(sample)
        region = grid_regions(state.batch.left.shape[-2:], (6, 6), learner.routing.halo)[0]
        rows = harness.collect(state, region, learner.actions, learner.model)
    assert denied not in seen
    skipped = rows[int(denied)]
    assert skipped.action_available is False
    assert skipped.unavailable_reason == "action_precondition"
    assert skipped.cost_ms is None
    timings, available = intervention_costs(rows, state.batch.left)
    assert torch.isnan(timings[int(denied)])
    assert not bool(available[int(denied)])
    other = Action.GAUSSIAN if denied == Action.STEREO else Action.STEREO
    assert rows[int(other)].cost_ms > 0
    assert rows[int(other)].action_available is True


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize(
    "denied", [(Action.STEREO,), (Action.GAUSSIAN,), (Action.STEREO, Action.GAUSSIAN)]
)
def test_joint_and_frozen_skip_unavailable_repair_and_adam_state(
    setup, monkeypatch, tmp_path, frozen, denied
):
    learner, sample, teacher, truth = metric_learner(setup)
    if frozen:
        learner = StudentLearner(
            learner.model,
            learner.config,
            learner.routing,
            intervention_config=replace(learner.intervention_config, student_update_mode="frozen"),
        )
    # Prime both Adam states: an unavailable action must not drift via momentum.
    learner.step(sample, joint=True, teacher_target=teacher, metric_truth=truth, position=0)
    model_before = copy.deepcopy(learner.model.state_dict())
    model_adam_before = copy.deepcopy(learner.model_optimizer.state_dict())
    actions = {Action.STEREO: learner.actions.stereo, Action.GAUSSIAN: learner.actions.gaussian}
    weights_before = {
        action: copy.deepcopy(module.state_dict()) for action, module in actions.items()
    }
    adam_before = {
        parameter: copy.deepcopy(learner.action_optimizer.state[parameter])
        for action in denied
        for parameter in actions[action].parameters()
    }
    deny_actions(monkeypatch, set(denied))
    execute = learner.actions.execute

    def guarded(state, region, action, model):
        assert action not in denied
        return execute(state, region, action, model)

    monkeypatch.setattr(learner.actions, "execute", guarded)
    result = learner.step(
        sample, joint=True, teacher_target=teacher, metric_truth=truth, position=1
    )
    assert learner.action_update_counts == {
        "stereo": 1 if Action.STEREO in denied else 2,
        "gaussian": 1 if Action.GAUSSIAN in denied else 2,
    }
    assert learner.optimizer_update_counts["actions"] == (1 if len(denied) == 2 else 2)
    for action in denied:
        assert_state_equal(weights_before[action], actions[action].state_dict())
        for parameter in actions[action].parameters():
            assert parameter.grad is None
            assert_state_equal(adam_before[parameter], learner.action_optimizer.state[parameter])
    if frozen:
        assert_state_equal(model_before, learner.model.state_dict())
        assert learner.model_optimizer.state_dict() == model_adam_before
    for region in result["interventions"]:
        for row in region["actions"][1:]:
            if Action[row["action"]] in denied:
                assert row["action_available"] is False and row["action_executed"] is False
                assert row["action_unavailable_reason"] == "action_precondition"
                assert row["cost_ms"] is None and row["after"] is None
                assert row["selector_label_available"] is False
                assert row["metric_diagnostics"] is None
            else:
                assert row["action_available"] is True and row["action_executed"] is True
                assert row["cost_ms"] > 0
    # Existing history stores complete execution rows only; skipped actions
    # cannot contribute a fabricated zero-cost observation.
    assert learner.cost_observations == 1
    assert bool((learner.cost_history[1:, : learner.cost_history_observations] > 0).all())
    identity = dict(
        source_sha256="source",
        manifest_hash="train",
        role_hash="roles",
        renderer_identity="synthetic",
    )
    path = tmp_path / "partial-action.pt"
    save_student_checkpoint(
        path,
        learner,
        identity,
        dict(epoch=2, next_sample_index=0, samples_seen=2, optimizer_steps=2),
    )
    restored = StudentLearner(
        copy.deepcopy(learner.model),
        learner.config,
        learner.routing,
        intervention_config=learner.intervention_config,
    )
    load_student_checkpoint(path, restored, identity, 1)
    assert restored.action_update_counts == learner.action_update_counts
    assert restored.optimizer_update_counts == learner.optimizer_update_counts


def test_missing_gt_preserves_execution_availability_and_measured_timings(setup):
    learner, sample, teacher, truth = metric_learner(setup)
    truth = replace(truth, valid_mask=torch.zeros_like(truth.valid_mask))
    result = learner.step(sample, joint=True, teacher_target=teacher, metric_truth=truth)
    for region in result["interventions"]:
        for row in region["actions"][1:]:
            assert row["action_available"] is True and row["action_executed"] is True
            assert row["action_unavailable_reason"] is None
            assert row["selector_label_available"] is False
            assert row["metric_diagnostics"]["metric_unavailable_reasons"]
            assert row["cost_ms"] > 0
    assert learner.cost_observations == 1
    assert learner.action_update_counts == {"stereo": 1, "gaussian": 1}


def test_cost_history_ignores_incomplete_execution_rows(setup):
    learner, _, _ = setup
    learner.record_action_costs(
        torch.tensor([[0.0, float("nan"), 7.0], [0.0, 11.0, 13.0]]),
        torch.tensor([[True, False, True], [True, True, True]]),
    )
    assert learner.cost_observations == 1 and learner.cost_history_observations == 1
    assert torch.equal(learner.cost_max_ms, torch.tensor([0.0, 11.0, 13.0]))
    assert torch.equal(learner.cost_history[:, 0], learner.cost_max_ms)


def test_known_context_trains_selector_when_current_actions_cannot_execute(
    setup, tmp_path, monkeypatch
):
    learner, sample, teacher, truth, context, context_truth = context_learner(setup, fitted=False)
    learner.config = replace(learner.config, joint_epochs=5)
    for position in range(2):
        learner.step(
            sample, joint=True, teacher_target=teacher, metric_truth=truth, position=position
        )
    assert bool(learner.selector.normalization.fitted)
    unavailable_states = set()
    get_state = learner.state

    def state_for(sample_):
        state = get_state(sample_)
        if sample_.sample_id == sample.sample_id:
            unavailable_states.add(id(state))
        return state

    monkeypatch.setattr(learner, "state", state_for)
    deny_actions(monkeypatch, {Action.STEREO, Action.GAUSSIAN}, unavailable_states)
    for position in range(2, 5):
        result = learner.step(
            sample,
            joint=True,
            teacher_target=teacher,
            metric_truth=truth,
            selector_context=((context, context_truth),),
            position=position,
        )
        assert result["repair_executed_actions"] == 0
        assert result["selector_observed_regions"] == 0
        assert result["selector_snapshot_known_regions"] > 0
    assert learner.optimizer_update_counts == {"model": 5, "actions": 2, "selector": 6}
    assert learner.action_update_counts == {"stereo": 2, "gaussian": 2}
    identity = dict(
        source_sha256="source",
        manifest_hash="train",
        role_hash="roles",
        renderer_identity="synthetic",
    )
    path = tmp_path / "context-only-selector.pt"
    save_student_checkpoint(
        path,
        learner,
        identity,
        dict(epoch=5, next_sample_index=0, samples_seen=5, optimizer_steps=5),
    )
    restored = StudentLearner(
        copy.deepcopy(learner.model),
        learner.config,
        learner.routing,
        intervention_config=learner.intervention_config,
    )
    load_student_checkpoint(path, restored, identity, 1)
    assert restored.optimizer_update_counts == learner.optimizer_update_counts
    assert restored.action_update_counts == learner.action_update_counts
    assert_state_equal(learner.selector.state_dict(), restored.selector.state_dict())
