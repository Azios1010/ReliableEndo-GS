"""Cost calibration and action-specific feasibility contracts."""

import pytest
import torch

from reliable_endo_gs.actions.repair import Action
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.routing.cost import (
    HISTORY_SIZE,
    CostCalibrationPolicy,
    calibrate_action_costs,
    capture_runtime_identity,
    estimated_costs,
    load_cost_calibration,
    new_cost_history,
    record_costs,
    save_cost_calibration,
)
from reliable_endo_gs.routing.repairability import select_repairability


@pytest.mark.parametrize("explicit_runtime", [False, True])
@pytest.mark.parametrize("tail_policy,quantile", [("max", None), ("quantile", 0.75)])
def test_captured_runtime_cost_artifact_weights_only_roundtrip(
    tmp_path, monkeypatch, explicit_runtime, tail_policy, quantile
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = capture_runtime_identity("cpu")
    assert type(runtime["torch"]) is str
    artifact = calibrate_action_costs(
        {0: lambda: None, 1: lambda: None, 2: lambda: None},
        binding={"collection_role": "train", "model_digest": "synthetic"},
        shape_hw=(4, 4),
        policy=CostCalibrationPolicy(tail_policy, quantile, 0, 2),
        runtime=runtime if explicit_runtime else None,
    )
    path = tmp_path / "cost.pt"
    save_cost_calibration(path, artifact)
    # Exercise the actual weights-only loader, without adding safe globals.
    loaded = load_cost_calibration(path)
    assert loaded.runtime == runtime
    assert loaded.binding == artifact.binding
    assert loaded.policy == artifact.policy
    torch.testing.assert_close(loaded.timings_ms, artifact.timings_ms, rtol=0, atol=0)
    torch.testing.assert_close(loaded.estimated_ms, artifact.estimated_ms, rtol=0, atol=0)


def test_training_history_is_diagnostic_and_does_not_replace_max() -> None:
    history = new_cost_history("cpu")
    observations = 0
    maximum = torch.zeros(3)
    for costs in (
        torch.tensor([0.0, 2635.0, 100.0]),
        torch.tensor([0.0, 110.0, 100.0]),
        torch.tensor([0.0, 109.0, 101.0]),
    ):
        maximum = torch.maximum(maximum, costs)
        observations = record_costs(history, costs, observations)
    estimate = estimated_costs(
        history,
        maximum,
        observations,
        history_observations=observations,
    )
    assert estimate[1] == 2635.0
    assert maximum[1] == 2635.0


def test_explicit_deployment_artifact_uses_declared_tail_policy() -> None:
    policy = CostCalibrationPolicy(
        tail_policy="max", quantile=None, warmup_repeats=1, measure_repeats=3
    )
    artifact = calibrate_action_costs(
        {
            0: lambda: None,
            1: lambda: torch.zeros(16),
            2: lambda: torch.zeros(16),
        },
        binding={
            "train_manifest_hash": "train",
            "model_digest": "model",
            "action_digest": "action",
        },
        runtime={"device": "cpu", "cuda_launch_blocking": None},
        shape_hw=(4, 4),
        policy=policy,
    )
    assert artifact.estimated_ms.shape == (3,)
    assert artifact.estimated_ms[0] == 0
    assert artifact.matches(
        {"train_manifest_hash": "train", "model_digest": "model", "action_digest": "action"},
        {"device": "cpu", "cuda_launch_blocking": None},
        (4, 4),
    )
    costs = estimated_costs(
        torch.zeros(3, HISTORY_SIZE),
        torch.tensor([0.0, 2635.0, 100.0]),
        observations=3,
        calibration=artifact,
        binding={
            "train_manifest_hash": "train",
            "model_digest": "model",
            "action_digest": "action",
        },
        runtime={"device": "cpu", "cuda_launch_blocking": None},
        shape_hw=(4, 4),
    )
    torch.testing.assert_close(costs, artifact.estimated_ms)
    with pytest.raises(ValueError, match="mismatch"):
        estimated_costs(
            torch.zeros(3, HISTORY_SIZE),
            torch.tensor([0.0, 1.0, 1.0]),
            observations=3,
            calibration=artifact,
            binding={
                "train_manifest_hash": "changed",
                "model_digest": "model",
                "action_digest": "action",
            },
            runtime={"device": "cpu", "cuda_launch_blocking": None},
            shape_hw=(4, 4),
        )


def test_empty_history_falls_back_without_fabricating_samples() -> None:
    history = new_cost_history("cpu")
    maximum = torch.tensor([0.0, 12.0, 13.0])
    estimate = estimated_costs(history, maximum, observations=1)
    torch.testing.assert_close(estimate, maximum)


def test_history_is_bounded_and_preserves_action_order() -> None:
    history = new_cost_history("cpu")
    observations = 0
    for index in range(HISTORY_SIZE + 3):
        observations = record_costs(
            history,
            torch.tensor([0.0, float(index + 1), float(index + 2)]),
            observations,
        )
    assert observations == HISTORY_SIZE + 3
    assert history.shape == (3, HISTORY_SIZE)
    assert bool((history[0] == 0).all())


def test_action_specific_validity_disables_unknown_source() -> None:
    config = InterventionConfig()
    decision = select_repairability(
        torch.tensor([[0.0, 1.0, 0.5]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 1.0]]),
        torch.tensor([[False, False, True]]),
        10.0,
        config,
    )
    assert decision.action == Action.GAUSSIAN
