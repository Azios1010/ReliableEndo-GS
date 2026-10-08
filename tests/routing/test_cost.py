"""Cost calibration and action-specific feasibility contracts."""

import pytest
import torch

from reliable_endo_gs.actions.repair import Action
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.routing.cost import (
    HISTORY_SIZE,
    CostCalibrationPolicy,
    calibrate_action_costs,
    estimated_costs,
    new_cost_history,
    record_costs,
)
from reliable_endo_gs.routing.repairability import select_repairability


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
        binding={"train_manifest_hash": "train", "model_digest": "model", "action_digest": "action"},
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
        binding={"train_manifest_hash": "train", "model_digest": "model", "action_digest": "action"},
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
            binding={"train_manifest_hash": "changed", "model_digest": "model", "action_digest": "action"},
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
