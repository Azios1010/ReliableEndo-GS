"""Immutable Phase II checkpoints, pinned upstream identity and exact cursor."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.reconstruction.checkpoint import _atomic_torch_save
from reliable_endo_gs.routing.cost import HISTORY_SIZE, new_cost_history
from reliable_endo_gs.training.action_learning import ActionLearner
from reliable_endo_gs.training.harness import capture_rng, checkpoint_rng, restore_rng


def save_action_checkpoint(
    path: Path, learner: ActionLearner, identity: Mapping[str, object], cursor: Mapping[str, int]
) -> None:
    """Atomically save Phase II weights/optimizers; never include Phase I weights."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    device = next(learner.selector.parameters()).device
    _atomic_torch_save(
        path,
        {
            "schema": "action_learning_checkpoint.v1",
            "identity": dict(identity),
            "routing_config_hash": learner.config.config_hash,
            "cursor": dict(cursor),
            "actions": learner.actions.state_dict(),
            "selector": learner.selector.state_dict(),
            "action_optimizer": learner.action_optimizer.state_dict(),
            "selector_optimizer": learner.selector_optimizer.state_dict(),
            "cost_max_ms": learner.cost_max_ms,
            "cost_observations": learner.cost_observations,
            "cost_history": learner.cost_history,
            "cost_history_observations": learner.cost_history_observations,
            "deployment_cost_calibration": None
            if learner.deployment_cost_calibration is None
            else learner.deployment_cost_calibration.to_payload(),
            "rng": capture_rng(device),
            "phase1_in_checkpoint": False,
        },
    )


def load_action_checkpoint(
    path: Path, learner: ActionLearner, identity: Mapping[str, object], samples_per_epoch: int
) -> dict[str, int]:
    """Reject config/source/data/base drift before loading any learned weights."""
    raw: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping) or raw.get("schema") != "action_learning_checkpoint.v1":
        raise ValueError("unsupported Phase II checkpoint")
    if (
        raw.get("identity") != dict(identity)
        or raw.get("routing_config_hash") != learner.config.config_hash
    ):
        raise ValueError("Phase II checkpoint identity/config mismatch")
    if raw.get("phase1_in_checkpoint") is not False:
        raise ValueError("Phase I must remain external and immutable")
    cursor = raw.get("cursor")
    names = {
        "epoch",
        "next_sample_index",
        "samples_seen",
        "optimizer_steps",
        "skips",
        "consecutive_skips",
    }
    if (
        not isinstance(cursor, dict)
        or set(cursor) != names
        or any(type(x) is not int or x < 0 for x in cursor.values())
    ):
        raise ValueError("invalid Phase II cursor")
    if (
        cursor["next_sample_index"] >= samples_per_epoch
        or cursor["samples_seen"]
        != cursor["epoch"] * samples_per_epoch + cursor["next_sample_index"]
        or cursor["optimizer_steps"] + cursor["skips"] != cursor["samples_seen"]
        or cursor["consecutive_skips"] > cursor["skips"]
    ):
        raise ValueError("inconsistent Phase II cursor")
    costs = raw.get("cost_max_ms")
    observations = raw.get("cost_observations")
    if (
        not isinstance(costs, torch.Tensor)
        or costs.shape != (3,)
        or not bool(torch.isfinite(costs).all() & (costs >= 0).all())
        or costs[0] != 0
        or type(observations) is not int
        or observations < 0
    ):
        raise ValueError("invalid Phase II cost calibration")
    history = raw.get("cost_history")
    history_observations = raw.get("cost_history_observations", 0)
    if history is not None and (
        not isinstance(history, torch.Tensor)
        or history.shape != (3, HISTORY_SIZE)
        or not bool(torch.isfinite(history).all() & (history >= 0).all())
        or type(history_observations) is not int
        or history_observations < 0
        or history_observations > observations
    ):
        raise ValueError("invalid Phase II cost history")
    calibration_raw = raw.get("deployment_cost_calibration")
    calibration = None
    if calibration_raw is not None:
        from reliable_endo_gs.routing.cost import DeploymentCostCalibration

        if not isinstance(calibration_raw, Mapping):
            raise ValueError("invalid Phase II deployment cost calibration")
        calibration = DeploymentCostCalibration.from_payload(calibration_raw)
        if not calibration.matches_training_identity(identity):
            raise ValueError("Phase II deployment cost calibration training identity mismatch")
    device = next(learner.selector.parameters()).device
    rng = checkpoint_rng(raw.get("rng"))
    previous = capture_rng(device)
    try:
        restore_rng(rng, device)
    finally:
        restore_rng(previous, device)
    learner.actions.load_state_dict(raw["actions"], strict=True)
    learner.selector.load_state_dict(raw["selector"], strict=True)
    learner.action_optimizer.load_state_dict(raw["action_optimizer"])
    learner.selector_optimizer.load_state_dict(raw["selector_optimizer"])
    learner.cost_max_ms = costs.to(device)
    learner.cost_observations = observations
    learner.cost_history = new_cost_history(device) if history is None else history.to(device)
    learner.cost_history_observations = history_observations
    learner.deployment_cost_calibration = calibration
    restore_rng(rng, device)
    return dict(cursor)
