"""Student/joint checkpoints with explicit schema and strict resume compatibility."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.reconstruction.checkpoint import _atomic_torch_save, _cpu_tree
from reliable_endo_gs.training.harness import capture_rng, checkpoint_rng, restore_rng
from reliable_endo_gs.training.student_learning import StudentLearner

SCHEMA = "student_joint_checkpoint.v1"


def save_student_checkpoint(
    path: Path, learner: StudentLearner, identity: Mapping[str, object], cursor: Mapping[str, int]
) -> None:
    """Save owned weights, all Adam states, measured costs and sample-boundary RNG."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    _atomic_torch_save(
        path,
        _cpu_tree(
            {
                "schema": SCHEMA,
                "identity": dict(identity),
                "cursor": dict(cursor),
                "student_config": asdict(learner.config),
                "routing_config": asdict(learner.routing),
                "construction_config": learner.model.config.to_dict(),
                "hashes": [
                    learner.config.config_hash,
                    learner.routing.config_hash,
                    learner.model.config.config_hash,
                ],
                "model": learner.model.state_dict(),
                "actions": learner.actions.state_dict(),
                "selector": learner.selector.state_dict(),
                "model_optimizer": learner.model_optimizer.state_dict(),
                "action_optimizer": learner.action_optimizer.state_dict(),
                "selector_optimizer": learner.selector_optimizer.state_dict(),
                "cost_max_ms": learner.cost_max_ms,
                "cost_observations": learner.cost_observations,
                "rng": capture_rng(next(learner.model.parameters()).device),
                "teacher_in_checkpoint": False,
                "reference_in_checkpoint": False,
            }
        ),
    )


def read_student_checkpoint(path: Path) -> dict[str, Any]:
    """Weights-only deserialization; frozen-reference checkpoints are incompatible."""
    raw: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError("unsupported student checkpoint schema")
    return raw


def load_student_checkpoint(
    path: Path,
    learner: StudentLearner,
    identity: Mapping[str, object],
    samples_per_epoch: int,
    *,
    resume: bool = True,
) -> dict[str, int]:
    """Validate compatibility and payload on copies before mutating live parameters.

    Resume requires the entire recorded identity. Deployment permits different
    hardware but pins code, data/roles, and renderer. RNG/Adam are not changed
    during deployment load. No external teacher/reference checkpoint is loaded.
    """
    raw = read_student_checkpoint(path)
    hashes = [
        learner.config.config_hash,
        learner.routing.config_hash,
        learner.model.config.config_hash,
    ]
    if (
        raw.get("hashes") != hashes
        or raw.get("teacher_in_checkpoint") is not False
        or raw.get("reference_in_checkpoint") is not False
    ):
        raise ValueError("student checkpoint config/ownership mismatch")
    if (
        raw.get("student_config") != asdict(learner.config)
        or raw.get("routing_config") != asdict(learner.routing)
        or raw.get("construction_config") != learner.model.config.to_dict()
    ):
        raise ValueError("student checkpoint serialized config mismatch")
    recorded = raw.get("identity")
    if not isinstance(recorded, dict):
        raise ValueError("missing student checkpoint identity")
    if resume:
        compatible = recorded == dict(identity)
    else:
        keys = ("source_sha256", "manifest_hash", "role_hash", "renderer_identity")
        compatible = all(
            k in identity and k in recorded and identity[k] == recorded[k] for k in keys
        )
        if recorded.get("run_kind") == "bounded_smoke":
            raise ValueError("smoke checkpoints cannot be deployed as full-run artifacts")
    if not compatible:
        raise ValueError("student checkpoint identity mismatch")
    cursor = raw.get("cursor")
    if (
        type(samples_per_epoch) is not int
        or samples_per_epoch < 1
        or not isinstance(cursor, dict)
        or set(cursor) != {"epoch", "next_sample_index", "samples_seen", "optimizer_steps"}
        or any(type(x) is not int or x < 0 for x in cursor.values())
    ):
        raise ValueError("invalid student cursor")
    total = learner.config.student_epochs + learner.config.joint_epochs
    if (
        cursor["next_sample_index"] >= samples_per_epoch
        or cursor["epoch"] > total
        or (cursor["epoch"] == total and cursor["next_sample_index"] != 0)
        or cursor["samples_seen"]
        != cursor["epoch"] * samples_per_epoch + cursor["next_sample_index"]
        or cursor["optimizer_steps"] != cursor["samples_seen"]
    ):
        raise ValueError("inconsistent student cursor")
    costs, observations = raw.get("cost_max_ms"), raw.get("cost_observations")
    if (
        not isinstance(costs, torch.Tensor)
        or costs.shape != (3,)
        or not bool(torch.isfinite(costs).all() & (costs >= 0).all())
        or costs[0] != 0
        or type(observations) is not int
        or observations < 0
        or observations > cursor["samples_seen"]
    ):
        raise ValueError("invalid student cost calibration")
    modules = (learner.model, learner.actions, learner.selector)
    for name, module in zip(("model", "actions", "selector"), modules, strict=True):
        state = raw.get(name)
        expected = module.state_dict()
        if (
            not isinstance(state, dict)
            or set(state) != set(expected)
            or any(
                not isinstance(v, torch.Tensor)
                or v.shape != expected[k].shape
                or v.dtype != expected[k].dtype
                or not bool(torch.isfinite(v).all())
                for k, v in state.items()
            )
        ):
            raise ValueError(f"invalid student checkpoint weights: {name}")
    optimizers = (learner.model_optimizer, learner.action_optimizer, learner.selector_optimizer)
    # Loading into a temporary optimizer detects group/ownership incompatibility
    # before any live weight has changed. Tensor states must match each parameter.
    for name, optimizer in zip(
        ("model_optimizer", "action_optimizer", "selector_optimizer"), optimizers, strict=True
    ):
        payload = raw.get(name)
        if not isinstance(payload, dict) or set(payload) != {"state", "param_groups"}:
            raise ValueError(f"invalid student optimizer schema: {name}")
        expected_groups = optimizer.state_dict()["param_groups"]
        if not isinstance(payload["state"], dict) or payload["param_groups"] != expected_groups:
            raise ValueError(f"student optimizer groups/config mismatch: {name}")
        group_ids = [index for group in payload["param_groups"] for index in group["params"]]
        if len(group_ids) != len(set(group_ids)) or set(payload["state"]) - set(group_ids):
            raise ValueError(f"invalid student optimizer ownership: {name}")
        test_optimizer = copy.deepcopy(optimizer)
        try:
            test_optimizer.load_state_dict(raw[name])
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError(f"invalid student optimizer: {name}") from error
        expected_steps = cursor["optimizer_steps"] if name == "model_optimizer" else observations
        if len(test_optimizer.state) != (len(group_ids) if expected_steps else 0):
            raise ValueError(f"incomplete student optimizer parameter state: {name}")
        for parameter, state in test_optimizer.state.items():
            if not isinstance(parameter, torch.Tensor) or set(state) != {
                "step",
                "exp_avg",
                "exp_avg_sq",
            }:
                raise ValueError(f"incomplete student Adam state: {name}")
            step = state["step"]
            if (
                not isinstance(step, torch.Tensor)
                or step.numel() != 1
                or not bool(torch.isfinite(step).all())
                or float(step) != expected_steps
            ):
                raise ValueError(f"invalid student Adam step: {name}")
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state and (
                    not isinstance(state[key], torch.Tensor)
                    or state[key].shape != parameter.shape
                    or not bool(torch.isfinite(state[key]).all())
                ):
                    raise ValueError(f"invalid student optimizer moments: {name}")
    device = next(learner.model.parameters()).device
    rng = checkpoint_rng(raw.get("rng"))
    if resume:
        previous = capture_rng(device)
        try:
            restore_rng(rng, device)
        finally:
            restore_rng(previous, device)
    for name, module in zip(("model", "actions", "selector"), modules, strict=True):
        module.load_state_dict(raw[name], strict=True)
    if resume:
        for name, optimizer in zip(
            ("model_optimizer", "action_optimizer", "selector_optimizer"), optimizers, strict=True
        ):
            optimizer.load_state_dict(raw[name])
        restore_rng(rng, device)
    learner.cost_max_ms = costs.to(device)
    learner.cost_observations = observations
    return dict(cursor)
