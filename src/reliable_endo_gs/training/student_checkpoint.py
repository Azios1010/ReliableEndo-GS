"""Student/joint checkpoints with explicit schema and strict resume compatibility."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.reconstruction.checkpoint import _atomic_torch_save, _cpu_tree
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.cost import HISTORY_SIZE, new_cost_history
from reliable_endo_gs.training.harness import capture_rng, checkpoint_rng, restore_rng
from reliable_endo_gs.training.student_learning import StudentLearner

SCHEMA = "student_joint_checkpoint.v1"
SELECTION_SCHEMA = "student_joint_intervention_checkpoint.v1"
BALANCED_SCHEMA = "student_joint_intervention_checkpoint.v2"
MASK_SCHEMA = "student_joint_mask_intervention_checkpoint.v1"


def _schema(learner: StudentLearner) -> str:
    return (
        MASK_SCHEMA
        if learner.mask_config is not None
        else BALANCED_SCHEMA
        if isinstance(learner.selector, BalancedSelector)
        else SCHEMA
        if learner.intervention_config is None
        else SELECTION_SCHEMA
    )


def save_student_checkpoint(
    path: Path, learner: StudentLearner, identity: Mapping[str, object], cursor: Mapping[str, int]
) -> None:
    """Save owned weights, all Adam states, measured costs and sample-boundary RNG."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    calibration = learner.deployment_cost_calibration
    _atomic_torch_save(
        path,
        _cpu_tree(
            {
                "schema": _schema(learner),
                "mask_config": None if learner.mask_config is None else asdict(learner.mask_config),
                "target_mask_index_sha256": None
                if learner.target_masks is None
                else learner.target_masks.identity,
                "intervention_config": None
                if learner.intervention_config is None
                else learner.intervention_config.to_dict(),
                "model_gradient_balance": None
                if learner.objective_balancer is None
                else learner.objective_balancer.state_dict(),
                "selector_gradient_balance": None
                if learner.selector_balancer is None
                else learner.selector_balancer.state_dict(),
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
                "cost_history": learner.cost_history,
                "cost_history_observations": learner.cost_history_observations,
                "deployment_cost_calibration": None
                if calibration is None
                else calibration.to_payload(),
                "rng": capture_rng(next(learner.model.parameters()).device),
                "teacher_in_checkpoint": False,
                "reference_in_checkpoint": False,
            }
        ),
    )


def read_student_checkpoint(path: Path) -> dict[str, Any]:
    """Weights-only deserialization; frozen-reference checkpoints are incompatible."""
    raw: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or raw.get("schema") not in (
        SCHEMA,
        SELECTION_SCHEMA,
        BALANCED_SCHEMA,
        MASK_SCHEMA,
    ):
        raise ValueError("unsupported student checkpoint schema")
    return raw


def initialize_student_model(path: Path, learner: StudentLearner) -> dict[str, str]:
    """Explicit weights-only warm start; new repairs, selector, RNG and Adam.

    It is a new experiment, never a resume. Scientific construction and model
    tensor shapes must match. Historical source identity is recorded separately.
    """
    import hashlib

    raw = read_student_checkpoint(path)
    if raw.get("construction_config") != learner.model.config.to_dict():
        raise ValueError("warm-start scientific construction mismatch")
    architecture = raw.get("student_config")
    if not isinstance(architecture, dict) or any(
        architecture.get(name) != getattr(learner.config, name)
        for name in ("width", "bins", "minimum_effective_px")
    ):
        raise ValueError("warm-start student architecture mismatch")
    if (
        raw.get("teacher_in_checkpoint") is not False
        or raw.get("reference_in_checkpoint") is not False
    ):
        raise ValueError("warm-start checkpoint ownership mismatch")
    state = raw.get("model")
    expected = learner.model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("warm-start student parameter keys mismatch")
    for key, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected[key].shape
            or value.dtype != expected[key].dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("warm-start student tensor mismatch")
    learner.model.load_state_dict(state, strict=True)
    return {
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": str(raw["identity"]["source_sha256"]),
    }


def import_frozen_student_actions(
    path: Path, learner: StudentLearner, *, expected_sha256: str
) -> dict[str, object]:
    """Pinned historical weights for a NEW frozen intervention experiment.

    This is not resume or production deployment. Model/action construction is
    exact; no selector, optimizer, RNG, normalizer or cost states are imported.
    The evaluator must record BOTH the historical and current source identities.
    Normal checkpoint source-compatibility checks remain unchanged.
    """
    import hashlib

    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("frozen intervention checkpoint SHA-256 mismatch")
    raw = read_student_checkpoint(path)
    protocol = (
        None if learner.intervention_config is None else learner.intervention_config.to_dict()
    )
    if (
        raw.get("student_config") != asdict(learner.config)
        or raw.get("routing_config") != asdict(learner.routing)
        or raw.get("intervention_config") != protocol
        or raw.get("identity", {}).get("run_kind") == "bounded_smoke"
    ):
        raise ValueError("frozen intervention config/protocol mismatch")
    state = raw.get("actions")
    expected = learner.actions.state_dict()
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
        raise ValueError("invalid frozen action weights")
    imported = initialize_student_model(path, learner)
    learner.actions.load_state_dict(state, strict=True)
    for module in (learner.model, learner.actions):
        module.eval().requires_grad_(False)
    return {
        **imported,
        "mode": "new_frozen_intervention_experiment",
        "selector_imported": False,
        "optimizer_imported": False,
    }


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
    expected_schema = _schema(learner)
    expected_intervention = (
        None if learner.intervention_config is None else learner.intervention_config.to_dict()
    )
    if (
        raw.get("schema") != expected_schema
        or raw.get("intervention_config") != expected_intervention
        or raw.get("mask_config")
        != (None if learner.mask_config is None else asdict(learner.mask_config))
        or (
            resume
            and raw.get("target_mask_index_sha256")
            != (None if learner.target_masks is None else learner.target_masks.identity)
        )
    ):
        raise ValueError("intervention checkpoint protocol/architecture mismatch")
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
    history = raw.get("cost_history")
    history_observations = raw.get("cost_history_observations", 0)
    if history is not None and (
        not isinstance(history, torch.Tensor)
        or history.shape != (3, HISTORY_SIZE)
        or not bool(torch.isfinite(history).all() & (history >= 0).all())
        or type(history_observations) is not int
        or history_observations < 0
    ):
        raise ValueError("invalid student cost history")
    calibration_raw = raw.get("deployment_cost_calibration")
    calibration = None
    if calibration_raw is not None:
        from reliable_endo_gs.routing.cost import DeploymentCostCalibration

        if not isinstance(calibration_raw, Mapping):
            raise ValueError("invalid student deployment cost calibration")
        calibration = DeploymentCostCalibration.from_payload(calibration_raw)
        if not calibration.matches_training_identity(identity):
            raise ValueError("student deployment cost calibration training identity mismatch")
    modules = (learner.model, learner.actions, learner.selector)
    for name, balancer in (
        ("model_gradient_balance", learner.objective_balancer),
        ("selector_gradient_balance", learner.selector_balancer),
    ):
        payload = raw.get(name)
        if balancer is None:
            if payload is not None:
                raise ValueError("unexpected gradient balance state")
        else:
            expected = balancer.state_dict()
            if (
                not isinstance(payload, dict)
                or set(payload) != set(expected)
                or any(
                    not isinstance(v, torch.Tensor)
                    or v.shape != expected[k].shape
                    or v.dtype != expected[k].dtype
                    or not bool(torch.isfinite(v).all())
                    for k, v in payload.items()
                )
            ):
                raise ValueError("invalid gradient balance checkpoint state")
            probe = copy.deepcopy(balancer)
            probe.load_state_dict(payload, strict=True)
            probe.validate_state()
            if int(probe.observations) > cursor["optimizer_steps"]:
                raise ValueError("gradient balance cursor mismatch")
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
    if isinstance(learner.selector, BalancedSelector):
        probe_selector = copy.deepcopy(learner.selector)
        probe_selector.load_state_dict(raw["selector"], strict=True)
        probe_selector.validate_state()
        if int(probe_selector.normalization.counts[2]) > observations:
            raise ValueError("normalization cursor mismatch")
        if not resume and not bool(probe_selector.normalization.fitted):
            raise ValueError("deployment requires fitted training normalization")
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
    for name, balancer in (
        ("model_gradient_balance", learner.objective_balancer),
        ("selector_gradient_balance", learner.selector_balancer),
    ):
        if balancer is not None:
            balancer.load_state_dict(raw[name], strict=True)
    if resume:
        for name, optimizer in zip(
            ("model_optimizer", "action_optimizer", "selector_optimizer"), optimizers, strict=True
        ):
            optimizer.load_state_dict(raw[name])
        restore_rng(rng, device)
    learner.cost_max_ms = costs.to(device)
    learner.cost_observations = observations
    learner.cost_history = (
        new_cost_history(device) if history is None else history.to(device)
    )
    learner.cost_history_observations = history_observations
    learner.deployment_cost_calibration = None
    if calibration is not None:
        learner.attach_cost_calibration(calibration)
    return dict(cursor)
