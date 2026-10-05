"""Bounded/epoch Phase II execution with committed cursors and immutable logs."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

import torch

from reliable_endo_gs.actions.repair import RepairState
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.training.a04_reference import A04ReferenceTrainer, _move_batch
from reliable_endo_gs.training.action_checkpoint import (
    load_action_checkpoint,
    save_action_checkpoint,
)
from reliable_endo_gs.training.action_learning import ActionLearner
from reliable_endo_gs.training.harness import TrainingLog, TrainingSignals
from reliable_endo_gs.utils.io import write_json_atomic


def frozen_state(source: A04ReferenceTrainer, sample: ScaredA00Sample) -> RepairState:
    """Run the pinned A04 path in eval/no_grad, without deployment GT."""
    if sample.supervision is not None:
        raise ValueError("Phase II source must not load GT supervision")
    batch = _move_batch(sample.batch, source.device)
    source.model.eval().requires_grad_(False)
    target, _ = source._target(sample, batch)
    with torch.no_grad(), source._autocast_context():
        result = source.model(
            batch,
            target=target,
            sample_ids=(sample.sample_id,),
            data_role=sample.data_role,
            calibration_metadata=sample.calibration_metadata,
        )
    return RepairState(batch, result)


def known_skip_reason(state: RepairState, source: A04ReferenceTrainer) -> str | None:
    """Only skip demonstrably empty geometry or wholly near-clipped geometry."""
    valid = state.base.gaussians.valid_mask
    if not bool(valid.any()):
        return "empty_geometry_support"
    if not bool(state.support.any()):
        return "empty_deployment_support"
    visibility = state.base.render_left.visibility
    if visibility is not None and not bool((visibility > 0).any()):
        near = getattr(source.renderer, "metric_near_plane_m", None)
        z = state.base.gaussians.means3d[..., 2][valid]
        if isinstance(near, (float, int)) and bool((z <= near).all()):
            return "all_centres_before_near"
        raise ValueError("base has no visible primitives for an unexplained reason")
    return None


def run_action_training(
    source: A04ReferenceTrainer,
    learner: ActionLearner,
    identity: Mapping[str, object],
    output_dir: Path,
    *,
    max_steps: int | None = None,
    resume: Path | None = None,
    run_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Actions warm up first; fixed actions then guide selector learning.

    max_steps counts consumed samples across both stages. Epochs are total
    target epochs, not additional resume epochs. CUDA failures retain the last
    committed on-disk checkpoint; never serialize potentially damaged state.
    """
    if source.config.data_role != "final_train":
        raise ValueError("Phase II learning is restricted to dataset_1–7/final_train")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be positive")
    size = len(source._dataset)
    if size == 0:
        raise ValueError("empty Phase II training manifest")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "run.json", dict(run_metadata or {}))
    cursor = {
        "epoch": 0,
        "next_sample_index": 0,
        "samples_seen": 0,
        "optimizer_steps": 0,
        "skips": 0,
        "consecutive_skips": 0,
    }
    if resume is not None:
        cursor = load_action_checkpoint(resume, learner, identity, size)
    initial = output_dir / "initial.pt"
    save_action_checkpoint(initial, learner, identity, cursor)
    latest = initial
    config = learner.config
    total_epochs = config.action_epochs + config.selector_epochs
    if cursor["epoch"] > total_epochs:
        raise ValueError("resume cursor exceeds configured epoch schedule")
    started, consumed = time.perf_counter(), 0
    with TrainingLog(output_dir / "train.jsonl") as log, TrainingSignals() as signals:
        log.emit(
            {
                "event": "train_start",
                "identity": dict(identity),
                "cursor": dict(cursor),
                "routing_config_hash": config.config_hash,
                "samples_per_epoch": size,
                "schedule": {
                    "action_epochs": config.action_epochs,
                    "selector_epochs": config.selector_epochs,
                },
            }
        )
        try:
            while cursor["epoch"] < total_epochs:
                stage = "actions" if cursor["epoch"] < config.action_epochs else "selector"
                if signals.requested() or (max_steps is not None and consumed >= max_steps):
                    break
                index = cursor["next_sample_index"]
                sample = source._dataset[index]
                if sample.data_role != "final_train" or sample.sample_id.split("/")[0] not in {
                    f"dataset_{i}" for i in range(1, 8)
                }:
                    raise ValueError("forbidden sample in Phase II training")
                state = frozen_state(source, sample)
                skip = known_skip_reason(state, source)
                details: dict[str, object] = {}
                if skip is None:
                    if stage == "actions":
                        details = learner.warmup_step(state)
                    else:
                        details = learner.selector_step(state, cursor["samples_seen"])
                    cursor["optimizer_steps"] += 1
                    cursor["consecutive_skips"] = 0
                else:
                    cursor["skips"] += 1
                    cursor["consecutive_skips"] += 1
                cursor["samples_seen"] += 1
                cursor["next_sample_index"] += 1
                if cursor["next_sample_index"] == size:
                    cursor["epoch"] += 1
                    cursor["next_sample_index"] = 0
                consumed += 1
                log.emit(
                    {
                        "event": "sample_progress",
                        "stage": stage,
                        "sample_id": sample.sample_id,
                        "cursor": dict(cursor),
                        "skip_reason": skip,
                        "elapsed_seconds": time.perf_counter() - started,
                        "provenance": state.base.provenance_record.to_dict(),
                        **details,
                    }
                )
                if consumed % config.checkpoint_every == 0 or cursor["next_sample_index"] == 0:
                    latest = output_dir / f"sample-{cursor['samples_seen']:09d}.pt"
                    save_action_checkpoint(latest, learner, identity, cursor)
                if cursor["consecutive_skips"] >= config.max_consecutive_skips:
                    raise ValueError("consecutive known skips reached the configured limit")
        except BaseException as error:
            # No attempted CUDA recovery save; the last checkpoint is immutable
            # and contains parameters AND cursor from the same committed state.
            log.emit(
                {
                    "event": "train_failed",
                    "error": str(error),
                    "last_safe_checkpoint": str(latest),
                    "cursor": dict(cursor),
                }
            )
            raise
        destination = output_dir / ("final.pt" if cursor["epoch"] == total_epochs else "stopped.pt")
        save_action_checkpoint(destination, learner, identity, cursor)
        log.emit(
            {
                "event": "train_complete" if cursor["epoch"] == total_epochs else "train_stopped",
                "checkpoint": str(destination),
                "cursor": dict(cursor),
                "stop_reason": None
                if cursor["epoch"] == total_epochs
                else signals.requested() or ("max_steps" if max_steps else None),
            }
        )
    return {"checkpoint": str(destination), "cursor": cursor, "phase1_frozen": True}
