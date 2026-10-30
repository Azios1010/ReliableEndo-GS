"""Sequential student/joint execution with exact cursors and immutable outputs."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.teachers import AlignedTeacherTarget
from reliable_endo_gs.training.harness import TrainingLog, TrainingSignals
from reliable_endo_gs.training.metric_quality import OfflineMetricTruth
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    restore_model_initialization,
    save_student_checkpoint,
    validate_model_initialization,
)
from reliable_endo_gs.training.student_learning import StudentLearner
from reliable_endo_gs.training.student_schedule import apply_model_learning_rate
from reliable_endo_gs.utils.io import write_json_atomic


class StudentSamples(Protocol):
    """Ordered indexable deployment samples; compatible with lists and datasets."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> ScaredA00Sample: ...


def run_student_training(
    samples: StudentSamples,
    learner: StudentLearner,
    identity: Mapping[str, object],
    output_dir: Path,
    *,
    max_steps: int | None = None,
    resume: Path | None = None,
    target_provider: Callable[[ScaredA00Sample], AlignedTeacherTarget] | None = None,
    metric_truth_provider: Callable[[ScaredA00Sample], OfflineMetricTruth] | None = None,
) -> dict[str, object]:
    """max_steps bounds consumed samples across warmup/joint epochs and resume.

    No silent invalid-sample skipping. Failed updates retain the last committed
    checkpoint; no recovery save of potentially damaged parameters is attempted.
    Each successful sample commits one cursor step. Joint mode updates model
    once; frozen mode leaves its weights/Adam unchanged. Eligible updates train
    repairs; the selector starts after any configured normalization burn-in.
    Fixed order, no implicit shuffle or split changes.
    """
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    if resume is not None:
        identity = restore_model_initialization(resume, identity)
    if learner.student_frozen:
        if "model_initialization" not in identity:
            raise ValueError("frozen training requires an explicit Phase I model checkpoint")
        validate_model_initialization(identity["model_initialization"])
    size = len(samples)
    if not size:
        raise ValueError("empty student training manifest")
    if learner.config.kd_weight and target_provider is None:
        raise ValueError("KD requires a frozen aligned target provider")
    if (
        learner.intervention_config is not None
        and learner.intervention_config.metric_quality is not None
        and (learner.config.joint_epochs or learner.intervention_config.geometry_supervision_weight)
        and metric_truth_provider is None
    ):
        raise ValueError("physical joint supervision requires an offline TRAIN truth provider")
    output_dir.mkdir(parents=True, exist_ok=False)
    cursor = {"epoch": 0, "next_sample_index": 0, "samples_seen": 0, "optimizer_steps": 0}
    if resume is not None:
        cursor = load_student_checkpoint(resume, learner, identity, size)
    write_json_atomic(
        output_dir / "run.json",
        {
            "identity": dict(identity),
            "student_config_hash": learner.config.config_hash,
            "routing_config_hash": learner.routing.config_hash,
        },
    )
    latest = output_dir / "initial.pt"
    save_student_checkpoint(latest, learner, identity, cursor)
    total = learner.config.student_epochs + learner.config.joint_epochs
    consumed = 0
    started = time.perf_counter()
    with TrainingLog(output_dir / "train.jsonl") as log, TrainingSignals() as signals:
        try:
            while cursor["epoch"] < total and (max_steps is None or consumed < max_steps):
                if signals.requested():
                    break
                model_rate = None
                if learner.config.model_lr_schedule != "constant" and not learner.student_frozen:
                    model_rate = apply_model_learning_rate(
                        learner.model_optimizer, learner.config, cursor["epoch"]
                    )
                sample = samples[cursor["next_sample_index"]]
                # Reject forbidden sample/GT before invoking any teacher provider.
                if (
                    sample.supervision is not None
                    or sample.data_role != "final_train"
                    or sample.sample_id.split("/")[0] not in {f"dataset_{i}" for i in range(1, 8)}
                ):
                    raise ValueError("student training source violates GT/role exclusion")
                target = (
                    target_provider(sample)
                    if learner.config.kd_weight and target_provider
                    else None
                )
                joint = cursor["epoch"] >= learner.config.student_epochs
                needs_truth = joint or (
                    learner.intervention_config is not None
                    and bool(learner.intervention_config.geometry_supervision_weight)
                )
                truth = (
                    metric_truth_provider(sample) if needs_truth and metric_truth_provider else None
                )
                selector_context: tuple[tuple[ScaredA00Sample, OfflineMetricTruth], ...] = ()
                context_frames = (
                    learner.intervention_config.selector_context_frames
                    if learner.intervention_config is not None
                    else 1
                )
                if joint and context_frames > 1:
                    if metric_truth_provider is None:
                        raise ValueError(
                            "fresh selector context requires an offline TRAIN provider"
                        )
                    context_samples = [
                        samples[index]
                        for index in range(
                            max(0, cursor["next_sample_index"] - context_frames + 1),
                            cursor["next_sample_index"],
                        )
                    ]
                    if any(
                        item.supervision is not None
                        or item.data_role != "final_train"
                        or item.sample_id.split("/")[0] not in {f"dataset_{i}" for i in range(1, 8)}
                        for item in context_samples
                    ):
                        raise ValueError("fresh selector context violates GT/role exclusion")
                    selector_context = tuple(
                        (context_sample, metric_truth_provider(context_sample))
                        for context_sample in context_samples
                    )
                if selector_context:
                    losses = learner.step(
                        sample,
                        joint=joint,
                        teacher_target=target,
                        position=cursor["samples_seen"],
                        metric_truth=truth,
                        selector_context=selector_context,
                    )
                elif truth is not None:
                    losses = learner.step(
                        sample,
                        joint=joint,
                        teacher_target=target,
                        position=cursor["samples_seen"],
                        metric_truth=truth,
                    )
                else:
                    losses = learner.step(
                        sample,
                        joint=joint,
                        teacher_target=target,
                        position=cursor["samples_seen"],
                    )
                cursor["samples_seen"] += 1
                cursor["optimizer_steps"] += 1
                cursor["next_sample_index"] += 1
                if cursor["next_sample_index"] == size:
                    cursor["epoch"] += 1
                    cursor["next_sample_index"] = 0
                consumed += 1
                log.emit(
                    {
                        "event": "sample_progress",
                        "sample_id": sample.sample_id,
                        "stage": "joint" if joint else "student",
                        "elapsed_seconds": time.perf_counter() - started,
                        "cursor": dict(cursor),
                        **({"model_learning_rate": model_rate} if model_rate is not None else {}),
                        **losses,
                    }
                )
                if (
                    consumed % learner.config.checkpoint_every == 0
                    or cursor["next_sample_index"] == 0
                ):
                    latest = output_dir / f"sample-{cursor['samples_seen']:09d}.pt"
                    save_student_checkpoint(latest, learner, identity, cursor)
        except BaseException as error:
            log.emit(
                {"event": "train_failed", "error": str(error), "last_safe_checkpoint": str(latest)}
            )
            raise
    destination = output_dir / ("final.pt" if cursor["epoch"] == total else "stopped.pt")
    save_student_checkpoint(destination, learner, identity, cursor)
    return {
        "model_initialization": identity.get("model_initialization"),
        "cursor": cursor,
        "checkpoint": str(destination),
        "schedule_complete": cursor["epoch"] == total,
    }
