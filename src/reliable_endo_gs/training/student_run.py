"""Sequential student/joint execution with exact cursors and immutable outputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.teachers import AlignedTeacherTarget
from reliable_endo_gs.training.harness import TrainingLog, TrainingSignals
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_learning import StudentLearner
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
) -> dict[str, object]:
    """max_steps bounds consumed samples across warmup/joint epochs and resume.

    No silent invalid-sample skipping. Failed updates retain the last committed
    checkpoint; no recovery save of potentially damaged parameters is attempted.
    Every successful sample updates model once; joint also updates repairs and
    selector once. Fixed order, no implicit shuffle or split changes.
    """
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    size = len(samples)
    if not size:
        raise ValueError("empty student training manifest")
    if learner.config.kd_weight and target_provider is None:
        raise ValueError("KD requires a frozen aligned target provider")
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
    with TrainingLog(output_dir / "train.jsonl") as log, TrainingSignals() as signals:
        try:
            while cursor["epoch"] < total and (max_steps is None or consumed < max_steps):
                if signals.requested():
                    break
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
                losses = learner.step(
                    sample, joint=joint, teacher_target=target, position=cursor["samples_seen"]
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
                        "cursor": dict(cursor),
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
        "cursor": cursor,
        "checkpoint": str(destination),
        "schedule_complete": cursor["epoch"] == total,
    }
