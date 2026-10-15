"""Checkpoint ancestry and strict recovery after explicit model warm starts."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.cli.student import build_parser
from reliable_endo_gs.training.student_checkpoint import (
    initialize_student_model,
    read_student_checkpoint,
    restore_model_initialization,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_run import run_student_training
from reliable_endo_gs.training.student_runtime import run_student_command
from tests.training.test_student_joint import setup as joint_fixture

setup = joint_fixture


def parent_checkpoint(tmp_path, learner, sample, target):
    learner.step(sample, joint=False, teacher_target=target)
    path = tmp_path / "phase1.pt"
    save_student_checkpoint(
        path,
        learner,
        {"source_sha256": "phase1-source"},
        {"epoch": 0, "next_sample_index": 1, "samples_seen": 1, "optimizer_steps": 1},
    )
    return path


@pytest.mark.parametrize("field", [None, "source_sha256", "manifest_hash", "role_hash"])
def test_warmstart_resume_restores_ancestry_and_retains_identity_checks(setup, tmp_path, field):
    learner, sample, target = setup
    parent = parent_checkpoint(tmp_path, learner, sample, target)
    new = type(learner)(copy.deepcopy(learner.model), learner.config, learner.routing)
    ancestry = initialize_student_model(parent, new)
    identity = {"source_sha256": "current-source", "manifest_hash": "data", "role_hash": "roles"}
    first = run_student_training(
        [sample],
        new,
        {**identity, "model_initialization": ancestry},
        tmp_path / "first",
        max_steps=1,
        target_provider=lambda _: target,
    )
    restored = type(learner)(copy.deepcopy(learner.model), learner.config, learner.routing)
    current_identity = dict(identity)
    if field:
        current_identity[field] = "different"
        before = copy.deepcopy(restored.model.state_dict())
        with pytest.raises(ValueError, match="identity mismatch"):
            run_student_training(
                [sample],
                restored,
                current_identity,
                tmp_path / "resume",
                resume=Path(first["checkpoint"]),
                target_provider=lambda _: target,
            )
        assert all(torch.equal(v, restored.model.state_dict()[k]) for k, v in before.items())
    else:
        # Resume is portable: its recorded ancestry survives relocation of the
        # initialization file and never reapplies those older model weights.
        parent.rename(tmp_path / "relocated-phase1.pt")
        report = run_student_training(
            [sample],
            restored,
            current_identity,
            tmp_path / "resume",
            resume=Path(first["checkpoint"]),
            target_provider=lambda _: target,
        )
        assert report["schedule_complete"]
        assert report["model_initialization"] == ancestry
        final = read_student_checkpoint(Path(report["checkpoint"]))
        assert final["identity"] == {**identity, "model_initialization": ancestry}


@pytest.mark.parametrize("corruption", ["hash", "missing_source", "mode", "supplied_mismatch"])
def test_ancestry_validation_rejects_corruption(setup, tmp_path, corruption):
    learner, sample, target = setup
    path = parent_checkpoint(tmp_path, learner, sample, target)
    raw = read_student_checkpoint(path)
    provenance = {"checkpoint_sha256": "a" * 64, "source_sha256": "phase1-source"}
    identity = {}
    if corruption == "hash":
        provenance["checkpoint_sha256"] = "invalid"
    elif corruption == "missing_source":
        del provenance["source_sha256"]
    elif corruption == "mode":
        provenance.update(mode="random", checkpoint_path="source.pt")
    else:
        identity["model_initialization"] = {**provenance, "checkpoint_sha256": "b" * 64}
    raw["identity"]["model_initialization"] = provenance
    malformed = tmp_path / "bad.pt"
    torch.save(raw, malformed)
    with pytest.raises(ValueError, match="model_initialization"):
        restore_model_initialization(malformed, identity)


def test_frozen_cli_requires_phase1_checkpoint_before_data_access(tmp_path):
    arguments = build_parser().parse_args(
        [
            "train",
            "--config",
            "configs/training/student_joint_routing.yaml",
            "--intervention-config",
            "configs/routing/intervention_metric_signed_gain_frozen_student.yaml",
            "--data-root",
            str(tmp_path / "data"),
            "--manifest",
            str(tmp_path / "manifest"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ValueError, match="requires --initialize-model-from"):
        run_student_command(arguments)
    assert not (tmp_path / "out").exists()


def test_frozen_initialization_rejects_untrained_checkpoint(setup, tmp_path):
    learner, _, _ = setup
    path = tmp_path / "random.pt"
    save_student_checkpoint(
        path,
        learner,
        {"source_sha256": "untrained"},
        {"epoch": 0, "next_sample_index": 0, "samples_seen": 0, "optimizer_steps": 0},
    )
    with pytest.raises(ValueError, match="trained Phase I"):
        initialize_student_model(path, learner, require_trained=True)
    raw = read_student_checkpoint(path)
    raw["optimizer_update_counts"]["model"] = 1
    claimed = tmp_path / "claimed-trained.pt"
    torch.save(raw, claimed)
    with pytest.raises(ValueError, match="trained model Adam"):
        initialize_student_model(claimed, learner, require_trained=True)


def test_frozen_training_entry_requires_model_ancestry(setup, tmp_path):
    from reliable_endo_gs.config.intervention import InterventionConfig
    from reliable_endo_gs.training.student_learning import StudentLearner

    learner, sample, target = setup
    frozen = StudentLearner(
        learner.model,
        replace(learner.config, student_epochs=0),
        learner.routing,
        intervention_config=InterventionConfig(
            schema="intervention_selection.v2", student_update_mode="frozen"
        ),
    )
    with pytest.raises(ValueError, match="explicit Phase I"):
        run_student_training(
            [sample], frozen, {}, tmp_path / "out", target_provider=lambda _: target
        )
    assert not (tmp_path / "out").exists()
