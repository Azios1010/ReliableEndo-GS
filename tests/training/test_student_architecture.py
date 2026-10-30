"""Stereo architecture contracts: epipolar support, KD and strict recovery."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch
from reliable_endo_gs.reconstruction.student import StereoStudent, StudentReconstructionModel
from reliable_endo_gs.training.distillation import stereo_distillation
from reliable_endo_gs.training.student_checkpoint import (
    initialize_student_model,
    load_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_config import StudentLearningConfig, load_student_config
from reliable_endo_gs.training.student_learning import StudentLearner
from tests.architecture.test_a04_reference_reconstruction import _camera, _target
from tests.routing.test_frozen_a04 import AnalyticRenderer
from tests.training.test_student_joint import setup as joint_fixture

setup = joint_fixture


def _stereo_batch(batch_size=1, hw=(19, 27)):
    left = torch.rand(batch_size, 3, *hw)
    return DeploymentStereoBatch(
        left,
        torch.rand_like(left),
        _camera(batch_size, torch.float32),
        torch.ones(batch_size, 1, *hw, dtype=torch.bool),
    )


def test_compact_default_retains_config_hash_keys_and_seeded_prediction():
    config = StudentLearningConfig()
    assert config.config_hash == "6c1b50685777e7db3c6605cd1abaa8823ebcb91c320f31f15cbc9ffb7229388d"
    assert "stereo_architecture" not in config.to_dict()
    torch.manual_seed(17)
    default = StereoStudent()
    torch.manual_seed(17)
    explicit = StereoStudent(architecture="compact")
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert not any("cost_regularizer" in k for k in default.state_dict())
    batch = _stereo_batch()
    assert torch.equal(default(batch), explicit(batch))


@pytest.mark.parametrize("batch_size", [1, 2])
def test_multiscale_output_and_detached_teacher_kd_gradients(batch_size):
    torch.manual_seed(3)
    batch = _stereo_batch(batch_size)
    model = StereoStudent(8, 8, architecture="multiscale")
    prediction = model(batch)
    offset = (batch.camera.K_L[:, 0, 2] - batch.camera.K_R[:, 0, 2])[:, None, None, None]
    assert prediction.shape == (batch_size, 1, 19, 27)
    assert torch.isfinite(prediction).all()
    assert ((prediction - offset) >= model.minimum_effective_px).all()
    teacher = _target(batch)
    teacher = replace(teacher, disparity=teacher.disparity.requires_grad_(True))
    measured = stereo_distillation(prediction, teacher, batch)
    assert measured.diagnostics["kd_accepted_pixels"] > 0
    measured.loss.backward()
    assert teacher.disparity.grad is None
    for branch in (model.features, model.decoder, model.cost_regularizer):
        grads = [p.grad for p in branch.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_learned_cost_cannot_restore_invalid_epipolar_candidates():
    model = StereoStudent(8, 8, architecture="multiscale")
    with torch.no_grad():
        model.cost_regularizer.body[-1].bias.fill_(1000)
    _, probabilities, features = model._matching(_stereo_batch())
    assert features.shape[-2:] == (5, 7)
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(1), torch.ones_like(probabilities[:, 0]))
    for column in range(probabilities.shape[-1]):
        assert torch.count_nonzero(probabilities[:, column + 1 :, :, column]) == 0
    assert torch.equal(probabilities[:, 0, :, 0], torch.ones_like(probabilities[:, 0, :, 0]))


def test_multiscale_config_and_warmstart_architecture_boundary(setup, tmp_path):
    config = load_student_config(Path("configs/training/student_kd_multiscale.yaml"))
    assert config.stereo_architecture == "multiscale" and config.kd_weight > 0
    assert config.joint_epochs == 0 and StudentLearningConfig(**config.to_dict()) == config
    for options in [
        {"stereo_architecture": "unknown"},
        {"stereo_architecture": "multiscale", "width": 4},
    ]:
        with pytest.raises(ValueError):
            StudentLearningConfig(**options)
    compact, sample, teacher = setup
    compact_path = tmp_path / "compact.pt"
    save_student_checkpoint(
        compact_path,
        compact,
        {"source_sha256": "compact-source"},
        {"epoch": 0, "next_sample_index": 0, "samples_seen": 0, "optimizer_steps": 0},
    )
    learning = replace(compact.config, stereo_architecture="multiscale", joint_epochs=0)
    model = StudentReconstructionModel(
        renderer=AnalyticRenderer(),
        config=compact.model.config,
        width=8,
        bins=4,
        stereo_architecture="multiscale",
    )
    learner = StudentLearner(model, learning, compact.routing)
    before = copy.deepcopy(learner.model.state_dict())
    with pytest.raises(ValueError, match="architecture mismatch"):
        initialize_student_model(compact_path, learner)
    assert all(torch.equal(v, before[k]) for k, v in learner.model.state_dict().items())
    learner.step(sample, joint=False, teacher_target=teacher)
    assert learner.optimizer_update_counts == {"model": 1, "actions": 0, "selector": 0}
    cursor = {"epoch": 0, "next_sample_index": 1, "samples_seen": 1, "optimizer_steps": 1}
    trained = tmp_path / "multiscale.pt"
    save_student_checkpoint(trained, learner, {"source": "current"}, cursor)
    restored = StudentLearner(copy.deepcopy(model), learning, compact.routing)
    assert load_student_checkpoint(trained, restored, {"source": "current"}, 2) == cursor
    assert restored.model.stereo_student.architecture == "multiscale"
