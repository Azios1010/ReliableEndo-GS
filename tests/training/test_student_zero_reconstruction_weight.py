"""A zero photometric weight (distillation-only student): allowed, loss is the KD term alone, the head stays put."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from reliable_endo_gs.training.student_config import StudentLearningConfig
from tests.training.test_student_joint import setup  # noqa: F401  (pytest fixture)


def test_zero_reconstruction_weight_needs_another_supervision_term():
    assert StudentLearningConfig(reconstruction_weight=0.0, kd_weight=1.0).kd_weight == 1.0
    assert StudentLearningConfig(reconstruction_weight=0.0, kd_weight=0.0, gt_weight=1.0).gt_weight
    with pytest.raises(ValueError, match="zero reconstruction weight"):
        StudentLearningConfig(reconstruction_weight=0.0, kd_weight=0.0)
    with pytest.raises(ValueError):
        StudentLearningConfig(reconstruction_weight=-1.0)
    # the default identity is unchanged
    default = StudentLearningConfig()
    assert default.config_hash == "6c1b50685777e7db3c6605cd1abaa8823ebcb91c320f31f15cbc9ffb7229388d"


def test_distillation_only_step_updates_the_stereo_network_but_not_the_head(setup):  # noqa: F811
    learner, sample, target = setup
    learner.config = replace(learner.config, reconstruction_weight=0.0, kd_weight=1.0)
    head_before = {k: v.clone() for k, v in learner.model.head.state_dict().items()}
    stereo_before = {k: v.clone() for k, v in learner.model.stereo_student.state_dict().items()}
    result = learner.step(sample, joint=False, teacher_target=target)
    assert result["loss"] == pytest.approx(result["kd"], rel=1e-6) and result["kd"] > 0
    assert result["reconstruction"] > 0  # still measured, only not optimised
    assert all(torch.equal(v, head_before[k]) for k, v in learner.model.head.state_dict().items())
    assert any(
        not torch.equal(v, stereo_before[k])
        for k, v in learner.model.stereo_student.state_dict().items()
    )
