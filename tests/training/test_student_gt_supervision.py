"""Opt-in GT-disparity supervision of the plain student path (KD attribution control): one loss owner, history unchanged."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from reliable_endo_gs.training.distillation import GroundTruthDisparity
from reliable_endo_gs.training.student_config import StudentLearningConfig, load_student_config
from reliable_endo_gs.training.student_run import run_student_training
from tests.training.test_student_joint import setup  # noqa: F401  (pytest fixture)


def _gt(sample, offset: float, valid: bool = True) -> GroundTruthDisparity:
    shape = sample.batch.support.shape
    disparity = torch.full(shape, 8.0 + offset)
    return GroundTruthDisparity(
        disparity,
        torch.full(shape, valid, dtype=torch.bool),
        sample.batch.support.clone(),
    )


def test_gt_weight_is_opt_in_and_keeps_historical_identities(tmp_path):
    default = StudentLearningConfig()
    assert default.gt_weight == 0.0 and "gt_weight" not in default.to_dict()
    assert default.config_hash == "6c1b50685777e7db3c6605cd1abaa8823ebcb91c320f31f15cbc9ffb7229388d"
    gt = StudentLearningConfig(gt_weight=1.0, kd_weight=0.0)
    assert gt.to_dict()["gt_weight"] == 1.0 and gt.config_hash != default.config_hash
    for bad in (-1.0, float("nan"), True):
        with pytest.raises(ValueError):
            StudentLearningConfig(gt_weight=bad)
    path = tmp_path / "gt.yaml"
    path.write_text("kd_weight: 0.0\ngt_weight: 1.0\nstudent_epochs: 3\n", encoding="utf-8")
    assert load_student_config(path).gt_weight == 1.0


def test_kd_step_is_unchanged_without_gt_weight(setup):  # noqa: F811
    learner, sample, target = setup
    result = learner.step(sample, joint=False, teacher_target=target)
    assert "gt" not in result and not any(key.startswith("gt_") for key in result)
    assert result["loss"] == pytest.approx(result["reconstruction"] + result["kd"])
    with pytest.raises(ValueError, match="gt_weight is zero"):
        learner.step(sample, joint=False, teacher_target=target, gt_target=_gt(sample, 0.0))


def test_gt_step_uses_the_shared_loss_and_requires_a_target(setup):  # noqa: F811
    learner, sample, _ = setup
    learner.config = replace(learner.config, kd_weight=0.0, gt_weight=1.0)
    with pytest.raises(ValueError, match="explicit GT target"):
        learner.step(sample, joint=False)
    prediction = learner.state(sample).base.stereo.disparity.detach()
    gt = _gt(sample, 1.5)
    expected = F.smooth_l1_loss(prediction, gt.disparity)
    result = learner.step(sample, joint=False, gt_target=gt)
    assert result["kd"] == 0.0 and result["gt"] == pytest.approx(float(expected), rel=1e-4)
    assert result["loss"] == pytest.approx(result["reconstruction"] + result["gt"], rel=1e-5)
    assert result["gt_accepted_pixels"] == float(gt.validity.sum()) and result["gt_empty"] == 0.0
    assert not any(key.startswith("kd_") for key in result)  # no KD diagnostics leak in


def test_gt_with_no_valid_pixels_contributes_zero_and_is_reported(setup):  # noqa: F811
    learner, sample, _ = setup
    learner.config = replace(learner.config, kd_weight=0.0, gt_weight=1.0)
    result = learner.step(sample, joint=False, gt_target=_gt(sample, 0.0, valid=False))
    assert result["gt"] == 0.0 and result["gt_empty"] == 1.0 and result["gt_accepted_pixels"] == 0.0
    assert result["loss"] == pytest.approx(result["reconstruction"], rel=1e-6)


def test_gt_supervision_moves_the_student_towards_gt(setup):  # noqa: F811
    learner, sample, _ = setup
    learner.config = replace(
        learner.config, kd_weight=0.0, gt_weight=1.0, reconstruction_weight=1.0
    )
    gt = _gt(sample, 4.0)
    start = float(F.l1_loss(learner.state(sample).base.stereo.disparity.detach(), gt.disparity))
    for _ in range(40):
        learner.step(sample, joint=False, gt_target=gt)
    end = float(F.l1_loss(learner.state(sample).base.stereo.disparity.detach(), gt.disparity))
    assert end < start


def test_run_loop_requires_and_uses_the_gt_provider(setup, tmp_path):  # noqa: F811
    learner, sample, _ = setup
    learner.config = replace(learner.config, kd_weight=0.0, gt_weight=1.0, checkpoint_every=1)
    with pytest.raises(ValueError, match="ground-truth provider"):
        run_student_training([sample], learner, {}, tmp_path / "missing", max_steps=1)
    run_student_training(
        [sample], learner, {}, tmp_path / "ok", max_steps=1, gt_provider=lambda s: _gt(s, 1.0)
    )
    event = json.loads((tmp_path / "ok" / "train.jsonl").read_text().splitlines()[-1])
    assert event["gt"] > 0 and event["gt_accepted_pixels"] > 0 and event["kd"] == 0.0
