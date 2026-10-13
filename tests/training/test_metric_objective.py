"""Metric supervision units, fixed support, isolation and joint ownership."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.config.intervention import InterventionConfig, load_intervention_config
from reliable_endo_gs.contracts.deployment import RectifiedStereoCamera
from reliable_endo_gs.training.metric_objective import metric_depth_objective
from reliable_endo_gs.training.metric_quality import OfflineMetricTruth
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_learning import StudentLearner
from reliable_endo_gs.training.student_run import run_student_training
from tests.architecture.test_a04_reference_reconstruction import _batch
from tests.routing.test_balanced import setup as balanced_fixture
from tests.routing.test_metric_joint import metric_learner

setup = balanced_fixture
SAMPLE_ID = "dataset_1/keyframe_1/frame_data000000"


def inputs():
    batch = _batch(dtype=torch.float32)
    kl, kr = batch.camera.K_L.clone(), batch.camera.K_R.clone()
    kl[:, 0, 0] = kr[:, 0, 0] = 100
    kl[:, 0, 2] = 3
    kr[:, 0, 2] = 1
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 0, 3] = -0.01
    camera = RectifiedStereoCamera(kl, kr, transform, True)
    batch = replace(batch, camera=camera)
    truth = OfflineMetricTruth(
        SAMPLE_ID, torch.full_like(batch.support, 12, dtype=torch.float32), batch.support.clone()
    )
    return batch, truth


def test_metric_units_principal_offset_and_gradient_direction():
    batch, truth = inputs()
    truth = replace(truth, disparity=truth.disparity.requires_grad_())
    prediction = torch.full_like(truth.disparity, 10, requires_grad=True)
    result = metric_depth_objective(prediction, batch, truth, sample_id=SAMPLE_ID, scale_mm=1)
    # fx=100px, B=.01m, effective disparities8/10px =>125/100mm.
    assert result.mae_mm == pytest.approx(25)
    assert float(result.loss.detach()) == pytest.approx(24.5)
    result.loss.backward()
    assert bool((prediction.grad[truth.valid_mask] < 0).all())
    assert truth.disparity.grad is None


def test_region_and_gt_support_are_fixed_and_ignore_unlabelled_pixels():
    batch, truth = inputs()
    region = torch.zeros_like(batch.support)
    region[..., 2, 2] = True
    prediction = torch.full_like(truth.disparity, 10, requires_grad=True)
    result = metric_depth_objective(
        prediction, batch, truth, sample_id=SAMPLE_ID, scale_mm=5, region_mask=region
    )
    assert result.pixels == 1
    assert float(result.loss.detach()) == pytest.approx(4.5)
    result.loss.backward()
    assert bool((prediction.grad[~region] == 0).all())
    assert prediction.grad[region] < 0


def test_invalid_prediction_cannot_shrink_denominator():
    batch, truth = inputs()
    prediction = truth.disparity.clone()
    prediction[..., 2, 2] = 2  # zero effective disparity, on supported GT
    with pytest.raises(ValueError, match="fixed TRAIN truth support"):
        metric_depth_objective(prediction, batch, truth, sample_id=SAMPLE_ID, scale_mm=1)


def test_empty_truth_retains_zero_gradient_without_claiming_observation():
    batch, truth = inputs()
    truth = replace(truth, valid_mask=torch.zeros_like(truth.valid_mask))
    prediction = truth.disparity.clone().requires_grad_()
    result = metric_depth_objective(prediction, batch, truth, sample_id=SAMPLE_ID, scale_mm=1)
    assert result.pixels == 0 and result.mae_mm is None and float(result.loss.detach()) == 0
    result.loss.backward()
    assert bool((prediction.grad == 0).all())


@pytest.mark.parametrize("change", ["identity", "role", "scale", "region"])
def test_metric_objective_rejects_unverified_inputs(change):
    batch, truth = inputs()
    kwargs = {"sample_id": SAMPLE_ID, "scale_mm": 1}
    if change == "identity":
        kwargs["sample_id"] = "dataset_2/keyframe_1/frame_data000000"
    elif change == "role":
        truth = replace(truth, data_role="final_test")
    elif change == "scale":
        kwargs["scale_mm"] = 0
    else:
        kwargs["region_mask"] = torch.zeros(1, 1, 1, 1, dtype=torch.bool)
    with pytest.raises(ValueError):
        metric_depth_objective(truth.disparity, batch, truth, **kwargs)


def test_opt_in_config_identity_and_historical_defaults():
    old = load_intervention_config(Path("configs/routing/intervention_metric_action_heads.yaml"))
    assert old.config_hash == "8fb6905c83f04b6bfbefef65a1b7ec4273cf0f5de6fe4dfa16ac2bf5fb530eec"
    cfg = replace(old, geometry_supervision_weight=1)
    assert cfg.config_hash != old.config_hash
    assert InterventionConfig(**cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="physical TRAIN truth"):
        InterventionConfig(geometry_supervision_weight=1)


@pytest.mark.parametrize("missing", [False, True])
def test_joint_metric_term_balances_and_updates_without_deployment_gt(setup, missing, tmp_path):
    old, sample, teacher, truth = metric_learner(setup)
    cfg = replace(old.intervention_config, geometry_supervision_weight=1)
    learner = StudentLearner(old.model, old.config, old.routing, intervention_config=cfg)
    truth = replace(truth, disparity=truth.disparity * 1.1)
    if missing:
        truth = replace(truth, valid_mask=torch.zeros_like(truth.valid_mask))
    result = learner.step(
        sample, joint=True, teacher_target=teacher, metric_truth=truth, position=1
    )
    assert learner.objective_balancer.names == ("rgb", "warp", "kd", "repair", "geometry")
    assert learner.objective_balancer.weights.shape == (5,)
    assert result["geometry_base_pixels"] == (0 if missing else int(truth.valid_mask.sum()))
    assert result["geometry"] == 0 if missing else result["geometry"] > 0
    assert sample.supervision is None
    assert result["grad_norm_student"] > 0 and result["grad_norm_stereo_repair"] > 0
    assert teacher.disparity.grad is None
    if not missing:
        assert "model_shared_grad_geometry" in result
    checkpoint = tmp_path / "metric.pt"
    cursor = {"epoch": 0, "next_sample_index": 1, "samples_seen": 1, "optimizer_steps": 1}
    save_student_checkpoint(checkpoint, learner, {}, cursor)
    assert load_student_checkpoint(checkpoint, learner, {}, 2) == cursor


def test_student_stage_receives_explicit_metric_truth_without_action_updates(setup, tmp_path):
    old, sample, teacher, truth = metric_learner(setup)
    cfg = replace(old.intervention_config, geometry_supervision_weight=1)
    learning = replace(old.config, student_epochs=1, joint_epochs=0)
    learner = StudentLearner(old.model, learning, old.routing, intervention_config=cfg)
    truth = replace(truth, disparity=truth.disparity * 1.1)
    calls = []

    def offline(sample):
        calls.append(sample.sample_id)
        return truth

    result = run_student_training(
        [sample],
        learner,
        {},
        tmp_path / "student-metric",
        target_provider=lambda _: teacher,
        metric_truth_provider=offline,
    )
    assert result["schedule_complete"] and calls == [sample.sample_id]
    assert learner.optimizer_update_counts == {"model": 1, "actions": 0, "selector": 0}
