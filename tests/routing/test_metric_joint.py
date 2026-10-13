"""Physical joint labels, asymmetric guards and dedicated deployment costs."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.actions.repair import Action
from reliable_endo_gs.config.intervention import InterventionConfig, load_intervention_config
from reliable_endo_gs.data.scared_a00 import A00_SUPERVISION_RESAMPLING, ScaredSupervision
from reliable_endo_gs.routing.cost import CostCalibrationPolicy
from reliable_endo_gs.routing.repairability import select_repairability
from reliable_endo_gs.routing.tradeoff import (
    COMPONENT_NAMES,
    TradeoffConfig,
    outcome_utility,
    outcome_utility_tensor,
)
from reliable_endo_gs.training.deployment_costs import calibrate_student_costs
from reliable_endo_gs.training.metric_quality import (
    OfflineMetricTruth,
    metric_action_policy,
    metric_targets,
)
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_learning import StudentLearner
from tests.evaluation.test_mask_validation import BoundedRenderer
from tests.routing.test_balanced import setup as balanced_fixture
from tests.routing.test_metric_tradeoff import vector

setup = balanced_fixture


def policy():
    return load_intervention_config(Path("configs/routing/intervention_metric.yaml"))


def test_metric_priority_changes_geometry_render_ratio_without_rewarding_action_name():
    cfg = policy()
    assert cfg.metric_scales is not None
    current = metric_action_policy(cfg, 1)
    neutral = replace(current, geometry_priority=1.0)
    changes = vector(local_depth_mae_gain_mm=1.12)
    assert outcome_utility(changes, cfg.metric_scales, current) == pytest.approx(
        3 * outcome_utility(changes, cfg.metric_scales, neutral)
    )
    # Equal outcomes have equal utility across repairs; no stereo action bonus.
    assert metric_targets(changes, cfg, 1)[0] == metric_targets(changes, cfg, 2)[0]
    assert metric_targets(vector(), cfg, 1) == (0.0, 0.0, False)
    render = vector(left_psnr_gain_db=0.1, right_psnr_gain_db=0.1)
    assert outcome_utility(render, cfg.metric_scales, current) == outcome_utility(
        render, cfg.metric_scales, neutral
    )
    tensor = torch.tensor(changes, requires_grad=True)
    utility = outcome_utility_tensor(tensor, cfg.metric_scales, current)
    utility.backward()
    assert bool(torch.isfinite(tensor.grad).all())


def test_metric_guard_catches_old_proxy_blind_spot_and_tightens_gaussian():
    cfg = policy()
    unsafe = vector(local_depth_mae_gain_mm=10, right_psnr_gain_db=-0.01236)
    assert metric_targets(unsafe, cfg, 1)[2]
    assert metric_targets(unsafe, cfg, 2)[2]
    # A small render regression is tolerated for stereo, tighter for Gaussian.
    borderline = vector(local_depth_mae_gain_mm=0.2, right_psnr_gain_db=-0.007)
    assert not metric_targets(borderline, cfg, 1)[2]
    assert metric_targets(borderline, cfg, 2)[2]
    for name in COMPONENT_NAMES:
        assert metric_targets(vector(**{name: -100}), cfg, 1)[2]


def test_action_risk_thresholds_and_gain_still_require_positive_utility():
    cfg = policy()
    gains = torch.tensor([[0.0, 1.0, 2.0]])
    risk = torch.tensor([[0.0, 0.15, 0.06]])
    costs = torch.tensor([[0.0, 120.0, 50.0]])
    valid = torch.tensor([True])
    assert select_repairability(gains, risk, costs, valid, 200, cfg).action == Action.STEREO
    assert select_repairability(gains, risk, costs, valid, 100, cfg).action == Action.STOP
    assert (
        select_repairability(
            gains, risk + torch.tensor([[0, 0.1, 0]]), costs, valid, 200, cfg
        ).action
        == Action.STOP
    )
    assert (
        select_repairability(torch.zeros_like(gains), risk, costs, valid, 200, cfg).action
        == Action.STOP
    )


def test_new_policy_roundtrip_and_invalid_settings():
    cfg = policy()
    assert InterventionConfig(**cfg.to_dict()) == cfg
    assert cfg.config_hash != replace(cfg, maximum_gaussian_harm_probability=0.1).config_hash
    with pytest.raises(ValueError):
        replace(cfg, metric_scales=(1.0,))
    with pytest.raises(ValueError):
        replace(cfg, gaussian_guard_multiplier=1.1)
    with pytest.raises(ValueError):
        replace(cfg, maximum_stereo_harm_probability=1.0)
    with pytest.raises(ValueError):
        replace(cfg, metric_quality=None)
    neutral = TradeoffConfig(
        schema="metric_action_tradeoff.v2", collateral_weight=0, local_render_weight=0
    )
    # Old default identity omits the added neutral priority field.
    assert neutral.config_hash != replace(neutral, geometry_priority=3).config_hash


def metric_learner(setup):
    old, sample, target = setup
    model = copy.deepcopy(old.model)
    object.__setattr__(model, "_renderer", BoundedRenderer())
    cfg = replace(policy(), normalization_frames=2, search_radius_px=4)
    learner = StudentLearner(model, old.config, old.routing, intervention_config=cfg)
    with torch.no_grad():
        state = learner.state(sample)
    truth = OfflineMetricTruth(
        sample.sample_id, state.base.target.disparity.clone(), state.support.clone()
    )
    return learner, sample, target, truth


def test_real_truth_requires_resampling_provenance_and_keeps_gt_separate(setup):
    _, sample, _, truth = metric_learner(setup)
    alignment = {
        "supervision": A00_SUPERVISION_RESAMPLING,
        "validity": "same_nearest_exact_source_index_as_disparity",
    }
    offline = replace(
        sample,
        supervision=ScaredSupervision(truth.disparity, truth.valid_mask),
        calibration_metadata={"runtime_resampling": alignment},
    )
    resolved = OfflineMetricTruth.from_a00(offline)
    assert resolved.resampling_identity == A00_SUPERVISION_RESAMPLING
    assert resolved.disparity is truth.disparity
    assert sample.supervision is None
    for metadata in (
        {},
        {"runtime_resampling": {**alignment, "supervision": "nearest_legacy"}},
        {"runtime_resampling": {**alignment, "validity": "different_source_pixel"}},
    ):
        with pytest.raises(ValueError, match="resampling identity"):
            OfflineMetricTruth.from_a00(replace(offline, calibration_metadata=metadata))
    with pytest.raises(ValueError, match="TRAIN sample identity"):
        OfflineMetricTruth.from_a00(replace(offline, data_role="final_test"))
    with pytest.raises(ValueError, match="missing separate"):
        OfflineMetricTruth.from_a00(sample)


def test_joint_retains_truth_resampling_identity_through_device_transfer(setup):
    learner, sample, target, truth = metric_learner(setup)
    truth = replace(truth, resampling_identity=A00_SUPERVISION_RESAMPLING)
    seen = []
    collect = learner.harness.collect

    def record(*args, **kwargs):
        seen.append(kwargs["metric_truth"].resampling_identity)
        return collect(*args, **kwargs)

    learner.harness.collect = record
    result = learner.step(sample, joint=True, teacher_target=target, metric_truth=truth)
    assert seen and set(seen) == {A00_SUPERVISION_RESAMPLING}
    assert result["offline_metric_truth_resampling_identity"] == A00_SUPERVISION_RESAMPLING


@pytest.mark.parametrize("missing", [False, True])
def test_joint_updates_keep_gt_separate_and_mask_unknown_labels(setup, missing):
    learner, sample, target, truth = metric_learner(setup)
    if missing:
        truth = replace(truth, valid_mask=torch.zeros_like(truth.valid_mask))
    result = learner.step(sample, joint=True, teacher_target=target, metric_truth=truth)
    assert sample.supervision is None
    assert learner.optimizer_update_counts == {"model": 1, "actions": 1, "selector": 0}
    assert (
        result["selector_supervision_semantics"]
        == "physical_metric_decision_local_geometry_auxiliary"
    )
    assert (
        result["selector_unknown_metric_regions"] > 0
        if missing
        else result["selector_observed_regions"] > 0
    )
    if missing:
        assert int(learner.selector.normalization.counts[2]) == 0
    for region in result["interventions"]:
        for row in region["actions"][1:]:
            assert row["selector_label_available"] is (not missing)
            if missing:
                assert row["global_gain"] is None and row["harm"] is None
            if not missing:
                action = int(Action[row["action"]])
                gain, local, harm = metric_targets(
                    row["metric_changes"], learner.intervention_config, action
                )
                assert (
                    row["global_gain"] == gain
                    and row["local_gain"] == local
                    and row["harm"] == harm
                )


def test_truth_identity_and_output_grid_rejected_before_updates(setup):
    learner, sample, target, truth = metric_learner(setup)
    for changed in (
        replace(truth, sample_id="dataset_9/keyframe_1/frame_data000000"),
        replace(truth, disparity=truth.disparity[..., :2, :2]),
    ):
        with pytest.raises(ValueError, match="identity|output grid"):
            learner.step(sample, joint=True, teacher_target=target, metric_truth=changed)
        assert learner.optimizer_update_counts == {"model": 0, "actions": 0, "selector": 0}
    with pytest.raises(ValueError, match="offline TRAIN truth"):
        learner.step(sample, joint=True, teacher_target=target)


def test_metric_joint_normalization_resume_and_all_three_owners(setup, tmp_path):
    learner, sample, target, truth = metric_learner(setup)
    for i in range(3):
        learner.step(sample, joint=True, teacher_target=target, metric_truth=truth, position=i)
    assert learner.optimizer_update_counts == {"model": 3, "actions": 3, "selector": 1}
    path = tmp_path / "metric.pt"
    identity = dict(
        source_sha256="source",
        manifest_hash="train",
        role_hash="roles",
        renderer_identity="synthetic",
    )
    save_student_checkpoint(
        path,
        learner,
        identity,
        dict(epoch=3, next_sample_index=0, samples_seen=3, optimizer_steps=3),
    )
    restored, _, _, _ = metric_learner(setup)
    load_student_checkpoint(path, restored, identity, 1)
    assert restored.optimizer_update_counts == learner.optimizer_update_counts
    assert restored.intervention_config == learner.intervention_config


def test_dedicated_train_calibration_preserves_weights_and_replaces_training_max(setup):
    learner, sample, _, _ = metric_learner(setup)
    learner.cost_max_ms[:] = torch.tensor([0.0, 2990.0, 139.0])
    before = learner.frozen_action_identity()
    artifact = calibrate_student_costs(
        learner,
        [sample],
        {"manifest_hash": "train"},
        CostCalibrationPolicy("quantile", 0.95, 0, 2),
    )
    assert learner.frozen_action_identity() == before
    assert artifact.binding["collection_role"] == "train"
    assert artifact.binding["regions_measured"] > 0
    learner.attach_cost_calibration(artifact)
    state = learner.state(sample)
    torch.testing.assert_close(learner.estimated_action_costs(state), artifact.estimated_ms)
    assert float(learner.cost_max_ms[1]) == 2990.0
    with torch.no_grad():
        next(learner.actions.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="changed"):
        learner.estimated_action_costs(state)


@pytest.mark.parametrize("unknown_position", [0, 3])
def test_metric_checkpoint_counts_unknown_updates_before_and_after_burn_in(
    setup, tmp_path, unknown_position
):
    learner, sample, target, truth = metric_learner(setup)
    learner.config = replace(learner.config, joint_epochs=4)
    unknown = replace(truth, valid_mask=torch.zeros_like(truth.valid_mask))
    for position in range(4):
        result = learner.step(
            sample,
            joint=True,
            teacher_target=target,
            metric_truth=unknown if position == unknown_position else truth,
            position=position,
        )
        if position == unknown_position:
            assert result["selector_skip_reason"] == "no_observed_metric_outcomes"
            assert result["selector_supervision_collected"] is False
            if position == 3:
                assert result["selector_normalization_fitted"] is True
                assert result["selector_burn_in"] is False
        # Also exercise an interrupted run before normalization can fit.
        path = tmp_path / f"step-{position + 1}.pt"
        identity = dict(
            source_sha256="source",
            manifest_hash="train",
            role_hash="roles",
            renderer_identity="synthetic",
        )
        save_student_checkpoint(
            path,
            learner,
            identity,
            dict(
                epoch=position + 1,
                next_sample_index=0,
                samples_seen=position + 1,
                optimizer_steps=position + 1,
            ),
        )
        restored, _, _, _ = metric_learner(setup)
        restored.config = replace(restored.config, joint_epochs=4)
        load_student_checkpoint(path, restored, identity, 1)
        assert restored.optimizer_update_counts == learner.optimizer_update_counts
    assert learner.optimizer_update_counts == {"model": 4, "actions": 4, "selector": 1}
    # A fabricated counter that fits the loose sample bound still cannot pass
    # the per-parameter Adam step validation.
    raw = torch.load(path, weights_only=True)
    raw["optimizer_update_counts"]["selector"] = 2
    corrupted = tmp_path / "wrong-adam-count.pt"
    torch.save(raw, corrupted)
    with pytest.raises(ValueError, match="Adam step"):
        load_student_checkpoint(corrupted, restored, identity, 1)
