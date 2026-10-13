"""Balanced pilot identities, independent metric guards and deployment choices."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.actions.repair import Action
from reliable_endo_gs.cli.mask_validation import main
from reliable_endo_gs.config.intervention import InterventionConfig, load_intervention_config
from reliable_endo_gs.routing.repairability import select_repairability
from reliable_endo_gs.routing.tradeoff import COMPONENT_NAMES, load_tradeoff_config, outcome_utility
from reliable_endo_gs.training.metric_quality import metric_action_policy, metric_targets


@pytest.fixture
def config():
    return load_intervention_config(Path("configs/routing/intervention_metric_balanced.yaml"))


def changes(**values):
    return [values.get(name, 0.0) for name in COMPONENT_NAMES]


def test_bounded_cli_defaults_use_the_matching_balanced_pair(monkeypatch):
    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("reliable_endo_gs.training.mask_validation.run_mask_validation", run)
    arguments = []
    for name in (
        "data-root",
        "training-manifest",
        "development-manifest",
        "worklist-path",
        "target-mask-index",
        "cache-root",
        "initial-checkpoint",
        "output-dir",
    ):
        arguments.extend(("--" + name, "placeholder"))
    assert main(arguments + ["--preflight-only"]) == 0
    cfg = load_intervention_config(captured["intervention_config_path"])
    metric = load_tradeoff_config(captured["metric_config_path"])
    assert (
        cfg.config_hash
        == load_intervention_config(
            Path("configs/routing/intervention_metric_balanced.yaml")
        ).config_hash
    )
    assert metric_action_policy(cfg, 1).config_hash == metric.config_hash
    assert captured["preflight_only"]


def test_balanced_config_matches_evaluation_and_has_a_new_identity(config):
    evaluation = load_tradeoff_config(Path("configs/routing/metric_tradeoff_joint_balanced.yaml"))
    assert metric_action_policy(config, 1).config_hash == evaluation.config_hash
    old = load_intervention_config(Path("configs/routing/intervention_metric.yaml"))
    assert config.config_hash != old.config_hash
    assert evaluation.config_hash != metric_action_policy(old, 1).config_hash
    assert InterventionConfig(**config.to_dict()) == config
    assert config.stereo_training_priority == old.stereo_training_priority == 3.0
    assert config.metric_scales == old.metric_scales


def test_equal_normalized_geometry_and_render_gains_receive_equal_group_utility(config):
    policy = metric_action_policy(config, 1)
    assert config.metric_scales is not None
    geometry = changes(global_depth_mae_gain_mm=1, local_depth_mae_gain_mm=1)
    render = changes(
        left_psnr_gain_db=0.1,
        right_psnr_gain_db=0.1,
        left_ssim_gain=0.001,
        right_ssim_gain=0.001,
    )
    assert outcome_utility(geometry, config.metric_scales, policy) == pytest.approx(
        outcome_utility(render, config.metric_scales, policy)
    )
    assert sum(policy.score_weights[:5]) == pytest.approx(1)
    assert sum(policy.score_weights[5:9]) == pytest.approx(1)
    for vector in (geometry, render):
        assert metric_targets(vector, config, 1)[0] == metric_targets(vector, config, 2)[0]
    assert metric_targets(changes(), config, 1) == (0, 0, False)


@pytest.mark.parametrize("action", [1, 2])
@pytest.mark.parametrize("component", COMPONENT_NAMES)
def test_every_guard_is_inclusive_and_blocks_large_tradeoffs(config, action, component):
    policy = metric_action_policy(config, action)
    limit = policy.minimum_changes[COMPONENT_NAMES.index(component)]
    assert not metric_targets(changes(**{component: limit}), config, action)[2]
    unsafe = changes(**{component: limit - 4 * policy.comparison_epsilon})
    # Large improvements elsewhere cannot compensate for a raw-unit violation.
    other = (
        "global_depth_mae_gain_mm"
        if component != "global_depth_mae_gain_mm"
        else "left_psnr_gain_db"
    )
    unsafe[COMPONENT_NAMES.index(other)] = 100
    assert metric_targets(unsafe, config, action)[2]


def test_gaussian_tightens_quality_without_a_second_risk_cutoff(config):
    assert config.maximum_stereo_harm_probability == config.maximum_gaussian_harm_probability == 0.2
    stereo, gaussian = (metric_action_policy(config, a) for a in (1, 2))
    assert gaussian.maximum_psnr_drop_db == stereo.maximum_psnr_drop_db / 2
    assert gaussian.maximum_local_rgb_increase == stereo.maximum_local_rgb_increase / 2
    assert gaussian.maximum_coverage_drop == stereo.maximum_coverage_drop == 0
    mild = changes(local_depth_mae_gain_mm=0.2, right_psnr_gain_db=-0.02)
    assert not metric_targets(mild, config, 1)[2]
    assert not metric_targets(mild, config, 2)[2]
    boundary = changes(local_depth_mae_gain_mm=0.2, right_psnr_gain_db=-0.04)
    assert not metric_targets(boundary, config, 1)[2]
    assert metric_targets(boundary, config, 2)[2]
    # Changing utility normalization cannot relax physical limits.
    different_scales = replace(config, metric_scales=tuple(s * 100 for s in config.metric_scales))
    assert metric_targets(boundary, different_scales, 2)[2]


def test_risk_gain_budget_and_unknown_evidence_still_control_selection(config):
    gains = torch.tensor([[0.0, 0.3, 0.5]])
    harms = torch.tensor([[0.0, 0.1, 0.15]])
    costs = torch.tensor([[0.0, 120.0, 50.0]])
    valid = torch.tensor([True])
    assert select_repairability(gains, harms, costs, valid, 200, config).action == Action.GAUSSIAN
    old = load_intervention_config(Path("configs/routing/intervention_metric.yaml"))
    assert select_repairability(gains, harms, costs, valid, 200, old).action == Action.STEREO
    assert select_repairability(gains, harms, costs, valid, 40, config).action == Action.STOP
    assert select_repairability(gains, harms + 0.2, costs, valid, 200, config).action == Action.STOP
    assert select_repairability(-gains, harms, costs, valid, 200, config).action == Action.STOP
    missing = torch.zeros_like(harms, dtype=torch.bool)
    assert (
        select_repairability(gains, harms, costs, valid, 200, config, risk_available=missing).action
        == Action.STOP
    )


def test_geometry_gain_pilot_changes_measured_reward_and_preserves_harm(config):
    candidate = load_intervention_config(Path("configs/routing/intervention_metric_geometry.yaml"))
    evaluation = load_tradeoff_config(Path("configs/routing/metric_tradeoff_joint_geometry.yaml"))
    assert metric_action_policy(candidate, 1).config_hash == evaluation.config_hash
    assert candidate.config_hash != config.config_hash
    assert candidate.metric_scales == config.metric_scales
    for action in (1, 2):
        assert (
            metric_action_policy(candidate, action).minimum_changes
            == metric_action_policy(config, action).minimum_changes
        )
        for vector in (
            changes(global_depth_mae_gain_mm=0.3, local_depth_mae_gain_mm=0.7),
            changes(global_depth_mae_gain_mm=-0.3, local_depth_mae_gain_mm=-0.7),
        ):
            old = metric_targets(vector, config, action)
            new = metric_targets(vector, candidate, action)
            assert new[0] == pytest.approx(old[0] * 1.5)
            assert new[1] == pytest.approx(old[1] * 1.5)
            assert new[2] == old[2]
        render = changes(left_psnr_gain_db=0.1, right_psnr_gain_db=0.1)
        assert metric_targets(render, candidate, action) == metric_targets(render, config, action)
        assert metric_targets(changes(), candidate, action) == (0, 0, False)
        severe = changes(global_depth_mae_gain_mm=100, right_psnr_gain_db=-1)
        assert metric_targets(severe, candidate, action)[2]


def test_geometry_gain_pilot_cutoffs_allow_stereo_without_forcing_it():
    candidate = load_intervention_config(Path("configs/routing/intervention_metric_geometry.yaml"))
    costs = torch.tensor([[0.0, 130.0, 60.0]])
    valid = torch.tensor([True])
    risks = torch.tensor([[0.0, 0.505, 0.21]])
    for gains, action in (
        ([0.0, 0.3, 0.1], Action.STEREO),
        ([0.0, 0.1, 0.3], Action.GAUSSIAN),
        ([0.0, -0.1, -0.3], Action.STOP),
    ):
        assert (
            select_repairability(torch.tensor([gains]), risks, costs, valid, 200, candidate).action
            == action
        )


def test_stereo_gain_multiplier_preserves_gaussian_harm_and_old_config_identity(config):
    candidate = load_intervention_config(
        Path("configs/routing/intervention_metric_stereo_gain.yaml")
    )
    evaluation = load_tradeoff_config(Path("configs/routing/metric_tradeoff_joint_balanced.yaml"))
    assert metric_action_policy(candidate, 1).config_hash == evaluation.config_hash
    assert candidate.stereo_utility_multiplier == 1.5
    assert candidate.config_hash != config.config_hash
    assert "stereo_utility_multiplier" not in config.to_dict()
    assert InterventionConfig(**candidate.to_dict()) == candidate
    assert config.config_hash == "7a28b74aa9eea239814c2f9bafa8cf3d1954d1f9e3f44c46a918fa348cd94c52"
    for vector in (
        changes(),
        changes(global_depth_mae_gain_mm=0.4, local_depth_mae_gain_mm=0.3),
        changes(global_depth_mae_gain_mm=-0.4, local_depth_mae_gain_mm=-0.3),
        changes(left_psnr_gain_db=0.1, right_ssim_gain=0.001),
        changes(global_depth_mae_gain_mm=100, right_psnr_gain_db=-1),
    ):
        old = metric_targets(vector, config, 1)
        new = metric_targets(vector, candidate, 1)
        assert new[0] == pytest.approx(old[0] * 1.5)
        assert new[1] == pytest.approx(old[1] * 1.5)
        assert new[2] == old[2]
        assert metric_targets(vector, candidate, 2) == metric_targets(vector, config, 2)
    for factor in (None, 0, -1, float("inf"), float("nan"), True):
        with pytest.raises(ValueError):
            replace(candidate, stereo_utility_multiplier=factor)
    with pytest.raises(ValueError, match="physical metric"):
        InterventionConfig(stereo_utility_multiplier=1.5)
