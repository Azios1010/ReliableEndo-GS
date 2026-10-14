"""Synthetic matching, screened KD, objective, allocation and joint contracts."""

from __future__ import annotations

import copy
from dataclasses import fields, replace

import pytest
import torch

from reliable_endo_gs.actions.repair import Action, preserve_zero_gaussian_residual
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.gaussians import AttributeResiduals
from reliable_endo_gs.geometry.stereo_matching import project_left_support, warp_right
from reliable_endo_gs.reconstruction import ReferenceConfig
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.regions.grid import grid_regions
from reliable_endo_gs.routing.config import RoutingConfig
from reliable_endo_gs.routing.repairability import (
    RepairabilitySelector,
    repairability_features,
    select_repairability,
)
from reliable_endo_gs.teachers import TargetCache, TargetCacheKey
from reliable_endo_gs.teachers.cache import TargetCacheError
from reliable_endo_gs.teachers.quality import screen_target
from reliable_endo_gs.teachers.workflow import sample_cache_key
from reliable_endo_gs.training.repairability import quality_gain, stereo_consistency_loss
from reliable_endo_gs.training.sample_order import SeededSampleOrder
from reliable_endo_gs.training.student_checkpoint import (
    initialize_student_model,
    load_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_config import StudentLearningConfig
from reliable_endo_gs.training.student_learning import StudentLearner
from tests.architecture.test_a01_teacher_targets import _target as _cache_target
from tests.architecture.test_a04_reference_reconstruction import _batch, _target
from tests.routing.test_frozen_a04 import AnalyticRenderer


@pytest.fixture
def setup():
    torch.manual_seed(8)
    cfg = InterventionConfig(
        matching_width=4, search_radius_px=4, search_step_px=2, kd_min_depth_m=0.0
    )
    model = StudentReconstructionModel(
        renderer=AnalyticRenderer(),
        config=replace(ReferenceConfig(), head_hidden_dims=(8, 8)),
        width=8,
        bins=4,
    )
    learner = StudentLearner(
        model,
        StudentLearningConfig(width=8, bins=4),
        RoutingConfig(width=8, tile_hw=(3, 3)),
        intervention_config=cfg,
    )
    sample = ScaredA00Sample(
        "dataset_1/keyframe_1/frame_data000000", "final_train", _batch(dtype=torch.float32), {}
    )
    return learner, sample, _target(sample.batch)


def test_warp_direction_roi_units_boundary_and_gradient():
    image = torch.arange(10.0).view(1, 1, 1, 10).expand(1, 1, 5, 10)
    disparity = torch.full((1, 1, 2, 3), 2.0, requires_grad=True)
    warped, valid = warp_right(image, disparity, origin_xy=(4, 1))
    torch.testing.assert_close(warped[0, 0, 0], torch.tensor([2.0, 3.0, 4.0]))
    assert bool(valid.all())
    warped.sum().backward()
    assert torch.isfinite(disparity.grad).all() and bool((disparity.grad != 0).any())
    _, outside = warp_right(image, torch.full_like(disparity, 100.0), origin_xy=(4, 1))
    assert not bool(outside.any())
    support = torch.ones((1, 1, 5, 10), dtype=torch.bool)
    support[..., 2:4] = False
    _, valid = warp_right(image, disparity.detach(), origin_xy=(4, 1), right_support=support)
    assert not bool(valid[..., 0].any())


def test_right_core_projection_is_pre_action_and_shifted():
    mask = torch.zeros((1, 1, 3, 8), dtype=torch.bool)
    mask[..., 3:6] = True
    d = torch.full_like(mask, 2.0, dtype=torch.float32)
    projected = project_left_support(mask, d)
    expected = torch.zeros_like(mask)
    expected[..., 1:4] = True
    assert torch.equal(projected, expected)
    assert not projected.requires_grad


def test_kd_screen_rejects_impossible_correspondence_without_mutating_cache(setup):
    learner, sample, target = setup
    # Fixture has cx_L-cx_R=3 px: raw disparity 4 has positive effective
    # disparity and observable correspondences in the 5-pixel-wide image.
    target = replace(target, disparity=torch.full_like(target.disparity, 4.0))
    before = target.disparity.clone()
    result = screen_target(sample.batch, target, learner.intervention_config)
    assert 0 < result.accepted_count <= result.declared_count
    bad = replace(target, disparity=torch.full_like(target.disparity, 400.0))
    rejected = screen_target(
        sample.batch, bad, replace(learner.intervention_config, kd_min_depth_m=0.01)
    )
    assert rejected.accepted_count == 0 and rejected.outside_right_count > 0
    assert torch.equal(target.disparity, before)


def test_quality_objective_geometry_damage_and_cost_separation():
    cfg = InterventionConfig()
    gain, harm = quality_gain(0.07, 0.063, 0.1, 0.05, 0.0, 0.0, cfg)
    assert gain == pytest.approx(0.6) and not harm
    gain, harm = quality_gain(0.07, 0.063, 0.1, 0.2, 0.0, 0.0, cfg)
    assert gain < 0 and harm
    gain, harm = quality_gain(0.07, 0.063, 0.1, 0.1, 0.0, 0.01, cfg)
    assert gain < 0 and harm


def test_cache_screening_identity_binds_actual_calibration(setup):
    _, sample, _ = setup
    key = sample_cache_key(sample, teacher_identity_hash="a" * 64, split_identity="split")
    changed = replace(sample, calibration_metadata={"protocol_identity": "different-grid"})
    altered = sample_cache_key(changed, teacher_identity_hash="a" * 64, split_identity="split")
    assert key != altered and key.calibration_transform_hash != altered.calibration_transform_hash


def test_seeded_order_preserves_pool_and_global_randomness(setup):
    import random

    _, sample, _ = setup
    samples = [
        replace(sample, sample_id=f"dataset_1/keyframe_1/frame_data{i:06d}") for i in range(32)
    ]
    before = random.getstate()
    first = SeededSampleOrder(samples, 20261006)
    second = SeededSampleOrder(samples, 20261006)
    assert random.getstate() == before
    assert first.indices == second.indices and first.order_hash == second.order_hash
    assert set(first.indices) == set(range(32)) and first.indices != tuple(range(32))
    assert len(first) == 32
    assert first[0].sample_id == samples[first.indices[0]].sample_id


def test_allocator_highest_gain_once_cost_budget_harm_and_stop():
    cfg = InterventionConfig()
    gains = torch.tensor([[0.0, 0.01, 0.02]])
    harm = torch.zeros_like(gains)
    cost = torch.tensor([[0.0, 10.0, 100.0]])
    valid = torch.tensor([True])
    assert select_repairability(gains, harm, cost, valid, 1000.0, cfg).action == Action.GAUSSIAN
    assert select_repairability(gains, harm, cost, valid, 50.0, cfg).action == Action.STEREO
    harm[:, 2] = 0.9
    assert select_repairability(gains, harm, cost, valid, 1000.0, cfg).action == Action.STEREO
    assert select_repairability(gains, harm, cost, valid, 0.0, cfg).action == Action.STOP
    assert (
        select_repairability(torch.zeros_like(gains), harm, cost, valid, 1000.0, cfg).action
        == Action.STOP
    )
    with pytest.raises(ValueError):
        select_repairability(gains, harm, cost, valid, float("nan"), cfg)


def test_allocator_can_penalize_measured_latency_after_budget_gate():
    cfg = InterventionConfig(cost_weight_per_ms=0.01)
    gains = torch.tensor([[0.0, 0.10, 0.11]])
    harm = torch.zeros_like(gains)
    costs = torch.tensor([[0.0, 1.0, 10.0]])
    valid = torch.tensor([True])
    # Raw Gaussian gain is larger, but its measured latency penalty makes
    # stereo the expected-utility winner.
    assert select_repairability(gains, harm, costs, valid, 100.0, cfg).action == Action.STEREO


def test_features_detached_no_gt_and_matching_zero_identity(setup):
    learner, sample, _ = setup
    state = learner.state(sample)
    regions = grid_regions((5, 5), (3, 3), 4)
    features = repairability_features(
        state,
        regions,
        learner.intervention_config,
        scale_unit_m=learner.model.config.context_scale_m,
    )
    assert (
        features.shape == (4, 26) and not features.requires_grad and torch.isfinite(features).all()
    )
    result = learner.actions.execute(state, regions[1], Action.STEREO, learner.model)
    torch.testing.assert_close(result.disparity, state.base.target.disparity)
    torch.testing.assert_close(result.field.means3d, state.base.gaussians.means3d)
    shifted = replace(state.batch, right=state.batch.right.flip(-1))
    assert torch.isfinite(stereo_consistency_loss(replace(state, batch=shifted), result.disparity))


def test_joint_updates_all_owners_empty_kd_continues_and_no_teacher_grad(setup):
    learner, sample, target = setup
    bad = replace(target, disparity=torch.full_like(target.disparity, 400.0, requires_grad=True))
    before = [
        {k: v.clone() for k, v in m.state_dict().items()}
        for m in (
            learner.model.stereo_student,
            learner.model.head,
            learner.actions.stereo,
            learner.actions.gaussian,
            learner.selector,
        )
    ]
    row = learner.step(sample, joint=True, teacher_target=bad)
    assert row["kd_empty"] == 1 and row["kd"] == 0 and bad.disparity.grad is None
    for module, state in zip(
        (
            learner.model.stereo_student,
            learner.model.head,
            learner.actions.stereo,
            learner.actions.gaussian,
            learner.selector,
        ),
        before,
        strict=True,
    ):
        assert any(not torch.equal(v, state[k]) for k, v in module.state_dict().items())
    assert row["grad_norm_selector"] > 0 and row["grad_norm_stereo_repair"] > 0


def test_zero_gaussian_action_is_exact_on_trained_base_and_still_learns(setup):
    learner, sample, _ = setup
    with torch.no_grad():
        learner.model.head.output.bias.copy_(torch.tensor([0.1, 0.2, -0.1, 0.02, 0.03, 0.01, -0.2]))
    state = learner.state(sample)
    region = grid_regions((5, 5), (5, 5), learner.routing.halo)[0]
    result = learner.actions.execute(state, region, Action.GAUSSIAN, learner.model)
    for item in fields(result.field):
        before, after = getattr(state.base.gaussians, item.name), getattr(result.field, item.name)
        assert (before is None and after is None) or torch.equal(before, after)
    loss = (
        (result.field.rotations * torch.tensor([0.4, 0.3, 0.2, 0.1])).sum()
        + result.field.scales.sum()
        + result.field.opacities.sum()
    )
    loss.backward()
    grad = learner.actions.gaussian.output.bias.grad
    assert grad is not None and torch.isfinite(grad).all() and bool((grad != 0).all())
    assert learner.model.head.output.bias.grad is not None
    assert torch.isfinite(learner.model.head.output.bias.grad).all()


def test_zero_identity_keeps_nonzero_composition_and_single_base_gradient(setup):
    learner, sample, _ = setup
    base = learner.state(sample).base.gaussians
    drift = torch.full_like(base.scales, 1e-6, requires_grad=True)
    changed = replace(base, scales=base.scales + drift)
    alpha = torch.zeros_like(base.opacities)
    alpha[:, -1] = 0.01
    residuals = AttributeResiduals(
        torch.zeros_like(base.scales), torch.zeros_like(base.scales), alpha
    )
    result = preserve_zero_gaussian_residual(base, changed, residuals)
    assert torch.equal(result.scales[:, :-1], base.scales[:, :-1])
    assert torch.equal(result.scales[:, -1], changed.scales[:, -1])
    drift_grad, base_grad = torch.autograd.grad(result.scales.sum(), (drift, base.scales))
    torch.testing.assert_close(drift_grad, torch.ones_like(drift))
    torch.testing.assert_close(base_grad, torch.ones_like(base.scales))


def test_selector_loss_targets_detached_and_ranking_learns():
    model = RepairabilitySelector(8)
    features = torch.randn(3, 26)
    gains = torch.tensor([[0.0, 0.3, 0.8], [0.0, 0.7, -0.2], [0.0, -0.1, -0.2]], requires_grad=True)
    harms = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]], requires_grad=True)
    loss = model.loss(features, gains, harms, InterventionConfig())
    loss.backward()
    assert gains.grad is None and harms.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_checkpoint_protocol_pinned_resume_and_inference_no_updates(setup, tmp_path):
    learner, sample, target = setup
    learner.step(sample, joint=True, teacher_target=target)
    identity = {
        "source_sha256": "source",
        "manifest_hash": "manifest",
        "role_hash": "role",
        "renderer_identity": "analytic",
        "run_kind": "full_manifest",
    }
    cursor = {"epoch": 1, "next_sample_index": 0, "samples_seen": 1, "optimizer_steps": 1}
    path = tmp_path / "new.pt"
    save_student_checkpoint(path, learner, identity, cursor)
    assert load_student_checkpoint(path, learner, identity, 1) == cursor
    legacy = StudentLearner(copy.deepcopy(learner.model), learner.config, learner.routing)
    with pytest.raises(ValueError, match="protocol"):
        load_student_checkpoint(path, legacy, identity, 1)
    before = [
        {k: v.clone() for k, v in m.state_dict().items()}
        for m in (learner.model, learner.actions, learner.selector)
    ]
    result = learner.infer(sample, selected=True)
    assert "remaining_ms" in result.diagnostics
    for module, state in zip(
        (learner.model, learner.actions, learner.selector), before, strict=True
    ):
        assert all(torch.equal(v, state[k]) for k, v in module.state_dict().items())


def test_readonly_cache_refuses_writes_and_does_not_call_missing_producer(tmp_path):
    root = tmp_path / "cache"
    key = TargetCacheKey(
        "dataset_1/keyframe_1/frame_000", "split", "final_train", "image", "calib", "teacher"
    )
    writer = TargetCache(root)
    writer.put(key, _cache_target())
    before = (root / "manifest.json").read_bytes()
    reader = TargetCache(root, read_only=True)
    assert reader.get(key) is not None
    with pytest.raises(TargetCacheError, match="read-only"):
        reader.put(key, _cache_target())
    called = []

    def produce():
        called.append(True)
        return _cache_target()

    with pytest.raises(TargetCacheError, match="producer"):
        reader.get_or_compute(replace(key, image_transform_hash="missing"), produce)
    assert not called and (root / "manifest.json").read_bytes() == before


def test_model_only_warmstart_preserves_new_heads_and_optimizers(setup, tmp_path):
    learner, _, _ = setup
    legacy = StudentLearner(copy.deepcopy(learner.model), learner.config, learner.routing)
    with torch.no_grad():
        for p in legacy.model.parameters():
            p.add_(0.02)
    identity = {
        "source_sha256": "old-source",
        "manifest_hash": "manifest",
        "role_hash": "role",
        "renderer_identity": "analytic",
        "run_kind": "full_manifest",
    }
    cursor = {"epoch": 0, "next_sample_index": 0, "samples_seen": 0, "optimizer_steps": 0}
    path = tmp_path / "legacy.pt"
    save_student_checkpoint(path, legacy, identity, cursor)
    actions = {k: v.clone() for k, v in learner.actions.state_dict().items()}
    selector = {k: v.clone() for k, v in learner.selector.state_dict().items()}
    result = initialize_student_model(path, learner)
    assert result["source_sha256"] == "old-source"
    assert all(
        torch.equal(v, legacy.model.state_dict()[k]) for k, v in learner.model.state_dict().items()
    )
    assert all(torch.equal(v, actions[k]) for k, v in learner.actions.state_dict().items())
    assert all(torch.equal(v, selector[k]) for k, v in learner.selector.state_dict().items())
    assert (
        not learner.model_optimizer.state
        and not learner.action_optimizer.state
        and not learner.selector_optimizer.state
    )
