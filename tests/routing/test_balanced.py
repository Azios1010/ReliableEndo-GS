"""Balanced routing contracts: gradients, units, local/global masks and resume."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.actions.repair import Action
from reliable_endo_gs.config.intervention import InterventionConfig, load_intervention_config
from reliable_endo_gs.contracts.rendering import RenderOutput
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.reconstruction import ReferenceConfig
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.regions.grid import grid_regions
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.config import RoutingConfig
from reliable_endo_gs.routing.context import (
    BALANCED_FEATURE_NAMES,
)
from reliable_endo_gs.routing.normalization import TrainingNormalization
from reliable_endo_gs.training.gradient_balance import GradientBalance
from reliable_endo_gs.training.quality_components import (
    component_targets,
    local_reconstruction_loss,
    measure_quality,
)
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    read_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_config import StudentLearningConfig
from reliable_endo_gs.training.student_learning import StudentLearner
from tests.architecture.test_a04_reference_reconstruction import _batch, _target
from tests.routing.test_frozen_a04 import AnalyticRenderer


@pytest.fixture
def setup():
    torch.manual_seed(3)
    cfg = InterventionConfig(
        schema="intervention_selection.v2",
        normalization_frames=2,
        normalization_rows=32,
        matching_width=4,
        search_radius_px=4,
        search_step_px=2,
        kd_min_depth_m=0.0,
        gradient_diagnostics_every=1,
    )
    model = StudentReconstructionModel(
        renderer=AnalyticRenderer(),
        config=replace(ReferenceConfig(), head_hidden_dims=(8, 8)),
        width=8,
        bins=4,
    )
    learner = StudentLearner(
        model,
        StudentLearningConfig(width=8, bins=4, student_epochs=0, joint_epochs=3),
        RoutingConfig(width=8, tile_hw=(3, 3)),
        intervention_config=cfg,
    )
    sample = ScaredA00Sample(
        "dataset_1/keyframe_1/frame_data000000", "final_train", _batch(dtype=torch.float32), {}
    )
    return learner, sample, _target(sample.batch)


def test_v1_hash_serialization_stays_identical():
    cfg = load_intervention_config(Path("configs/routing/intervention_selection.yaml"))
    assert cfg.config_hash == "3db4be52b6f227b6cbbaf79564510c2f6162f4db7109b231842c75e7a543964f"
    assert "normalization_frames" not in cfg.to_dict()
    with pytest.raises(ValueError):
        InterventionConfig(schema="bad")


def test_stereo_priority_neutral_preserves_v2_identity():
    cfg = load_intervention_config(Path("configs/routing/intervention_balanced.yaml"))
    assert cfg.config_hash == "9abdf5e35f91b62a1bb49efa61c0209b0565476d69ed61542f5c4894b1aa5266"
    assert "stereo_training_priority" not in cfg.to_dict()
    prioritized = replace(cfg, stereo_training_priority=3.0)
    assert prioritized.to_dict()["stereo_training_priority"] == 3.0
    assert prioritized.config_hash != cfg.config_hash
    assert InterventionConfig(**prioritized.to_dict()) == prioritized
    with pytest.raises(ValueError):
        replace(cfg, stereo_training_priority=0.0)


def test_stereo_priority_after_balance_survives_inverse_norm():
    cfg = InterventionConfig(schema="intervention_selection.v2")
    balance = GradientBalance(("rgb", "warp"), cfg)
    p = torch.nn.Parameter(torch.ones(2))
    losses = (2 * p[0], 7 * p[1])
    pending, _ = balance.measure(losses, (p,))
    one = balance.weighted_loss(losses, torch.tensor([1.0, 1.0]), pending)
    three = balance.weighted_loss(losses, torch.tensor([1.0, 3.0]), pending)
    g1 = torch.autograd.grad(one, p, retain_graph=True)[0]
    g3 = torch.autograd.grad(three, p)[0]
    assert g3[0] == g1[0] and g3[1] == pytest.approx(3 * g1[1])
    assert balance.observations == 0  # priority application never updates EMA
    with pytest.raises(ValueError):
        balance.weighted_loss(losses, torch.tensor([float("nan"), 1.0]), pending)


def test_component_targets_global_is_independent_of_local_aux():
    cfg = InterventionConfig(schema="intervention_selection.v2")
    base = dict(
        global_rgb=0.07,
        global_warp=0.1,
        local_rgb=0.07,
        local_warp=0.1,
        outside_rgb=0.07,
        outside_warp=0.1,
    )
    after = {
        **base,
        "global_rgb": 0.063,
        "global_warp": 0.09,
        "local_rgb": 0.035,
        "local_warp": 0.05,
    }
    g, local, h = component_targets(base, after, cfg)
    assert g == pytest.approx(0.2) and local == pytest.approx(1.0) and not h
    altered = {**after, "local_rgb": 0.001}
    assert component_targets(base, altered, cfg)[0] == pytest.approx(g)
    altered = {**after, "outside_rgb": 0.08}
    assert component_targets(base, altered, cfg)[2]


def test_normalizer_train_only_freeze_constants_common_gain_and_resume():
    cfg = InterventionConfig(
        schema="intervention_selection.v2", normalization_frames=2, normalization_rows=8
    )
    norm = TrainingNormalization(3, (2,), cfg)
    x = torch.tensor([[1.0, 1e-4, 1.0], [3.0, 3e-4, 1.0]])
    labels = torch.tensor([[0.0, 0.1, 0.2], [0.0, -0.1, 0.05]])
    with pytest.raises(ValueError):
        norm.observe(x, labels, labels, sample_id="dataset_8/keyframe_1/a", role="final_test")
    norm.observe(x, labels, labels, sample_id="dataset_1/keyframe_1/a", role="final_train")
    assert not norm.fitted
    norm.observe(x, labels, labels, sample_id="dataset_2/keyframe_1/a", role="final_train")
    assert norm.fitted
    out = norm(x)
    torch.testing.assert_close(out[:, 0], out[:, 1])
    assert torch.equal(out[:, 2], x[:, 2])
    before = copy.deepcopy(norm.state_dict())
    norm.observe(x * 100, labels, labels, sample_id="dataset_3/keyframe_1/a", role="final_train")
    assert all(torch.equal(v, before[k]) for k, v in norm.state_dict().items())
    clone = TrainingNormalization(3, (2,), cfg)
    clone.load_state_dict(before)
    clone.validate_state()
    torch.testing.assert_close(norm(x), clone(x))
    constant = TrainingNormalization(2, (), replace(cfg, normalization_frames=1))
    constant.observe(
        torch.ones(2, 2), labels, labels, sample_id="dataset_1/keyframe_1/a", role="final_train"
    )
    assert torch.equal(constant(torch.ones(2, 2) * 100), torch.zeros(2, 2))


def test_balancing_scales_actual_shared_gradient_and_ignores_zero():
    cfg = InterventionConfig(
        schema="intervention_selection.v2", balance_minimum=0.01, balance_maximum=1000.0
    )
    balance = GradientBalance(("small", "large", "zero"), cfg)
    p = torch.nn.Parameter(torch.ones(3))
    losses = (p.sum(), 1000 * p.sum(), p.sum() * 0)
    pending, diag = balance.measure(losses, (p,))
    assert diag["shared_grad_large"] / diag["shared_grad_small"] == pytest.approx(1000)
    assert pending[1, 1] == pytest.approx(0.01)  # explicit cap, not unbounded inverse norm
    assert pending[1, 2] == 1
    assert balance.observations == 0
    balance.commit(pending)
    balance.validate_state()
    assert balance.observations == 1


def test_single_region_warmup_waits_for_two_rows():
    cfg = InterventionConfig(schema="intervention_selection.v2", normalization_frames=1)
    norm = TrainingNormalization(2, (), cfg)
    features = torch.ones(1, 2)
    labels = torch.zeros(1, 3)
    norm.observe(features, labels, labels, sample_id="dataset_1/keyframe_1/a", role="final_train")
    norm.validate_state()
    assert not norm.fitted
    norm.observe(features, labels, labels, sample_id="dataset_1/keyframe_1/b", role="final_train")
    norm.validate_state()
    assert norm.fitted
    with pytest.raises(ValueError, match="capacity"):
        replace(cfg, normalization_rows=1)


def test_harm_encoder_has_no_gain_parameter_gradient(setup):
    learner, _, _ = setup
    selector = learner.selector
    assert isinstance(selector, BalancedSelector)
    x = torch.randn(4, len(BALANCED_FEATURE_NAMES))
    target = torch.tensor([[0.0, 0.1, 0.2]]).expand(4, -1)
    labels = torch.zeros_like(target)
    losses, _ = selector.objective(x, target, target, labels)
    grads = torch.autograd.grad(
        losses["harm"],
        tuple(selector.gain_network.parameters()),
        allow_unused=True,
        retain_graph=True,
    )
    assert all(g is None for g in grads)
    grads = torch.autograd.grad(
        losses["global"], tuple(selector.harm_network.parameters()), allow_unused=True
    )
    assert all(g is None for g in grads)


def test_raw_gain_decoder_stop_and_common_action_scale(setup):
    learner, _, _ = setup
    selector = learner.selector
    with torch.no_grad():
        selector.gain_network[-1].bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    g, local, h = selector.outputs(torch.zeros(2, len(BALANCED_FEATURE_NAMES)))
    assert g[0].tolist() == pytest.approx([0.0, 0.05, 0.1])
    assert local[0].tolist() == pytest.approx([0.0, 0.15, 0.2])
    assert h[0, 0] == 0


def test_context_evidence_no_gt_and_spatial_global_missing_flags(setup):
    learner, sample, _ = setup
    state = learner.state(sample)
    evidence = state.base.matching_evidence
    assert set(evidence) == {"cost_margin", "cost_entropy", "feature_energy"}
    assert all(
        v.shape == sample.batch.support.shape and not v.requires_grad for v in evidence.values()
    )
    regions = grid_regions((5, 5), (3, 3), learner.routing.halo)
    feature = learner.selection_features(state, regions)
    assert (
        feature.shape == (4, len(BALANCED_FEATURE_NAMES))
        and torch.isfinite(feature).all()
        and not feature.requires_grad
    )
    assert feature[:, BALANCED_FEATURE_NAMES.index("matching_evidence_available")].eq(1).all()
    assert (
        feature[:, BALANCED_FEATURE_NAMES.index("visibility_available")].eq(0).all()
    )  # analytic output is pixel visibility, not N radii
    assert not set(learner.model.parameters()) & set(learner.selector.parameters())


def test_local_rgb_uses_projected_right_support(setup):
    learner, sample, _ = setup
    state = learner.state(sample)
    region = grid_regions((5, 5), (3, 3), learner.routing.halo)[1]
    stop = learner.actions.execute(state, region, Action.STOP, learner.model)
    value = local_reconstruction_loss(state, stop, region)
    assert torch.isfinite(value)
    values = measure_quality(state, stop, region).values()
    assert set(values) == {
        "global_rgb",
        "global_warp",
        "local_rgb",
        "local_warp",
        "outside_rgb",
        "outside_warp",
    }
    assert component_targets(values, values, learner.intervention_config) == (0.0, 0.0, False)


def test_local_rgb_excludes_left_rectangle_in_right_view(setup):
    learner, sample, _ = setup
    state = learner.state(sample)
    # A left core x=3:5 corresponds to right x=1:3 under raw disparity 2.
    state = replace(
        state,
        base=replace(
            state.base,
            target=replace(
                state.base.target, disparity=torch.full_like(state.base.target.disparity, 2.0)
            ),
        ),
    )
    region = grid_regions((5, 5), (3, 3), learner.routing.halo)[1]
    stop = learner.actions.execute(state, region, Action.STOP, learner.model)
    correct = sample.batch.right.clone()
    correct[:, :, 0:3, 1] += 0.2
    incorrect = sample.batch.right.clone()
    incorrect[:, :, 0:3, 4] += 0.2
    left = RenderOutput(sample.batch.left.clone())
    included = replace(stop, left=left, right=RenderOutput(correct))
    excluded = replace(stop, left=left, right=RenderOutput(incorrect))
    assert local_reconstruction_loss(state, included, region) > 0.01
    assert local_reconstruction_loss(state, excluded, region) == pytest.approx(0.001)


def test_harm_scaling_cannot_change_gain_encoder_gradient(setup):
    learner, _, _ = setup
    selector = learner.selector
    x = torch.randn(4, len(BALANCED_FEATURE_NAMES))
    target = torch.tensor([[0.0, 0.1, 0.2]]).expand(4, -1)
    loss, _ = selector.objective(x, target, target, torch.zeros_like(target))
    params = tuple(selector.gain_network.parameters())
    reference = torch.autograd.grad(loss["global"], params, retain_graph=True)
    amplified = torch.autograd.grad(loss["global"] + 1000 * loss["harm"], params)
    assert all(torch.equal(a, b) for a, b in zip(reference, amplified, strict=True))


def test_balanced_router_can_fit_two_action_synthetic_labels():
    torch.manual_seed(5)
    cfg = InterventionConfig(
        schema="intervention_selection.v2", normalization_frames=1, normalization_rows=8
    )
    router = BalancedSelector(8, cfg)
    x = torch.zeros(8, len(BALANCED_FEATURE_NAMES))
    x[:, 0] = torch.tensor([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    targets = torch.tensor([[0.0, 0.02, 0.1]] * 4 + [[0.0, 0.1, 0.02]] * 4)
    router.normalization.observe(
        x, targets, targets, sample_id="dataset_1/keyframe_1/synthetic", role="final_train"
    )
    optimizer = torch.optim.Adam(router.parameters(), lr=0.02)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        losses, _ = router.objective(x, targets, targets, torch.zeros_like(targets))
        total = losses["global"] + losses["local"] + losses["ranking"] + losses["harm"]
        total.backward()
        optimizer.step()
    prediction, risk = router.predict(x)
    assert torch.equal(prediction.argmax(1), targets.argmax(1))
    assert risk[:, 1:].max() < 0.5


def test_cold_normalization_blocks_selected_inference_then_eval_is_immutable(setup):
    learner, sample, target = setup
    with pytest.raises(ValueError, match="fitted"):
        learner.infer(sample, selected=True)
    for i in range(2):
        learner.step(sample, joint=True, teacher_target=target, position=i)
    before = copy.deepcopy(learner.selector.state_dict())
    learner.infer(sample, selected=True)
    assert all(torch.equal(v, before[k]) for k, v in learner.selector.state_dict().items())


def test_all_regions_joint_update_records_and_normalization_commit(setup):
    learner, sample, target = setup
    bad = replace(target, disparity=torch.full_like(target.disparity, 400.0, requires_grad=True))
    records = []
    before = [
        copy.deepcopy(m.state_dict())
        for m in (
            learner.model.stereo_student,
            learner.model.head,
            learner.actions.stereo,
            learner.actions.gaussian,
            learner.selector,
        )
    ]
    for i in range(3):
        records.append(learner.step(sample, joint=True, teacher_target=bad, position=i))
    row = records[-1]
    assert row["kd_empty"] == 1 and bad.disparity.grad is None
    assert row["selector_regions"] == len(row["interventions"]) == 4
    assert len(row["features_raw"]) == 4
    assert all(len(x["actions"]) == 3 for x in row["interventions"])
    assert learner.selector.normalization.fitted
    for m, b in zip(
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
        assert any(not torch.equal(v, b[k]) for k, v in m.state_dict().items())
    assert (
        learner.selector_balancer.observations == 3 and learner.objective_balancer.observations == 3
    )
    assert row["selector_global_loss"] >= 0 and row["optimizer_update_norm_selector"] > 0


def test_checkpoint_pins_stats_balancers_and_rejects_corrupt_state(setup, tmp_path):
    learner, sample, target = setup
    for i in range(2):
        learner.step(sample, joint=True, teacher_target=target, position=i)
    identity = dict(
        source_sha256="source",
        manifest_hash="manifest",
        role_hash="role",
        renderer_identity="analytic",
        run_kind="full_manifest",
    )
    cursor = dict(epoch=0, next_sample_index=2, samples_seen=2, optimizer_steps=2)
    path = tmp_path / "checkpoint.pt"
    save_student_checkpoint(path, learner, identity, cursor)
    raw = read_student_checkpoint(path)
    assert raw["schema"] == "student_joint_intervention_checkpoint.v2"
    clone = StudentLearner(
        copy.deepcopy(learner.model),
        learner.config,
        learner.routing,
        intervention_config=learner.intervention_config,
    )
    assert load_student_checkpoint(path, clone, identity, 4) == cursor
    assert all(
        torch.equal(v, clone.selector.state_dict()[k])
        for k, v in learner.selector.state_dict().items()
    )
    assert torch.equal(learner.selector_balancer.ema, clone.selector_balancer.ema)
    raw["selector"]["normalization.scale"].zero_()
    bad = tmp_path / "bad.pt"
    torch.save(raw, bad)
    before = copy.deepcopy(clone.model.state_dict())
    with pytest.raises(ValueError):
        load_student_checkpoint(bad, clone, identity, 4)
    assert all(torch.equal(v, before[k]) for k, v in clone.model.state_dict().items())
