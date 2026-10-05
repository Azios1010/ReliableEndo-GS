"""Actual canonical geometry, repair gradients, selection, and resume on CPU."""

from __future__ import annotations

import copy
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from reliable_endo_gs.actions.repair import Action, RepairState, merge_field, primitive_core
from reliable_endo_gs.cli.phase2 import build_parser, run
from reliable_endo_gs.contracts.rendering import RenderOutput
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.gaussians import GeometryPriorPolicy
from reliable_endo_gs.reconstruction import ReferenceConfig, ReferenceReconstructionModel
from reliable_endo_gs.regions.grid import grid_regions
from reliable_endo_gs.rendering.field import FieldRenderRequest
from reliable_endo_gs.routing.config import RoutingConfig, load_routing_config
from reliable_endo_gs.routing.selector import ActionSelector, region_features, select_action
from reliable_endo_gs.training.action_checkpoint import (
    load_action_checkpoint,
    save_action_checkpoint,
)
from reliable_endo_gs.training.action_learning import ActionLearner
from reliable_endo_gs.training.action_run import frozen_state, run_action_training
from reliable_endo_gs.training.interventions import reconstruction_loss, utility
from tests.architecture.test_a04_reference_reconstruction import _batch, _FakeTeacher, _target


class AnalyticRenderer:
    identity = "phase2-analytic-gradient-test"

    def render(self, request: FieldRenderRequest) -> RenderOutput:
        field = request.field
        # Deliberately differentiable in depth and every Gaussian attribute;
        # this validates ownership/chain rule, not native rasterizer parity.
        value = (
            field.means3d[..., 2].mean() * 0.05
            + field.scales.mean() * 0.3
            + field.opacities.mean() * 0.2
            + field.rotations[..., 1:].mean() * 0.02
        )
        image = value.reshape(1, 1, 1, 1).expand(1, 3, *request.image_hw)
        return RenderOutput(image, visibility=torch.ones_like(image[:, :1]))


@pytest.fixture
def setup():
    batch = _batch(dtype=torch.float32)
    config = replace(
        ReferenceConfig(),
        head_hidden_dims=(8, 8),
        prior_policy=GeometryPriorPolicy(alpha_valid=0.5),
    )
    model = ReferenceReconstructionModel(
        teacher=_FakeTeacher(), renderer=AnalyticRenderer(), config=config
    )
    learner = ActionLearner(model, RoutingConfig(tile_hw=(3, 3), width=8))
    target = _target(batch)
    target = replace(target, disparity=target.disparity.float())
    with torch.no_grad():
        base = model(batch, target=target, sample_ids=("dataset_1/keyframe_1/frame_data000000",))
    return learner, RepairState(batch, base)


def test_grid_unique_core_coverage_and_edges():
    image = torch.zeros(1, 3, 5, 7)
    regions = grid_regions((5, 7), (3, 4), 1)
    assert len(regions) == 4
    assert torch.equal(
        torch.stack([r.mask(image).long() for r in regions]).sum(0),
        torch.ones_like(image[:, :1], dtype=torch.long),
    )
    assert regions[-1].core == (3, 4, 5, 7)
    assert regions[-1].halo == (2, 3, 5, 7)


@pytest.mark.parametrize("tile,halo", [((0, 2), 1), ((1, 1), -1), ((True, 2), 1)])
def test_bad_grid(tile, halo):
    with pytest.raises(ValueError):
        grid_regions((5, 7), tile, halo)


def test_stop_is_exact_identity_without_render(setup):
    learner, state = setup
    result = learner.actions.execute(
        state, learner.regions(state)[0], Action.STOP, learner.reference
    )
    assert result.field is state.base.gaussians
    assert result.disparity is state.base.target.disparity
    assert result.left is state.base.render_left
    assert result.right is state.base.render_right
    assert result.changed_fields == ()


def test_zero_repair_identity_before_learning(setup):
    learner, state = setup
    errors = learner.check_zero_repairs(state)
    assert max(errors.values()) < 1e-6


def test_gaussian_zero_does_not_reclip_learned_near_unit_opacity(setup):
    learner, state = setup
    field = replace(
        state.base.gaussians, opacities=torch.full_like(state.base.gaussians.opacities, 0.99998)
    )
    state = replace(state, base=replace(state.base, gaussians=field))
    region = grid_regions((5, 5), (5, 5), 4)[0]
    result = learner.actions.execute(state, region, Action.GAUSSIAN, learner.reference)
    torch.testing.assert_close(result.field.opacities, field.opacities, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("action", [Action.STEREO, Action.GAUSSIAN])
def test_repairs_core_only_gradients_and_immutable_base(setup, action):
    learner, state = setup
    region = learner.regions(state)[0]
    snapshot = copy.deepcopy(state.base.gaussians)
    with torch.no_grad():
        if action == Action.STEREO:
            learner.actions.stereo[-1].bias.fill_(0.2)
        else:
            learner.actions.gaussian.output.bias.fill_(0.2)
    result = learner.actions.execute(state, region, action, learner.reference)
    mask = primitive_core(state, region)
    for item in fields(snapshot):
        name = item.name
        original, after = getattr(snapshot, name), getattr(result.field, name)
        if original is not None:
            assert torch.equal(original[~mask], after[~mask])
            assert torch.equal(original, getattr(state.base.gaussians, name))
    if action == Action.GAUSSIAN:
        assert torch.equal(result.field.means3d, snapshot.means3d)
        assert torch.equal(result.disparity, state.base.target.disparity)
        assert not torch.equal(result.field.scales[mask], snapshot.scales[mask])
    else:
        assert not torch.equal(result.field.means3d[mask], snapshot.means3d[mask])
        pixel_mask = region.mask(state.batch.left)
        assert torch.equal(result.disparity[~pixel_mask], state.base.target.disparity[~pixel_mask])
    reconstruction_loss(state, result).backward()
    branch = learner.actions.stereo if action == Action.STEREO else learner.actions.gaussian
    assert any(p.grad is not None and bool((p.grad != 0).any()) for p in branch.parameters())
    assert all(p.grad is None and not p.requires_grad for p in learner.reference.parameters())


def test_warmup_updates_both_branches_but_reference_unchanged(setup):
    learner, state = setup
    before = {k: v.clone() for k, v in learner.reference.state_dict().items()}
    actions = {k: v.clone() for k, v in learner.actions.state_dict().items()}
    result = learner.warmup_step(state)
    assert set(result["repair_losses"]) == {"stereo", "gaussian"}
    for prefix in ("stereo", "gaussian"):
        assert any(
            not torch.equal(v, learner.actions.state_dict()[k])
            for k, v in actions.items()
            if k.startswith(prefix)
        )
    assert all(torch.equal(v, learner.reference.state_dict()[k]) for k, v in before.items())


def test_harness_labels_features_and_frozen_repairs(setup):
    learner, state = setup
    learner.warmup_step(state)
    before = {k: v.clone() for k, v in learner.actions.state_dict().items()}
    features = region_features(state, (learner.regions(state)[0],)).clone()
    result = learner.selector_step(state, 0)
    assert result["features"] == features.tolist()
    assert result["interventions"][0]["utility"] == 0
    assert result["interventions"][0]["cost_ms"] == 0
    assert all(row["cost_ms"] > 0 for row in result["interventions"][1:])
    assert all(torch.equal(v, learner.actions.state_dict()[k]) for k, v in before.items())
    assert all(not p.requires_grad for p in learner.actions.parameters())
    assert any(p.grad is not None for p in learner.selector.parameters())
    assert learner.cost_observations == 1
    assert len(features[0]) == 18


def test_harness_rejects_moving_action_labels(setup):
    learner, state = setup
    with pytest.raises(ValueError, match="freeze repair"):
        learner.harness.collect(
            state, learner.regions(state)[0], learner.actions, learner.reference
        )


def test_canonical_utility():
    assert utility(0.2, 0.1, 10.0, 0.001) == pytest.approx(0.09)
    assert utility(0.1, 0.1, 0.0, 0.001) == 0
    with pytest.raises(ValueError):
        utility(0.1, 0.2, -1.0, 0.0)


@pytest.mark.parametrize(
    "budget,valid,expected",
    [
        (0.0, True, Action.STOP),
        (5.0, True, Action.GAUSSIAN),
        (4.0, True, Action.STOP),
        (100.0, False, Action.STOP),
    ],
)
def test_budget_selection(budget, valid, expected):
    utilities = torch.tensor([[0.0, 0.2, 0.1]])
    costs = torch.tensor([[0.0, 20.0, 5.0]])
    result = select_action(utilities, costs, torch.tensor([valid]), budget)
    assert result.action == expected
    assert result.estimated_ms <= budget


def test_negative_utility_and_nonfinite_stop():
    assert (
        select_action(
            torch.tensor([[0.0, -0.1, -0.2]]),
            torch.tensor([[0.0, 1.0, 1.0]]),
            torch.tensor([True]),
            100,
        ).action
        == Action.STOP
    )
    with pytest.raises(ValueError):
        select_action(
            torch.tensor([[0.0, float("nan"), 1.0]]), torch.zeros(1, 3), torch.tensor([True]), 100
        )


def test_deployment_executes_selected_action_only(setup, monkeypatch):
    learner, state = setup
    learner.cost_max_ms[:] = torch.tensor([0.0, 1.0, 1.0])
    learner.cost_observations = 1
    with torch.no_grad():
        learner.selector.network[-1].bias[:] = torch.tensor([0.2, 0.1])
    calls = []
    original = learner.actions.execute

    def record(*args):
        calls.append(args[2])
        return original(*args)

    monkeypatch.setattr(learner.actions, "execute", record)
    result = learner.infer(state)
    assert result["decision"]["action"] == Action.STEREO
    assert calls == [Action.STEREO, Action.STOP]


def test_checkpoint_identity_cursor_rng_and_no_reference_weights(setup, tmp_path):
    learner, state = setup
    learner.warmup_step(state)
    learner.selector_step(state, 0)
    cursor = {
        "epoch": 0,
        "next_sample_index": 1,
        "samples_seen": 1,
        "optimizer_steps": 1,
        "skips": 0,
        "consecutive_skips": 0,
    }
    path = tmp_path / "phase2.pt"
    save_action_checkpoint(path, learner, {"base": "immutable"}, cursor)
    expected = torch.rand(3)
    payload = torch.load(path, weights_only=True)
    assert payload["phase1_in_checkpoint"] is False
    assert not any("reference" in key or "teacher" in key for key in payload["actions"])
    assert load_action_checkpoint(path, learner, {"base": "immutable"}, 2) == cursor
    torch.testing.assert_close(expected, torch.rand(3))
    with pytest.raises(ValueError, match="mismatch"):
        load_action_checkpoint(path, learner, {"base": "changed"}, 2)
    with pytest.raises(FileExistsError):
        save_action_checkpoint(path, learner, {"base": "immutable"}, cursor)


def test_fixed_masks_penalize_worse_render(setup):
    learner, state = setup
    outcome = learner.actions.execute(
        state, learner.regions(state)[0], Action.STOP, learner.reference
    )
    bad = replace(
        outcome,
        left=RenderOutput(
            torch.ones_like(outcome.left.image),
            visibility=torch.zeros_like(outcome.left.image[:, :1]),
        ),
    )
    assert reconstruction_loss(state, bad) > reconstruction_loss(state, outcome)


def test_optional_covariance_changes_rejected(setup):
    _, state = setup
    with pytest.raises(ValueError, match="availability"):
        merge_field(
            state.base.gaussians,
            replace(state.base.gaussians, cov_surface=None),
            state.base.gaussians.valid_mask,
        )


def test_invalid_support_is_stop_and_no_oracle_leakage(setup):
    learner, state = setup
    batch = replace(state.batch, support=torch.zeros_like(state.batch.support))
    state = replace(state, batch=batch)
    features = region_features(state, learner.regions(state))
    assert features.shape == (4, 18)
    assert torch.isfinite(features).all()
    learner.cost_max_ms[:] = torch.tensor([0.0, 1.0, 1.0])
    learner.cost_observations = 1
    # Without an observation mask, quality assessment correctly fails instead
    # of manufacturing a zero loss; allocator itself emits exact STOP.
    decision = select_action(
        learner.selector(features),
        learner.cost_max_ms.expand(4, -1),
        torch.zeros(4, dtype=torch.bool),
        100,
    )
    assert decision.action == Action.STOP


def test_config_identity_and_unknown_keys(tmp_path):
    config = load_routing_config(Path("configs/routing/frozen_a04.yaml"))
    assert len(config.config_hash) == 64
    assert config.config_hash != replace(config, budget_ms=2000).config_hash
    path = tmp_path / "bad.yaml"
    path.write_text("unknown: 1\n")
    with pytest.raises(ValueError):
        load_routing_config(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"halo": -1},
        {"width": True},
        {"cost_weight_per_ms": float("nan")},
        {"learning_rate": 0},
        {"schema": "unknown"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        RoutingConfig(**kwargs)


def test_cli_parser_and_test_learning_guard(tmp_path):
    arguments = build_parser().parse_args(
        [
            "train",
            "--data-root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--reference-checkpoint",
            str(tmp_path / "base.pt"),
            "--output-dir",
            str(tmp_path / "run"),
            "--data-role",
            "final_test",
        ]
    )
    with pytest.raises(ValueError, match="cannot consume final_test"):
        run(arguments)


def test_run_staged_schedule_and_resume(setup, tmp_path, monkeypatch):
    learner, state = setup
    samples = [
        ScaredA00Sample(f"dataset_1/keyframe_1/frame_data00000{i}", "final_train", state.batch, {})
        for i in range(2)
    ]
    source = SimpleNamespace(
        config=learner.reference.config, _dataset=samples, renderer=learner.reference.renderer
    )
    monkeypatch.setattr("reliable_endo_gs.training.action_run.frozen_state", lambda *_: state)
    learner.config = replace(learner.config, checkpoint_every=1)
    first = run_action_training(source, learner, {}, tmp_path / "first", max_steps=1)
    assert first["cursor"]["next_sample_index"] == 1
    result = run_action_training(
        source, learner, {}, tmp_path / "resumed", resume=Path(first["checkpoint"])
    )
    assert result["cursor"]["epoch"] == 2
    assert result["cursor"]["samples_seen"] == 4
    assert result["cursor"]["optimizer_steps"] == 4
    assert learner.cost_observations == 2
    assert Path(result["checkpoint"]).is_file()


def test_data_failure_retains_last_safe_checkpoint(setup, tmp_path, monkeypatch):
    learner, state = setup
    source = SimpleNamespace(
        config=learner.reference.config,
        _dataset=[
            ScaredA00Sample("dataset_1/keyframe_1/frame_data000000", "final_train", state.batch, {})
        ],
    )

    def fail(*_):
        raise RuntimeError("synthetic CUDA failure")

    monkeypatch.setattr("reliable_endo_gs.training.action_run.frozen_state", fail)
    with pytest.raises(RuntimeError, match="synthetic CUDA"):
        run_action_training(source, learner, {}, tmp_path / "failed")
    payload = torch.load(tmp_path / "failed/initial.pt", weights_only=True)
    assert payload["cursor"]["samples_seen"] == 0
    assert not (tmp_path / "failed/final.pt").exists()


def test_gt_supervision_rejected_before_inference(setup):
    learner, state = setup
    sample = ScaredA00Sample(
        "dataset_1/keyframe_1/frame_data000000",
        "final_train",
        state.batch,
        {},
        supervision=object(),
    )
    with pytest.raises(ValueError, match="must not load GT"):
        frozen_state(SimpleNamespace(model=learner.reference), sample)


def test_selector_small_and_feature_contract():
    selector = ActionSelector()
    assert sum(p.numel() for p in selector.parameters()) == 1730
    assert selector(torch.ones(3, 18)).shape == (3, 3)
    assert torch.equal(selector(torch.ones(3, 18))[:, 0], torch.zeros(3))
    with pytest.raises(ValueError):
        selector(torch.ones(3, 19))
