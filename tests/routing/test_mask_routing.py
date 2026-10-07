"""CPU mask proposals, real action writes, shared matching and joint learning."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from reliable_endo_gs.actions.repair import Action, primitive_core
from reliable_endo_gs.actions.stereo_matching import MatchingStereoRepair
from reliable_endo_gs.cli.student import build_parser
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.data.target_masks import TargetMaskIndex
from reliable_endo_gs.gaussians import ReferenceAttributeHead
from reliable_endo_gs.reconstruction import ReferenceConfig
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.regions.masks import MaskRegion, TargetROI
from reliable_endo_gs.routing.config import RoutingConfig
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES
from reliable_endo_gs.routing.masks import (
    MaskProposalConfig,
    candidate_scores,
    load_mask_config,
    mask_features,
    propose_masks,
)
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    read_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_config import StudentLearningConfig
from reliable_endo_gs.training.student_learning import StudentLearner
from tests.architecture.test_a04_reference_reconstruction import _batch
from tests.routing.test_frozen_a04 import AnalyticRenderer


def index_file(
    tmp_path, mask, *, source="input_roi", sample_id="dataset_1/keyframe_1/frame_data000000"
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "target.npy"
    np.save(path, mask.numpy())
    entry = {
        "path": "target.npy",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "output_hw": list(mask.shape),
        "source": source,
        "source_identity": "synthetic_roi.v1",
    }
    manifest = tmp_path / "masks.json"
    manifest.write_text(
        json.dumps({"schema": "deployment_target_masks.v1", "samples": {sample_id: entry}})
    )
    return manifest


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(3)
    model = StudentReconstructionModel(
        renderer=AnalyticRenderer(),
        config=replace(ReferenceConfig(), head_hidden_dims=(8, 8)),
        width=8,
        bins=4,
    )
    learner = StudentLearner(
        model,
        StudentLearningConfig(width=8, bins=4, kd_weight=0, student_epochs=0, joint_epochs=1),
        RoutingConfig(width=8, tile_hw=(3, 3)),
        intervention_config=InterventionConfig(
            schema="intervention_selection.v2",
            normalization_frames=1,
            normalization_rows=32,
            matching_width=4,
            search_radius_px=4,
            search_step_px=2,
            gradient_diagnostics_every=1,
        ),
        mask_config=MaskProposalConfig(
            cell_size=2, max_pixels=12, max_primitives=12, threshold=0.5
        ),
        target_masks=TargetMaskIndex(index_file(tmp_path, torch.ones(5, 5, dtype=torch.bool))),
    )
    sample = ScaredA00Sample(
        "dataset_1/keyframe_1/frame_data000000", "final_train", _batch(dtype=torch.float32), {}
    )
    return learner, sample


def test_sparse_region_crosses_grid_and_preserves_holes():
    pixels = (12, 13, 17, 19)  # 5x5: spans both old x/y quadrant boundaries.
    region = MaskRegion.from_indices(0, (5, 5), pixels, 1)
    assert region.pixel_count == 4
    assert region.core == (2, 2, 4, 5)
    mask = region.mask(torch.zeros(1, 3, 5, 5))
    assert tuple(mask.reshape(-1).nonzero().flatten().tolist()) == pixels
    assert not mask[0, 0, 3, 3]  # hole in the bounding box
    with pytest.raises(ValueError, match="grid mismatch"):
        region.mask(torch.zeros(1, 3, 6, 5))
    with pytest.raises(ValueError, match="unique sorted"):
        replace(region, flat_indices=(12, 12))


def test_target_index_hash_grid_origin_and_missing_fail_closed(tmp_path):
    path = index_file(tmp_path, torch.eye(5, dtype=torch.bool))
    index = TargetMaskIndex(path)
    roi = index.load("dataset_1/keyframe_1/frame_data000000", (5, 5), torch.device("cpu"))
    assert roi.pixels.sum() == 5 and "mask_sha256" in roi.source_identity
    with pytest.raises(ValueError, match="no implicit resize"):
        index.load("dataset_1/keyframe_1/frame_data000000", (4, 5), torch.device("cpu"))
    with pytest.raises(ValueError, match="missing deployment"):
        index.load("dataset_2/keyframe_1/frame_data000000", (5, 5), torch.device("cpu"))
    (tmp_path / "target.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        index.load("dataset_1/keyframe_1/frame_data000000", (5, 5), torch.device("cpu"))
    with pytest.raises(ValueError, match="invalid deployment"):
        TargetMaskIndex(index_file(tmp_path, torch.ones(5, 5, dtype=torch.bool), source="gt_valid"))


def test_matching_evidence_is_same_pass_and_detached(setup, monkeypatch):
    learner, sample = setup
    seen = []
    matching = learner.model.stereo_student._matching

    def count(batch):
        seen.append(batch)
        return matching(batch)

    monkeypatch.setattr(learner.model.stereo_student, "_matching", count)
    result = learner.model(sample.batch, include_matching_evidence=True)
    assert len(seen) == 1
    assert all(not value.requires_grad for value in result.matching_evidence.values())
    assert result.target.disparity.requires_grad
    ordinary = learner.model(sample.batch)
    torch.testing.assert_close(ordinary.target.disparity, result.target.disparity, rtol=0, atol=0)
    result.render_left.image.mean().backward()
    assert any(
        p.grad is not None and bool(p.grad.abs().sum())
        for p in learner.model.stereo_student.parameters()
    )


def test_proposal_target_only_cross_boundary_budgets_and_ablations(setup, monkeypatch):
    learner, sample = setup
    state = learner.state(sample)
    score = torch.zeros_like(state.batch.support, dtype=torch.float32)
    score[..., 1:4, 1:4] = 1
    score[..., 2, 2] = 0  # good target pixel must remain excluded
    monkeypatch.setattr("reliable_endo_gs.routing.masks.candidate_scores", lambda *args: score)
    roi = TargetROI(torch.ones_like(state.batch.support), "input_roi", "toy")
    regions = propose_masks(state, roi, learner.mask_config, halo=1)
    assert len(regions) == 1 and regions[0].pixel_count == 8
    assert not regions[0].mask(state.batch.left)[..., 2, 2]
    restricted = roi.pixels.clone()
    restricted[..., 1, 1] = False
    regions = propose_masks(
        state, TargetROI(restricted, "input_roi", "toy"), learner.mask_config, halo=1
    )
    assert regions[0].pixel_count == 7
    bounded = propose_masks(
        state, roi, replace(learner.mask_config, max_pixels=3, max_primitives=2), halo=1
    )
    assert sum(r.pixel_count for r in bounded) <= 3
    assert sum(int(primitive_core(state, r).sum()) for r in bounded) <= 2
    assert not propose_masks(state, roi, replace(learner.mask_config, max_read_pixels=1), halo=1)
    assert not propose_masks(
        state,
        TargetROI(torch.zeros_like(roi.pixels), "input_roi", "empty"),
        learner.mask_config,
        halo=1,
    )
    monkeypatch.undo()
    stereo = candidate_scores(state, replace(learner.mask_config, evidence="stereo"))
    gaussian = candidate_scores(state, replace(learner.mask_config, evidence="gaussian"))
    combined = candidate_scores(state, learner.mask_config)
    torch.testing.assert_close(combined, torch.maximum(stereo, gaussian))
    assert torch.isfinite(combined).all() and not combined.requires_grad


@pytest.mark.parametrize("halo", [(7, 10, 11, 14), (0, 0, 4, 5), (18, 24, 21, 29)])
def test_matching_roi_features_match_full_features_and_gradients(halo):
    torch.manual_seed(8)
    model = MatchingStereoRepair(4, 4, 2, 0.1, 8)
    with torch.no_grad():
        model.decoder[-1].weight.normal_(0, 0.1)
    left, right = torch.rand(1, 3, 21, 29), torch.rand(1, 3, 21, 29)
    disparity = torch.full((1, 1, 21, 29), 3.25)
    inputs = (
        left,
        right,
        disparity,
        left - right,
        halo,
        torch.ones_like(disparity, dtype=torch.bool),
        torch.zeros(1, 1, 1, 1),
    )
    full = model.forward_pair(*inputs)
    full.sum().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    crop = model.forward_pair(*inputs, roi_features=True)
    torch.testing.assert_close(crop, full, atol=1e-6, rtol=1e-5)
    crop.sum().backward()
    for p, grad in zip(model.parameters(), grads, strict=True):
        torch.testing.assert_close(p.grad, grad, atol=1e-5, rtol=1e-4)


def test_selected_pointwise_head_forward_and_gradient_parity():
    torch.manual_seed(5)
    head = ReferenceAttributeHead(hidden_dims=(8, 8))
    with torch.no_grad():
        head.output.weight.normal_(0, 0.1)
    compact_head = copy.deepcopy(head)
    context = torch.randn(2, 16, 13)
    selected = torch.rand(2, 16) > 0.8
    full = head(context, selected)
    compact = compact_head.forward_selected(context, selected)
    full_sum, compact_sum = 0, 0
    for item in fields(full):
        a, b = getattr(full, item.name), getattr(compact, item.name)
        # Historical valid_mask zeros inputs, not the MLP's biased outputs.
        # Only selected outputs enter the field; compact output elsewhere is 0.
        torch.testing.assert_close(a[selected], b[selected], atol=1e-6, rtol=1e-5)
        assert not bool(b[~selected].any())
        full_sum += a[selected].sum()
        compact_sum += b[selected].sum()
    full_sum.backward()
    compact_sum.backward()
    for p, cp in zip(head.parameters(), compact_head.parameters(), strict=True):
        torch.testing.assert_close(p.grad, cp.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("action", [Action.STEREO, Action.GAUSSIAN])
def test_real_action_only_writes_selected_pixels_primitives(setup, action):
    learner, sample = setup
    state = learner.state(sample)
    region = MaskRegion.from_indices(0, (5, 5), (12, 13, 17), 1)
    before = {
        item.name: getattr(state.base.gaussians, item.name).clone()
        for item in fields(state.base.gaussians)
        if isinstance(getattr(state.base.gaussians, item.name), torch.Tensor)
    }
    with torch.no_grad():
        if action == Action.STEREO:
            learner.actions.stereo.decoder[-1].bias.fill_(0.1)
        else:
            learner.actions.gaussian.output.bias.fill_(0.1)
    sizes = []
    head = learner.model.head if action == Action.STEREO else learner.actions.gaussian
    hook = head.register_forward_pre_hook(lambda module, args: sizes.append(args[0].shape[1]))
    outcome = learner.actions.execute(state, region, action, learner.model)
    hook.remove()
    selected = primitive_core(state, region)
    assert sizes == [int(selected.sum())]
    assert torch.equal(
        outcome.disparity[~region.mask(state.batch.left)],
        state.base.target.disparity[~region.mask(state.batch.left)],
    )
    for item in fields(outcome.field):
        a, b = getattr(outcome.field, item.name), getattr(state.base.gaussians, item.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a[~selected], b[~selected]), item.name
            assert torch.equal(b, before[item.name]), "base state mutated"
    stop = learner.actions.execute(state, region, Action.STOP, learner.model)
    assert stop.field is state.base.gaussians and stop.left is state.base.render_left


def test_mask_joint_learning_same_actions_local_global_and_all_components(setup, monkeypatch):
    learner, sample = setup
    monkeypatch.setattr(
        "reliable_endo_gs.routing.masks.candidate_scores",
        lambda state, config: torch.ones_like(state.batch.left[:, :1]),
    )
    modules = (
        learner.model.stereo_student,
        learner.model.head,
        learner.actions.stereo,
        learner.actions.gaussian,
        learner.selector,
    )
    snapshots = [[p.detach().clone() for p in m.parameters()] for m in modules]
    loss = learner.step(sample, joint=True)
    assert loss["selector_regions"] > 0 and np.isfinite(loss["loss"])
    for module, before in zip(modules, snapshots, strict=True):
        assert any(
            not torch.equal(p, old) for p, old in zip(module.parameters(), before, strict=True)
        )
    for row in loss["interventions"]:
        assert row["write_pixel_count"] == len(row["write_flat_indices"]) <= 12
        assert row["region_protocol"] == "target_mask_proposals.v1"
        assert {r["action"] for r in row["actions"]} == {"STOP", "STEREO", "GAUSSIAN"}
        assert "outside_rgb" in row["actions"][0]["before"]


def test_quiet_target_skips_actions_without_fake_labels(setup, monkeypatch):
    learner, sample = setup
    monkeypatch.setattr(
        "reliable_endo_gs.routing.masks.candidate_scores",
        lambda state, config: torch.zeros_like(state.batch.left[:, :1]),
    )
    loss = learner.step(sample, joint=True)
    assert loss["mask_joint_skipped"] == "no_eligible_target_candidates"
    assert learner.cost_observations == 0
    assert "interventions" not in loss
    assert all(p.grad is None for p in learner.actions.parameters())
    result = learner.infer(sample, selected=True)
    assert result.diagnostics["selected_action"] == int(Action.STOP)
    assert result.diagnostics["target_mask_index_sha256"] == learner.target_masks.identity


def test_feature_ablations_keep_proposals_fixed_and_remove_source_columns(setup):
    learner, sample = setup
    state = learner.state(sample)
    regions = (MaskRegion.from_indices(0, (5, 5), (12, 13, 17), 1),)
    combined = learner.selection_features(state, regions)
    stereo = mask_features(
        state,
        regions,
        learner.intervention_config,
        replace(learner.mask_config, feature_evidence="stereo"),
        scale_unit_m=learner.model.config.context_scale_m,
    )
    gaussian = mask_features(
        state,
        regions,
        learner.intervention_config,
        replace(learner.mask_config, feature_evidence="gaussian"),
        scale_unit_m=learner.model.config.context_scale_m,
    )
    column = {name: i for i, name in enumerate(BALANCED_FEATURE_NAMES)}
    assert combined.shape == stereo.shape == gaussian.shape
    assert stereo[0, column["opacity"]] == 0
    assert gaussian[0, column["cost_entropy"]] == 0
    assert gaussian[0, column["matching_evidence_available"]] == 0
    assert combined[0, column["matching_evidence_available"]] == 1
    for feature in ("rgb_r", "core_x", "fx_over_width", "baseline_m"):
        assert (
            combined[0, column[feature]]
            == stereo[0, column[feature]]
            == gaussian[0, column[feature]]
        )


def test_masked_infer_exercises_selected_action_and_preserves_parameters(setup, monkeypatch):
    learner, sample = setup
    monkeypatch.setattr(
        "reliable_endo_gs.routing.masks.candidate_scores",
        lambda state, config: torch.ones_like(state.batch.left[:, :1]),
    )
    learner.step(sample, joint=True)
    learner.step(sample, joint=True, position=1)
    before = [
        p.detach().clone()
        for module in (learner.model, learner.actions, learner.selector)
        for p in module.parameters()
    ]

    def predict(features):
        gains = features.new_zeros((len(features), 3))
        gains[:, int(Action.GAUSSIAN)] = 1
        return gains, torch.zeros_like(gains)

    monkeypatch.setattr(learner.selector, "predict", predict)
    result = learner.infer(sample, selected=True)
    assert result.diagnostics["selected_action"] == int(Action.GAUSSIAN)
    assert 0 < result.diagnostics["selected_pixels"] <= learner.mask_config.max_pixels
    assert 0 < result.diagnostics["selected_primitives"] <= learner.mask_config.max_primitives
    assert result.diagnostics["region_protocol"] == "target_mask_proposals.v1"
    after = [
        p
        for module in (learner.model, learner.actions, learner.selector)
        for p in module.parameters()
    ]
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_checkpoint_mask_protocol_rejects_grid_resume(setup, tmp_path):
    learner, _ = setup
    identity = {
        "source_sha256": "toy",
        "manifest_hash": "toy",
        "role_hash": "toy",
        "renderer_identity": "toy",
    }
    cursor = {"epoch": 0, "next_sample_index": 0, "samples_seen": 0, "optimizer_steps": 0}
    path = tmp_path / "checkpoint.pt"
    save_student_checkpoint(path, learner, identity, cursor)
    assert (
        read_student_checkpoint(path)["schema"] == "student_joint_mask_intervention_checkpoint.v1"
    )
    assert load_student_checkpoint(path, learner, identity, 1) == cursor
    learner.mask_config = None
    learner.target_masks = None
    with pytest.raises(ValueError, match="protocol"):
        load_student_checkpoint(path, learner, identity, 1)


def test_cli_and_config_are_opt_in():
    cfg = load_mask_config(Path("configs/routing/target_masks.yaml"))
    assert cfg.max_pixels == 4096
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--data-root",
            "/tmp/data",
            "--manifest",
            "/tmp/manifest",
            "--output-dir",
            "/tmp/run",
            "--mask-config",
            "configs/routing/target_masks.yaml",
            "--target-mask-index",
            "/tmp/roi.json",
        ]
    )
    assert args.target_mask_index.name == "roi.json"
