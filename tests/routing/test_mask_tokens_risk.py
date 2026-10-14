"""Masked tokens, explicit physical risks and the shared guiding harness."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from reliable_endo_gs.config.intervention import load_intervention_config
from reliable_endo_gs.config.selector_refit import SelectorRefitConfig
from reliable_endo_gs.data.target_masks import TargetMaskIndex
from reliable_endo_gs.evaluation.risk_events import risk_event_metrics
from reliable_endo_gs.regions.masks import MaskRegion
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.calibrated_selector import CalibratedSelector
from reliable_endo_gs.routing.masks import MaskProposalConfig
from reliable_endo_gs.routing.tokens import mask_feature_tokens, selector_feature_names
from reliable_endo_gs.routing.tradeoff import COMPONENT_NAMES, grouped_harm
from reliable_endo_gs.training.metric_quality import metric_action_policy
from reliable_endo_gs.training.selector_fitting import fit_selector_heads
from reliable_endo_gs.training.selector_outcomes import (
    SelectorOutcomes,
    collect_selector_outcomes,
    load_selector_outcomes,
)
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_learning import StudentLearner
from tests.routing.test_balanced import setup as balanced_fixture
from tests.routing.test_mask_routing import index_file
from tests.routing.test_metric_joint import metric_learner
from tests.training.test_student_freeze_ablation import assert_state_equal

setup = balanced_fixture


def config(*, tokens=True, frozen=False):
    base = load_intervention_config(
        Path("configs/routing/intervention_metric_geometry_supervision.yaml")
    )
    return replace(
        base,
        selector_risk_mode="local_global",
        selector_feature_mode="mask_tokens" if tokens else "statistics",
        selector_token_channels=8 if tokens else 0,
        normalization_frames=1,
        normalization_rows=64,
        student_update_mode="frozen" if frozen else "joint",
        search_radius_px=4,
    )


def test_explicit_configs_historical_identity_and_feature_names():
    old = load_intervention_config(Path("configs/routing/intervention_metric_action_heads.yaml"))
    assert old.config_hash == "8fb6905c83f04b6bfbefef65a1b7ec4273cf0f5de6fe4dfa16ac2bf5fb530eec"
    assert (
        not {"selector_feature_mode", "selector_token_channels", "selector_risk_mode"}
        & old.to_dict().keys()
    )
    cfg = config()
    assert len(selector_feature_names(cfg)) == 61 + 3 * 8
    assert type(cfg)(**cfg.to_dict()) == cfg
    for kwargs in (
        {"selector_token_channels": 0},
        {"selector_feature_mode": "statistics"},
        {"selector_risk_mode": "unknown"},
        {"metric_quality": None},
    ):
        with pytest.raises(ValueError):
            replace(cfg, **kwargs)


def test_risk_groups_keep_every_metric_and_exact_or():
    cfg = config(tokens=False)
    for action in (1, 2):
        policy = metric_action_policy(cfg, action)
        assert grouped_harm([0.0] * 12, policy) == (False, False, False)
        for i, name in enumerate(COMPONENT_NAMES):
            changes = [0.0] * 12
            changes[i] = policy.minimum_changes[i] - policy.comparison_epsilon - 1e-5
            local, global_, any_ = grouped_harm(changes, policy)
            assert any_ and local != global_
            assert local == name.startswith("local_")
        with pytest.raises(ValueError):
            grouped_harm([float("nan")] * 12, policy)


def test_mask_holes_and_exact_native_feature_coordinates():
    image = torch.zeros(1, 3, 12, 12)
    native = torch.tensor([[0.0, 1.0, 2.0], [3.0, 1000.0, 5.0], [6.0, 7.0, 8.0]])[None, None]
    state = SimpleNamespace(
        batch=SimpleNamespace(left=image, support=torch.ones_like(image[:, :1], dtype=torch.bool)),
        support=torch.ones_like(image[:, :1], dtype=torch.bool),
        base=SimpleNamespace(matching_evidence={"student_features": native}),
    )
    region = MaskRegion.from_indices(0, (12, 12), (0, 8 * 12 + 8), 0)
    tokens = mask_feature_tokens(state, (region,), 1)
    torch.testing.assert_close(tokens[0, :2], torch.tensor([4.0, 8.0]))
    assert tokens[0, 2] == native.mean()
    assert not tokens.requires_grad
    with pytest.raises(ValueError):
        mask_feature_tokens(state, (MaskRegion.from_indices(0, (24, 24), (0, 8), 0),), 1)
    with pytest.raises(ValueError):
        mask_feature_tokens(state, (region,), False)
    bad = copy.copy(state)
    bad.base = SimpleNamespace(matching_evidence={"student_features": native.requires_grad_()})
    with pytest.raises(ValueError):
        mask_feature_tokens(bad, (region,), 1)


def test_grouped_risk_bounds_stop_and_binary_supervision():
    head = BalancedSelector(8, config(tokens=False))
    x = torch.randn(8, 61, requires_grad=True)
    events = head.risk_event_outputs(x)
    assert events.shape == (8, 3, 3) and bool((events[:, 0] == 0).all())
    p = events[:, 1:]
    assert bool((p[:, :, 2] >= p[:, :, :2].amax(-1)).all())
    assert bool((p[:, :, 2] <= p[:, :, :2].sum(-1).clamp(max=1)).all())
    _, union = head.predict(x)
    torch.testing.assert_close(union, events[:, :, 2])
    components = torch.zeros(8, 3, 2)
    components[:4, 1, 0] = 1
    components[4:, 2, 1] = 1
    harms = components.bool().any(-1).float()
    gain = torch.zeros(8, 3)
    losses, diag = head.objective(x, gain, gain, harms, risk_components=components)
    losses["harm"].backward()
    assert x.grad is None and all(p.grad is None for p in head.gain_network.parameters())
    assert all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in head.harm_network.parameters()
    )
    assert (
        "selector_stereo_risk_local_brier" in diag and "selector_gaussian_risk_global_loss" in diag
    )
    with pytest.raises(ValueError, match="exact OR"):
        head.objective(x, gain, gain, torch.zeros_like(harms), risk_components=components)
    with pytest.raises(ValueError, match="exact OR"):
        head.objective(x, gain, gain, harms)
    calibrated = CalibratedSelector(head, head.config, {}, "mask")
    assert calibrated.head.config.selector_risk_mode == "local_global"


def test_grouped_risk_learns_local_global_and_union_on_synthetic_evidence():
    torch.manual_seed(5)
    head = BalancedSelector(16, config(tokens=False))
    x = torch.zeros(32, 61)
    x[:, 0] = torch.arange(32).remainder(2).float() * 2 - 1
    x[:, 1] = torch.arange(32).div(2, rounding_mode="floor").remainder(2).float() * 2 - 1
    components = torch.zeros(32, 3, 2)
    components[:, 1, 0] = (x[:, 0] > 0).float()
    components[:, 1, 1] = (x[:, 1] > 0).float()
    components[:, 2] = 1 - components[:, 1]
    harms = components.bool().any(-1).float()
    zeros = torch.zeros(32, 3)
    optimizer = torch.optim.Adam(head.harm_network.parameters(), lr=0.01)
    initial = float(
        head.objective(x, zeros, zeros, harms, risk_components=components)[0]["harm"].detach()
    )
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = head.objective(x, zeros, zeros, harms, risk_components=components)[0]["harm"]
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < initial * 0.1
    target = torch.cat((components, harms[..., None]), -1)
    assert torch.equal(head.risk_event_outputs(x)[:, 1:] > 0.5, target[:, 1:].bool())


def test_event_metrics_exclude_unknown_and_separate_candidate_from_selected():
    cfg = config(tokens=False)
    local_failure = [0.0] * 12
    local_failure[10] = -0.02
    record = {
        "valid_regions": [True],
        "actual_label_available": [[True, True, False]],
        "predicted_risk_events": [[[0.0, 0.0, 0.0], [0.8, 0.1, 0.85], [0.1, 0.1, 0.15]]],
        "policies": {"selector": {"action": "STEREO", "region": 0}},
        "counterfactual_metric_outcomes": [
            {"action": "STEREO", "region_id": 0, "metric_changes": local_failure},
            {"action": "GAUSSIAN", "region_id": 0, "metric_changes": None},
        ],
    }
    result = risk_event_metrics([record], cfg)
    assert result["STEREO"]["selected"]["local"]["harm_rate"] == 1
    assert result["STEREO"]["selected"]["global"]["harm_rate"] == 0
    assert result["GAUSSIAN"]["unknown_candidates"] == 1
    assert result["GAUSSIAN"]["candidates"]["any"]["brier"] is None


def learner_inputs(setup, tmp_path, frozen):
    old, sample, teacher, truth = metric_learner(setup)
    h, w = sample.batch.left.shape[-2:]
    index = TargetMaskIndex(
        index_file(tmp_path, torch.ones(h, w, dtype=torch.bool), sample_id=sample.sample_id)
    )
    learner = StudentLearner(
        old.model,
        old.config,
        old.routing,
        intervention_config=config(frozen=frozen),
        mask_config=MaskProposalConfig(
            threshold=0, cell_size=2, max_regions=2, max_pixels=32, max_primitives=128
        ),
        target_masks=index,
    )
    return learner, sample, teacher, replace(truth, disparity=truth.disparity * 1.1)


@pytest.mark.parametrize("frozen", [False, True])
def test_shared_harness_updates_resume_and_unknown_exclusion(setup, tmp_path, frozen):
    learner, sample, teacher, truth = learner_inputs(setup, tmp_path, frozen)
    start = copy.deepcopy(learner.model.state_dict())
    for step in range(3):
        result = learner.step(
            sample, joint=True, teacher_target=teacher, metric_truth=truth, position=step
        )
    assert result["selector_updates_this_snapshot"] == 4
    assert len(result["features_raw"][0]) == 85
    assert "predicted_risk_events" in result and "selector_gaussian_risk_local_brier" in result
    for region in result["interventions"]:
        for action in region["actions"]:
            assert action["harm"] == (action["risk_local"] or action["risk_global"])
    if frozen:
        assert_state_equal(start, learner.model.state_dict())
    else:
        assert any(not torch.equal(v, learner.model.state_dict()[k]) for k, v in start.items())
    state = learner.state(sample)
    assert not state.base.matching_evidence["student_features"].requires_grad
    checkpoint = tmp_path / "token.pt"
    cursor = {"epoch": 0, "next_sample_index": 3, "optimizer_steps": 3, "samples_seen": 3}
    save_student_checkpoint(checkpoint, learner, {}, cursor)
    assert load_student_checkpoint(checkpoint, learner, {}, 10) == cursor
    historical = StudentLearner(
        copy.deepcopy(learner.model),
        learner.config,
        learner.routing,
        intervention_config=replace(
            learner.intervention_config,
            selector_risk_mode="union",
            selector_feature_mode="statistics",
            selector_token_channels=0,
        ),
        mask_config=learner.mask_config,
        target_masks=learner.target_masks,
    )
    before = copy.deepcopy(historical.model.state_dict())
    with pytest.raises(ValueError):
        load_student_checkpoint(checkpoint, historical, {}, 10)
    assert_state_equal(before, historical.model.state_dict())
    counts = dict(learner.optimizer_update_counts)
    result = learner.step(
        sample,
        joint=True,
        teacher_target=teacher,
        metric_truth=replace(truth, valid_mask=torch.zeros_like(truth.valid_mask)),
        position=3,
    )
    assert result["selector_updates_this_snapshot"] == 0
    assert learner.optimizer_update_counts["selector"] == counts["selector"]
    assert all(
        a["risk_local"] is None and a["risk_global"] is None
        for r in result["interventions"]
        for a in r["actions"][1:]
    )


def test_frozen_outcomes_keep_dynamic_schema_and_risk_labels(setup, tmp_path):
    learner, sample, _, truth = learner_inputs(setup, tmp_path / "roi", False)
    out = tmp_path / "outcomes"
    collect_selector_outcomes(
        learner,
        [sample],
        lambda _: truth,
        collection_role="train",
        provenance={
            key: "a" * 64
            for key in (
                "checkpoint_sha256",
                "checkpoint_source_sha256",
                "source_sha256",
                "training_manifest_hash",
                "development_manifest_hash",
                "role_hash",
                "target_mask_index_sha256",
            )
        },
        output_dir=out,
    )
    table = load_selector_outcomes(out, "train")
    assert table.features.shape[1] == 85
    assert table.risk_components.shape == (*table.harms.shape, 2)
    assert torch.equal(table.risk_components.bool().any(-1), table.harms.bool())
    meta = json.loads((out / "result.json").read_text())
    meta["feature_names"][-1] = "unverified_feature"
    (out / "result.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="identity"):
        load_selector_outcomes(out, "train")


def test_tokens_share_one_student_matching_pass(setup, tmp_path, monkeypatch):
    learner, sample, _, _ = learner_inputs(setup, tmp_path, False)
    calls = []
    original = learner.model.stereo_student._matching

    def matching(batch):
        calls.append(1)
        return original(batch)

    monkeypatch.setattr(learner.model.stereo_student, "_matching", matching)
    state = learner.state(sample)
    regions = learner.candidate_regions(state, sample.sample_id)
    features = learner.selection_features(state, regions)
    assert calls == [1] and features.shape[1] == 85
    assert sample.supervision is None and not features.requires_grad


def test_stationary_guiding_fit_accepts_token_schema_and_grouped_risk():
    torch.manual_seed(2)
    cfg = config()
    features = torch.randn(16, len(selector_feature_names(cfg)))
    gain = torch.zeros(16, 3)
    gain[:, 1:] = features[:, :2] * 0.1
    local = gain * 0.25
    components = torch.zeros(16, 3, 2)
    components[:, 1:] = (features[:, :4].reshape(16, 2, 2) > 0).float()
    harms = components.bool().any(-1).float()

    def table(fold):
        return SelectorOutcomes(
            {"collection_role": "train"},
            [],
            features,
            gain,
            local,
            harms,
            tuple(f"dataset_{fold}/keyframe_1/frame_data{i:06d}" for i in range(16)),
            components,
        )

    fitted, report = fit_selector_heads(
        table(1),
        table(2),
        cfg,
        SelectorRefitConfig(max_steps=4, min_steps=1, validate_every=1),
        8,
        torch.device("cpu"),
    )
    assert bool(fitted.normalization.fitted)
    assert fitted.risk_event_outputs(features).shape == (16, 3, 3)
    assert report
