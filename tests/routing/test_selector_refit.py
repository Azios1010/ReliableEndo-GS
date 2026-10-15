"""Stationary label identity, independent TRAIN roles, calibrated deployment."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from reliable_endo_gs.actions.repair import Action
from reliable_endo_gs.config.intervention import load_intervention_config
from reliable_endo_gs.config.selector_refit import SelectorRefitConfig, partition_selector_training
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.calibrated_selector import (
    CalibratedSelector,
    load_calibrated_selector,
)
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES
from reliable_endo_gs.training.selector_fitting import (
    calibrate_selector,
    fit_gain_affine,
    fit_selector_heads,
    keyframe_weights,
)
from reliable_endo_gs.training.selector_outcomes import (
    SelectorOutcomes,
    collect_selector_outcomes,
    load_selector_outcomes,
)
from tests.routing.test_balanced import setup as balanced_fixture
from tests.routing.test_metric_joint import metric_learner
from tests.routing.test_selector_optimization import _on_device

setup = balanced_fixture


def policy():
    return load_intervention_config(Path("configs/routing/intervention_metric_gain_fit.yaml"))


def table(keys, *, offset=0.0, harms=None):
    ids = tuple(f"dataset_1/keyframe_{k}/frame_data{i:06}" for k in keys for i in range(4))
    x = torch.zeros(len(ids), len(BALANCED_FEATURE_NAMES))
    x[:, 0] = torch.arange(len(ids)).float() / len(ids) + offset
    y = torch.tensor([[0.0, 0.2, 0.1]] * len(ids))
    h = (
        torch.tensor([[0.0, float(i % 2), float((i // 2) % 2)] for i in range(len(ids))])
        if harms is None
        else harms
    )
    records = [{"sample_id": s} for s in ids]
    return SelectorOutcomes({"collection_role": "train"}, records, x, y, y * 0.5, h, ids)


def fitted_head():
    cfg = replace(policy(), normalization_frames=1, normalization_rows=32)
    head = BalancedSelector(8, cfg)
    data = table((1, 2))
    head.normalization.observe(
        data.features,
        data.global_gain,
        data.local_gain,
        sample_id=data.sample_ids[0],
        role="final_train",
    )
    return head


def test_partition_is_whole_keyframe_deterministic_and_preserves_original_roles():
    ids = tuple(
        f"dataset_{d}/keyframe_{k}/frame_data{i:06}"
        for d in range(1, 8)
        for k in (1, 3)
        for i in range(4)
    )
    a = partition_selector_training(ids, SelectorRefitConfig())
    b = partition_selector_training(tuple(reversed(ids)), SelectorRefitConfig())
    assert [len(a[n + "_sample_ids"]) for n in ("fit", "validation", "calibration")] == [32, 8, 16]
    assert a["parent_roles_changed"] is False
    groups = []
    for n in ("fit", "validation", "calibration"):
        assert set(a[n + "_keyframes"]) == set(b[n + "_keyframes"])
        groups.append(set(a[n + "_keyframes"]))
    assert not any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    for bad in (ids + (ids[0],), ("dataset_8/keyframe_1/frame_data000000",), ids[:8]):
        with pytest.raises(ValueError):
            partition_selector_training(bad, SelectorRefitConfig())


def test_row_weights_preserve_default_and_uniform_objective():
    head = fitted_head()
    data = table((1, 2))
    old, _ = head.objective(data.features, data.global_gain, data.local_gain, data.harms)
    weighted, _ = head.objective(
        data.features, data.global_gain, data.local_gain, data.harms, row_weights=torch.ones(8)
    )
    for name in old:
        torch.testing.assert_close(old[name], weighted[name])
    with pytest.raises(ValueError, match="row weights"):
        head.objective(
            data.features, data.global_gain, data.local_gain, data.harms, row_weights=torch.zeros(8)
        )
    unequal = table((1, 1, 1, 2))
    w = keyframe_weights(unequal, torch.device("cpu"))
    assert float(w[:12].sum()) == pytest.approx(0.5)
    assert float(w[12:].sum()) == pytest.approx(0.5)


def test_gain_affine_preserves_sign_scale_and_identity_fallback():
    x = torch.linspace(-1, 1, 20)
    a, b = fit_gain_affine(x, 0.5 * x + 0.2, torch.ones_like(x), torch.tensor(1.0), 0.05)
    assert 0 <= a <= 2 and abs(b) <= 1
    assert ((a * x + b - (0.5 * x + 0.2)) ** 2).mean() < ((x - (0.5 * x + 0.2)) ** 2).mean()
    assert fit_gain_affine(x, x, torch.ones_like(x), torch.tensor(1.0), 0.05) == (1.0, 0.0)
    with pytest.raises(ValueError):
        fit_gain_affine(x, x, torch.zeros_like(x), torch.tensor(1.0), 0.05)


def test_unsupported_calibration_abstains_and_overlap_is_rejected():
    model = CalibratedSelector(fitted_head(), policy(), {}, "mask")
    # Strong positive utility and low risk cannot bypass missing support.
    output = model.head.gain_network[-1]
    risk = model.head.harm_network[-1]
    assert isinstance(output, nn.Linear) and isinstance(risk, nn.Linear)
    with torch.no_grad():
        output.bias.fill_(1)
        risk.bias.fill_(-10)
    x = table((5, 6)).features
    decision = model.select(
        x,
        torch.tensor([0.0, 1.0, 1.0]).expand(len(x), -1),
        torch.ones(len(x), dtype=torch.bool),
        1000,
    )
    assert decision.action == Action.STOP
    calibration = table((5, 6), harms=torch.zeros(8, 3))
    report = calibrate_selector(
        model, calibration, SelectorRefitConfig(), excluded_keyframes={"dataset_1/keyframe_1"}
    )
    assert all(not r["supported"] for r in report["actions"].values())
    assert not bool(model.calibration_available.any())
    with pytest.raises(ValueError, match="disjoint"):
        calibrate_selector(
            model, calibration, SelectorRefitConfig(), excluded_keyframes={"dataset_1/keyframe_5"}
        )


def test_separate_validation_stopping_normalization_and_artifact_binding(tmp_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fit, validation, calibration = table((1, 2)), table((3, 4), offset=100), table((5, 6))
    cfg = policy()
    refit = SelectorRefitConfig(max_steps=12, min_steps=2, validate_every=2, patience=2)
    head, report = fit_selector_heads(fit, validation, cfg, refit, 8, device)
    assert report["normalization_fit_sample_ids"] == list(fit.sample_ids)
    assert float(head.normalization.center[0]) < 1
    assert set(report["selected_steps"]) == {"gain", "risk"}
    with pytest.raises(ValueError, match="overlap"):
        fit_selector_heads(fit, fit, cfg, refit, 8, device)
    wrapped = CalibratedSelector(head, cfg, {"model": "fixed"}, "mask").to(device)
    cal = calibrate_selector(
        wrapped,
        calibration,
        refit,
        excluded_keyframes={"dataset_1/keyframe_" + str(k) for k in (1, 2, 3, 4)},
    )
    assert bool(wrapped.calibration_available.all())
    assert cal["cutoffs_tuned"] is False and cal["development_used_for_calibration"] is False
    partition = {
        "parent_roles_changed": False,
        **{
            name + "_keyframes": ["dataset_1/keyframe_" + str(k) for k in keys]
            for name, keys in (("fit", (1, 2)), ("validation", (3, 4)), ("calibration", (5, 6)))
        },
    }
    raw = {
        "schema": "calibrated_selector.v1",
        "width": 8,
        "action_identity": {"model": "fixed"},
        "decision_config": cfg.to_dict(),
        "head_config": head.config.to_dict(),
        "target_mask_index_sha256": "mask",
        "source_sha256": "source",
        "development_used_for_fitting": False,
        "state": {k: v.cpu() for k, v in wrapped.state_dict().items()},
        "calibration": cal,
        "partition": partition,
    }
    path = tmp_path / "selector.pt"
    torch.save(raw, path)
    args = dict(
        expected_action_identity={"model": "fixed"},
        expected_config=cfg,
        expected_mask_identity="mask",
        expected_source_sha256="source",
    )
    loaded = load_calibrated_selector(path, **args).to(device)
    for a, b in zip(
        wrapped.predict(calibration.features.to(device)),
        loaded.predict(calibration.features.to(device)),
        strict=True,
    ):
        torch.testing.assert_close(a, b)
    for key, bad in (
        ("expected_action_identity", {}),
        ("expected_mask_identity", "other"),
        ("expected_source_sha256", "other"),
    ):
        with pytest.raises(ValueError, match="binding"):
            load_calibrated_selector(path, **{**args, key: bad})
    raw["calibration"]["actions"]["STEREO"]["supported"] = False
    torch.save(raw, path)
    with pytest.raises(ValueError, match="availability"):
        load_calibrated_selector(path, **args)


def test_frozen_collection_is_gt_separate_and_hashes_actual_canonical_labels(setup, tmp_path):
    learner, sample, _, truth = metric_learner(setup)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learner.model.to(device)
    learner.actions.to(device)
    learner.selector.to(device)
    sample, truth = (_on_device(v, device) for v in (sample, truth))
    states = [
        copy.deepcopy(m.state_dict()) for m in (learner.model, learner.actions, learner.selector)
    ]
    flags = [
        p.requires_grad
        for m in (learner.model, learner.actions, learner.selector)
        for p in m.parameters()
    ]
    observed = []
    original = learner.selection_features

    def features(*args):
        observed.append("features")
        return original(*args)

    learner.selection_features = features

    def gt(sid):
        assert observed == ["features"] and sample.supervision is None
        return truth

    provenance = {
        k: "identity"
        for k in (
            "checkpoint_sha256",
            "checkpoint_source_sha256",
            "source_sha256",
            "training_manifest_hash",
            "development_manifest_hash",
            "role_hash",
            "target_mask_index_sha256",
        )
    }
    out = tmp_path / "outcomes"
    collect_selector_outcomes(
        learner, [sample], gt, collection_role="train", provenance=provenance, output_dir=out
    )
    data = load_selector_outcomes(out, "train")
    assert data.sample_ids and data.features.shape[1] == len(BALANCED_FEATURE_NAMES)
    for m, state in zip((learner.model, learner.actions, learner.selector), states, strict=True):
        assert all(torch.equal(v, state[k]) for k, v in m.state_dict().items())
    assert flags == [
        p.requires_grad
        for m in (learner.model, learner.actions, learner.selector)
        for p in m.parameters()
    ]
    with pytest.raises(ValueError, match="identity"):
        load_selector_outcomes(out, "development")
    rows = json.loads((out / "samples.jsonl").read_text())
    unknown = copy.deepcopy(rows)
    first = unknown["interventions"][0]["actions"][1]
    first.update(
        selector_label_available=False,
        global_gain=None,
        local_gain=None,
        harm=None,
        metric_changes=None,
    )
    content = json.dumps(unknown) + "\n"
    (out / "samples.jsonl").write_bytes(content.encode("utf-8"))
    meta = json.loads((out / "result.json").read_text())
    meta["samples_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    (out / "result.json").write_text(json.dumps(meta))
    reduced = load_selector_outcomes(out, "train")
    assert len(reduced.sample_ids) == len(data.sample_ids) - 1
    assert reduced.records[0]["interventions"][0]["actions"][1]["harm"] is None
    summary = reduced.observation_summary()
    assert summary["unknown_regions"] == 1
    assert summary["known_training_rows"] == len(reduced.sample_ids)
    assert summary["known_regions"] == len(reduced.sample_ids)
    assert summary["unknown_is_safe"] is False
    assert rows["interventions"][0]["actions"][1]["metric_diagnostics"] is not None
    rows["interventions"][0]["actions"][1]["global_gain"] += 1
    content = json.dumps(rows) + "\n"
    (out / "samples.jsonl").write_bytes(content.encode("utf-8"))
    meta = json.loads((out / "result.json").read_text())
    meta["samples_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    (out / "result.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="canonical"):
        load_selector_outcomes(out, "train")
