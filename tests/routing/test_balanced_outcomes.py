"""Balanced utility, proper risk events and immutable policy replay contracts."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.evaluation.outcome_selection import (
    component_constraint_targets,
    constraint_targets,
    evaluate_outcome_predictions,
)
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES
from reliable_endo_gs.routing.outcome_predictor import OutcomePredictor, load_outcome_predictor
from reliable_endo_gs.routing.tradeoff import (
    COMPONENT_NAMES,
    TradeoffConfig,
    constraint_margins_tensor,
    feasible_changes,
    load_tradeoff_config,
    outcome_utility,
    outcome_utility_tensor,
)
from reliable_endo_gs.training.outcome_calibration import (
    calibrate_joint_risk,
    split_training_calibration,
)
from reliable_endo_gs.training.outcome_learning import (
    fit_outcome_predictor,
    load_outcome_table,
    run_outcome_pilot,
)

from .test_metric_tradeoff import vector, write_table


def balanced_config(**kwargs):
    return replace(
        load_tradeoff_config(Path("configs/routing/metric_tradeoff_balanced.yaml")), **kwargs
    )


def grouped_table(path, role, cfg, *, offset=0):
    write_table(path, role, cfg)
    template = [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines()]
    rows = []
    for i in range(28):
        row = json.loads(json.dumps(template[i % len(template)]))
        row["sample_id"] = f"dataset_{i % 7 + 1}/keyframe_{i // 7 % 2 + 1}/frame_{i + offset:06d}"
        for o in row["outcomes"]:
            if o["action"] == "gaussian" and o["region_id"] == 0:
                o["metric_changes"][8] = -0.01
        rows.append(row)
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    meta = json.loads((path / "result.json").read_text())
    ids = [r["sample_id"] for r in rows]
    meta.update(
        samples=len(rows),
        sample_ids=ids,
        sample_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        outcome_sha256=hashlib.sha256(data).hexdigest(),
    )
    (path / "samples.jsonl").write_bytes(data)
    (path / "result.json").write_text(json.dumps(meta))
    return path


def test_historical_hash_and_balanced_groups_do_not_duplicate_spatial_rewards():
    old = load_tradeoff_config(Path("configs/routing/metric_tradeoff.yaml"))
    assert old.config_hash == "3b192945e065b38bf4f5c8e297078e065a090cf743bce5f80ab018750fa1da48"
    cfg = balanced_config()
    assert sum(cfg.score_weights[:2]) == pytest.approx(1)
    assert sum(cfg.score_weights[5:9]) == pytest.approx(1)
    assert cfg.score_weights[9:11] == (0, 0)
    with pytest.raises(ValueError, match="guards"):
        replace(cfg, collateral_weight=1)


def test_utility_is_signed_bounded_monotone_and_identical_for_training_and_selection():
    cfg = balanced_config()
    values = torch.tensor(
        [vector(global_depth_mae_gain_mm=1000, left_psnr_gain_db=1000)], requires_grad=True
    )
    score = outcome_utility_tensor(values, [1.0] * 12, cfg)
    assert 0 < float(score[0]) < 2
    assert outcome_utility(values[0].detach().tolist(), [1.0] * 12, cfg) == pytest.approx(
        float(score[0])
    )
    score.sum().backward()
    assert values.grad[0, 0] > 0 and values.grad[0, 5] > 0
    assert outcome_utility(vector(outside_rgb_gain=1000), [1.0] * 12, cfg) == 0
    assert not feasible_changes(vector(global_depth_mae_gain_mm=1000, left_psnr_gain_db=-0.02), cfg)
    assert feasible_changes(vector(left_psnr_gain_db=-0.005, global_depth_mae_gain_mm=0.1), cfg)


def test_margin_loss_uses_each_raw_guard_and_cannot_leak_risk_gradients(tmp_path):
    cfg = balanced_config(epochs=2, width=8)
    table = load_outcome_table(
        grouped_table(tmp_path / "train", "train", TradeoffConfig()), "train"
    )
    model, _ = fit_outcome_predictor(table, cfg)
    margins = constraint_margins_tensor(table.changes, model.metric_scale.tolist(), cfg)
    assert torch.equal(margins.lt(0), component_constraint_targets(table, cfg).bool())
    x = table.features.clone().requires_grad_(True)
    model.requires_grad_(True)
    _, _, component_logits = model.forward_details(x)
    gradients = torch.autograd.grad(
        component_logits.sum(), tuple(model.change_network.parameters()), allow_unused=True
    )
    assert all(g is None for g in gradients) and x.grad is None


def test_group_calibration_owns_only_training_frames_and_normalization(tmp_path):
    cfg = balanced_config(epochs=30, width=8)
    table = load_outcome_table(
        grouped_table(tmp_path / "train", "train", TradeoffConfig()), "train"
    )
    fit, cal, definition = split_training_calibration(table, cfg)
    fit_ids, cal_ids = set(definition["fit_sample_ids"]), set(definition["calibration_sample_ids"])
    assert fit_ids.isdisjoint(cal_ids) and fit_ids | cal_ids == set(table.metadata["sample_ids"])
    assert set(s.rsplit("/", 1)[0] for s in fit_ids).isdisjoint(
        s.rsplit("/", 1)[0] for s in cal_ids
    )
    assert len(definition["calibration_keyframes"]) == 7
    assert not definition["development_used_for_fitting"]
    model, logs = fit_outcome_predictor(fit, cfg)
    assert logs[-1]["risk"] < logs[0]["risk"]
    assert logs[-1]["component_risk"] < logs[0]["component_risk"]
    assert torch.equal(model.risk_positive_weight, torch.ones(3))
    before = model.feature_center.clone(), model.feature_scale.clone(), model.metric_scale.clone()
    report = calibrate_joint_risk(model, cal, cfg)
    assert all(x["supported"] for x in report["actions"].values())
    assert all(x["calibrated_log_loss"] <= x["raw_log_loss"] for x in report["actions"].values())
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            before, (model.feature_center, model.feature_scale, model.metric_scale), strict=True
        )
    )
    components = component_constraint_targets(cal, cfg)
    assert torch.equal(components.bool().any(-1), constraint_targets(cal, cfg).bool())
    with pytest.raises(ValueError, match="training"):
        split_training_calibration(replace(table, metadata={"collection_role": "development"}), cfg)
    with pytest.raises(ValueError, match="training"):
        calibrate_joint_risk(model, replace(cal, metadata={"collection_role": "development"}), cfg)


def test_unsupported_risk_source_abstains_and_invariant_components_stay_zero():
    model = OutcomePredictor(8, component_risk=True)
    x = torch.zeros(2, len(BALANCED_FEATURE_NAMES))
    y = torch.zeros(2, 3, len(COMPONENT_NAMES))
    model.fit_normalization(x, y, torch.ones(2, 3, dtype=torch.bool))
    changes, risk = model.predict(x)
    assert torch.equal(risk, torch.ones_like(risk))
    assert torch.equal(changes[:, 2, [0, 1, 2, 3, 4, 11]], torch.zeros(2, 6))
    assert torch.equal(
        model.predict_component_risks(x)[:, 2, [0, 1, 2, 3, 4, 11]], torch.zeros(2, 6)
    )
    model.risk_calibrated.fill_(True)
    model.risk_bias.fill_(-2)
    _, adjusted = model.predict(x)
    torch.testing.assert_close(adjusted, torch.full((2, 3), torch.sigmoid(torch.tensor(-2.0))))


def test_explicit_relabel_preserves_tables_and_binds_new_calibration_artifact(tmp_path):
    cfg = balanced_config(epochs=30, width=8, bootstrap_repetitions=10)
    train = grouped_table(tmp_path / "train", "train", TradeoffConfig())
    dev = grouped_table(tmp_path / "dev", "development", TradeoffConfig(), offset=100)
    config = tmp_path / "balanced.json"
    config.write_text(json.dumps(asdict(cfg)))
    originals = {p: p.read_bytes() for d in (train, dev) for p in d.iterdir()}
    with pytest.raises(ValueError, match="contract"):
        run_outcome_pilot(
            training_table=train,
            development_table=dev,
            config_path=config,
            output_dir=tmp_path / "rejected",
        )
    result = run_outcome_pilot(
        training_table=train,
        development_table=dev,
        config_path=config,
        output_dir=tmp_path / "new",
        relabel_policy=True,
        budgets_ms=(1000.0,),
    )
    assert result["policy_relabel_explicit"]
    assert result["collection_config_hash"] != result["config_hash"]
    assert result["fit_frames"] == 14
    assert originals == {p: p.read_bytes() for p in originals}
    kwargs = dict(
        expected_config=cfg,
        expected_action_identity={"checkpoint_sha256": "synthetic"},
        expected_candidate_hash="candidate",
    )
    model = load_outcome_predictor(tmp_path / "new/mlp.pt", **kwargs)
    assert model.component_risk
    raw = torch.load(tmp_path / "new/mlp.pt", weights_only=True)
    raw["state_dict"]["risk_temperature"][0] = 99
    torch.save(raw, tmp_path / "tampered.pt")
    with pytest.raises(ValueError, match="parameters"):
        load_outcome_predictor(tmp_path / "tampered.pt", **kwargs)
    raw["training_partition"]["development_used_for_fitting"] = True
    torch.save(raw, tmp_path / "contaminated.pt")
    with pytest.raises(ValueError, match="partition"):
        load_outcome_predictor(tmp_path / "contaminated.pt", **kwargs)


def test_missing_local_gt_does_not_remove_available_render_metrics(tmp_path):
    cfg = TradeoffConfig()
    table = load_outcome_table(write_table(tmp_path / "dev", "development", cfg), "development")
    names = ("left_psnr", "right_psnr", "left_ssim", "right_ssim")
    for record in table.records:
        record["base_metrics"] = {name: {"value": 1.0} for name in names}
        for outcome in record["outcomes"]:
            outcome["metrics"] = {name: {"value": 1.2} for name in names}
    mask = table.observed.clone()
    mask[0, 2] = False
    table = replace(table, observed=mask)
    prediction = torch.zeros_like(table.changes)
    prediction[:, 2, 5:9] = 1
    summary, rows = evaluate_outcome_predictions(
        table,
        prediction,
        torch.zeros_like(table.observed, dtype=torch.float32),
        scales=[1.0] * 12,
        estimated_costs=torch.ones(3) * 10,
        config=cfg,
        budgets_ms=(1000.0,),
        predictor_ms=(0.0,) * len(table.records),
        policy="constant",
    )
    assert not rows[0]["metric_observation_available"]
    assert rows[0]["actual_constraints_satisfied"] is None
    metric = summary["1000.0"]["render_changes_independent_of_gt"]["left_psnr"]
    assert metric["samples_available"] == 8 and metric["mean_gain"] == pytest.approx(0.2)
    assert summary["1000.0"]["metric_samples_available"] == 7
