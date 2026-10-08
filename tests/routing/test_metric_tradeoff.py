"""Metric units, safety, STOP, immutable tables and detached predictor learning."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest
import torch

from reliable_endo_gs.cli.selection import build_parser
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES
from reliable_endo_gs.routing.outcome_predictor import OutcomePredictor, load_outcome_predictor
from reliable_endo_gs.routing.tradeoff import (
    ACTION_NAMES,
    COMPONENT_NAMES,
    TradeoffConfig,
    feasible_changes,
    load_tradeoff_config,
    outcome_utility,
    select_tradeoff,
)
from reliable_endo_gs.training.outcome_learning import (
    evaluate_outcome_predictions,
    fit_outcome_predictor,
    load_outcome_table,
    run_outcome_pilot,
)


def vector(**changes):
    result = [0.0] * len(COMPONENT_NAMES)
    for name, value in changes.items():
        result[COMPONENT_NAMES.index(name)] = value
    return result


def test_geometry_improvement_cannot_override_render_constraint():
    cfg = TradeoffConfig()
    v = vector(global_depth_mae_gain_mm=100.0, left_psnr_gain_db=-0.01)
    result = select_tradeoff(
        [[v, vector(), vector(left_psnr_gain_db=0.1, right_psnr_gain_db=0.1)]],
        [[0.0, 0.0, 0.0]],
        [[50.0, 50.0, 20.0]],
        [[True] * 3],
        remaining_ms=100.0,
        scales=[1.0] * len(COMPONENT_NAMES),
        config=cfg,
    )
    assert result.action == "gaussian"
    assert not feasible_changes(v, cfg)
    assert feasible_changes(v, replace(cfg, maximum_psnr_drop_db=0.02))


@pytest.mark.parametrize("component", COMPONENT_NAMES)
def test_each_constraint_has_its_own_metric_units(component):
    assert not feasible_changes(vector(**{component: -0.1}), TradeoffConfig())
    assert feasible_changes(vector(**{component: 0.1}), TradeoffConfig())


def test_stop_missing_observation_nan_budget_and_deterministic_ties():
    cfg = TradeoffConfig()
    v = vector(left_psnr_gain_db=1.0, right_psnr_gain_db=1.0)
    kw = dict(remaining_ms=100.0, scales=[1.0] * len(COMPONENT_NAMES), config=cfg)
    values = [[v, v, v]]
    risks = [[0.0, 0.0, 0.0]]
    costs = [[20.0, 20.0, 20.0]]
    assert select_tradeoff(values, risks, costs, [[True] * 3], **kw).action == "correspondence"
    assert select_tradeoff(values, risks, costs, [[False] * 3], **kw).action == "STOP"
    assert select_tradeoff(values, [[0.2] * 3], costs, [[True] * 3], **kw).action == "STOP"
    assert select_tradeoff(values, risks, [[101.0] * 3], [[True] * 3], **kw).action == "STOP"
    assert select_tradeoff([[vector()] * 3], risks, costs, [[True] * 3], **kw).action == "STOP"
    bad = vector(left_psnr_gain_db=float("nan"))
    assert select_tradeoff([[bad] * 3], risks, costs, [[True] * 3], **kw).action == "STOP"
    with pytest.raises(ValueError):
        select_tradeoff(values, [[float("nan")] * 3], costs, [[True] * 3], **kw)
    # Changing statistical utility scales cannot silently change safety limits.
    assert feasible_changes(v, cfg)
    assert outcome_utility(v, [1.0] * len(COMPONENT_NAMES), cfg) > outcome_utility(
        v, [2.0] * len(COMPONENT_NAMES), cfg
    )


def write_table(directory, role, cfg, *, offset=0):
    directory.mkdir()
    records = []
    for frame in range(8):
        sid = f"dataset_{frame % 7 + 1}/keyframe_{frame % 2 + 1}/frame_{frame + offset:06d}"
        features = torch.zeros(2, len(BALANCED_FEATURE_NAMES))
        features[:, 0] = torch.tensor([-1.0, 1.0]) + frame / 100
        features[:, 1] = frame / 10
        outcomes = []
        for ri in range(2):
            for ai, action in enumerate(ACTION_NAMES):
                # Known safe geometry/render tradeoff; the other repair has real harm.
                depth = (
                    (0.2 if ri == 0 else -0.2)
                    if ai == 0
                    else (0.1 if ri == 1 else -0.1)
                    if ai == 1
                    else 0.0
                )
                changes = vector(
                    global_depth_mae_gain_mm=depth,
                    local_depth_mae_gain_mm=depth,
                    global_depth_rmse_gain_mm=depth,
                    local_depth_rmse_gain_mm=depth,
                    global_depth_p95_gain_mm=depth,
                    left_psnr_gain_db=0.1 + ai * 0.1,
                    right_psnr_gain_db=0.1 + ai * 0.1,
                    left_ssim_gain=0.01,
                    right_ssim_gain=0.01,
                )
                outcomes.append(
                    {
                        "region_id": ri,
                        "action": action,
                        "action_ms_samples": [10.0 + ai] * 3,
                        "action_ms_max": 10.0 + ai,
                        "metric_changes": changes,
                        "metric_observation_available": True,
                    }
                )
        records.append(
            {
                "sample_id": sid,
                "base_ms": 10.0,
                "base_ms_without_matching": 10.0,
                "feature_ms": 1.0,
                "features_raw": features.tolist(),
                "valid_regions": [True, True],
                "outcomes": outcomes,
            }
        )
    data = "".join(json.dumps(r) + "\n" for r in records).encode()
    ids = [r["sample_id"] for r in records]
    metadata = {
        "status": "metric_outcome_table_complete",
        "table_schema": "metric_outcome_table.v1",
        "samples": len(records),
        "sample_ids": ids,
        "sample_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "collection_role": role,
        "outcome_sha256": hashlib.sha256(data).hexdigest(),
        "feature_names": list(BALANCED_FEATURE_NAMES),
        "component_names": list(COMPONENT_NAMES),
        "action_names": list(ACTION_NAMES),
        "weights_unchanged": True,
        "teacher_inference_executed": False,
        "training_executed": False,
        "timing_repeats": 3,
        "source_sha256": "synthetic",
        "training_manifest_hash": "training",
        "development_manifest_hash": "development",
        "role_hash": "roles",
        "candidate_config_hash": "candidate",
        "renderer_identity": "synthetic",
        "gpu_uuid": "synthetic",
        "tradeoff_config_hash": cfg.config_hash,
        "tradeoff_config": asdict(cfg),
        "imported_weights": {"checkpoint_sha256": "synthetic"},
    }
    (directory / "samples.jsonl").write_bytes(data)
    (directory / "result.json").write_text(json.dumps(metadata))
    return directory


def test_predictor_train_only_normalization_gradient_separation_and_reload(tmp_path):
    cfg = replace(TradeoffConfig(), epochs=30, width=8, bootstrap_repetitions=10)
    train = load_outcome_table(write_table(tmp_path / "train", "train", cfg), "train")
    model, log = fit_outcome_predictor(train, cfg)
    assert log[-1]["components"] < log[0]["components"]
    assert log[-1]["risk"] < log[0]["risk"]
    inputs = train.features.clone().requires_grad_(True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    a, r = model.predict(inputs)
    assert not a.requires_grad and not r.requires_grad
    assert torch.equal(a[:, 2, :5], torch.zeros_like(a[:, 2, :5]))
    assert torch.equal(a[:, 2, -1], torch.zeros_like(a[:, 2, -1]))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    restored = OutcomePredictor(8)
    restored.load_state_dict(model.state_dict())
    aa, rr = restored.predict(inputs)
    torch.testing.assert_close(a, aa)
    torch.testing.assert_close(r, rr)
    with pytest.raises(ValueError):
        model.fit_normalization(train.features, train.changes, train.observed)
    with pytest.raises(ValueError):
        fit_outcome_predictor(replace(train, metadata={"collection_role": "development"}), cfg)
    model.requires_grad_(True)
    _, logits = model(inputs)
    risk_grad = torch.autograd.grad(
        logits.sum(), tuple(model.change_network.parameters()), allow_unused=True
    )
    assert all(g is None for g in risk_grad)
    assert inputs.grad is None


def test_table_checks_checksum_duplicates_correspondence_invariance_and_final_test(tmp_path):
    cfg = TradeoffConfig()
    p = write_table(tmp_path / "table", "train", cfg)
    with pytest.raises(ValueError):
        load_outcome_table(p, "development")
    original = (p / "samples.jsonl").read_text()
    metadata = json.loads((p / "result.json").read_text())
    (p / "samples.jsonl").write_text(original + "\n")
    with pytest.raises(ValueError):
        load_outcome_table(p, "train")
    rows = [json.loads(line) for line in original.splitlines()]
    rows[0]["outcomes"].append(rows[0]["outcomes"][0])
    content = "".join(json.dumps(r) + "\n" for r in rows)
    metadata["outcome_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    (p / "samples.jsonl").write_bytes(content.encode("utf-8"))
    (p / "result.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="duplicate"):
        load_outcome_table(p, "train")
    rows[0]["outcomes"].pop()
    rows[0]["outcomes"][2]["metric_changes"][0] = 1.0
    content = "".join(json.dumps(r) + "\n" for r in rows)
    metadata["outcome_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    (p / "samples.jsonl").write_bytes(content.encode("utf-8"))
    (p / "result.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="invariance"):
        load_outcome_table(p, "train")


def test_predicted_decision_does_not_use_development_metric_labels_or_actual_costs(tmp_path):
    cfg = TradeoffConfig()
    table = load_outcome_table(write_table(tmp_path / "dev", "development", cfg), "development")
    prediction = torch.ones_like(table.changes) * 0.1
    prediction[:, 2, :5] = 0
    prediction[:, 2, -1] = 0
    risk = torch.zeros_like(table.observed, dtype=torch.float32)
    kw = dict(
        scales=[1.0] * len(COMPONENT_NAMES),
        estimated_costs=torch.tensor([10.0, 10.0, 10.0]),
        config=cfg,
        budgets_ms=(100.0,),
        predictor_ms=(0.0,) * len(table.records),
        policy="constant",
    )
    _, rows = evaluate_outcome_predictions(table, prediction, risk, **kw)
    poisoned = replace(
        table, changes=-table.changes - 1, costs=table.costs + 10000, observed=~table.observed
    )
    _, other = evaluate_outcome_predictions(poisoned, prediction, risk, **kw)
    assert [(r["action"], r["region"]) for r in rows] == [(r["action"], r["region"]) for r in other]
    assert all(r["budget_overrun_ms"] > 0 for r in other)


def test_full_bounded_cpu_pilot_records_comparisons_and_preserves_inputs(tmp_path):
    cfg = replace(TradeoffConfig(), epochs=20, width=8, bootstrap_repetitions=10)
    train = write_table(tmp_path / "train", "train", cfg)
    dev = write_table(tmp_path / "dev", "development", cfg, offset=100)
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps(asdict(cfg)))
    paths = [p for d in (train, dev) for p in d.iterdir()]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = run_outcome_pilot(
        training_table=train,
        development_table=dev,
        config_path=config,
        output_dir=tmp_path / "output",
        budgets_ms=(100.0,),
        device_name="cpu",
    )
    assert result["status"] == "metric_outcome_pilot_complete"
    assert not result["Gate_II_certified"] and not result["upstream_training_executed"]
    assert not result["native_predictor_timing"] and result["shared_keyframes"] > 0
    assert set(result["calibration"]) == {"constant", "linear", "mlp"}
    assert "fixed_gaussian" in result["policies"] and "mlp_gaussian_only" in result["policies"]
    assert (tmp_path / "output/mlp.pt").exists()
    imported = load_outcome_predictor(
        tmp_path / "output/mlp.pt",
        expected_config=cfg,
        expected_action_identity={"checkpoint_sha256": "synthetic"},
        expected_candidate_hash="candidate",
    )
    predicted, risks = imported.predict(load_outcome_table(dev, "development").features)
    assert predicted.shape == (16, 3, len(COMPONENT_NAMES)) and risks.shape == (16, 3)
    with pytest.raises(ValueError, match="identity"):
        load_outcome_predictor(
            tmp_path / "output/mlp.pt",
            expected_config=cfg,
            expected_action_identity={"checkpoint_sha256": "different-action-weights"},
            expected_candidate_hash="candidate",
        )
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with pytest.raises(FileExistsError):
        run_outcome_pilot(
            training_table=train,
            development_table=dev,
            config_path=config,
            output_dir=tmp_path / "output",
        )
    with pytest.raises(ValueError, match="overlap"):
        # Explicitly rewrite only this synthetic test fixture with overlapping IDs.
        data = (train / "samples.jsonl").read_bytes()
        meta = json.loads((dev / "result.json").read_text())
        train_meta = json.loads((train / "result.json").read_text())
        for k in ("sample_ids", "sample_ids_sha256", "outcome_sha256"):
            meta[k] = train_meta[k]
        (dev / "samples.jsonl").write_bytes(data)
        (dev / "result.json").write_text(json.dumps(meta))
        run_outcome_pilot(
            training_table=train,
            development_table=dev,
            config_path=config,
            output_dir=tmp_path / "bad",
        )


def test_config_and_cli_are_explicit_and_fail_unknown_policy(tmp_path):
    config = tmp_path / "cfg.yaml"
    config.write_text("schema: metric_action_tradeoff.v1\nmade_up: 1\n")
    with pytest.raises(ValueError):
        load_tradeoff_config(config)
    with pytest.raises(ValueError):
        TradeoffConfig(geometry_weight=0)
    with pytest.raises(ValueError):
        TradeoffConfig(psnr_weight=0, ssim_weight=0)
    args = build_parser().parse_args(
        [
            "outcome-pilot",
            "--training-table",
            "train",
            "--development-table",
            "dev",
            "--config",
            "config",
            "--output-dir",
            "out",
            "--device",
            "cpu",
        ]
    )
    assert args.command == "outcome-pilot" and args.device == "cpu"


def test_class_weight_odds_correction_and_immutable_risk_state():
    model = OutcomePredictor(8)
    features = torch.zeros(2, len(BALANCED_FEATURE_NAMES))
    changes = torch.zeros(2, 3, len(COMPONENT_NAMES))
    observed = torch.ones(2, 3, dtype=torch.bool)
    model.fit_normalization(features, changes, observed)
    model.risk_positive_weight.copy_(torch.tensor([4.0, 1.0, 0.25]))
    _, risk = model.predict(features)
    torch.testing.assert_close(risk, torch.tensor([[0.2, 0.5, 0.8], [0.2, 0.5, 0.8]]))
    assert torch.equal(model.risk_positive_weight, torch.tensor([4.0, 1.0, 0.25]))


def test_complete_checksum_cannot_admit_final_test_fold_or_missing_action(tmp_path):
    cfg = TradeoffConfig()
    p = write_table(tmp_path / "table", "train", cfg)
    original = [json.loads(line) for line in (p / "samples.jsonl").read_text().splitlines()]
    metadata = json.loads((p / "result.json").read_text())
    rows = json.loads(json.dumps(original))
    rows[0]["sample_id"] = rows[0]["sample_id"].replace("dataset_1/", "dataset_8/")
    ids = [r["sample_id"] for r in rows]
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    metadata.update(
        sample_ids=ids,
        sample_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        outcome_sha256=hashlib.sha256(data).hexdigest(),
    )
    (p / "samples.jsonl").write_bytes(data)
    (p / "result.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="original-SCARED"):
        load_outcome_table(p, "train")
    original[0]["outcomes"].pop()
    ids = [r["sample_id"] for r in original]
    data = "".join(json.dumps(r) + "\n" for r in original).encode()
    metadata.update(
        sample_ids=ids,
        sample_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        outcome_sha256=hashlib.sha256(data).hexdigest(),
    )
    (p / "samples.jsonl").write_bytes(data)
    (p / "result.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="missing interventions"):
        load_outcome_table(p, "train")
