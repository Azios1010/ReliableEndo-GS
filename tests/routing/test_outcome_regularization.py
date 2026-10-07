"""Independent grouped stopping, deterministic inference and artifact provenance."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest
import torch

from reliable_endo_gs.config.outcome_training import OutcomeTrainingConfig
from reliable_endo_gs.routing.outcome_predictor import OutcomePredictor, load_outcome_predictor
from reliable_endo_gs.routing.tradeoff import TradeoffConfig
from reliable_endo_gs.training import outcome_learning
from reliable_endo_gs.training.outcome_calibration import (
    split_training_calibration,
    split_training_validation_calibration,
)
from reliable_endo_gs.training.outcome_learning import load_outcome_table
from reliable_endo_gs.training.outcome_validation import validate_training_views, validation_score

from .test_balanced_outcomes import balanced_config, grouped_table


def four_keyframe_table(path, role="train", offset=0):
    grouped_table(path, role, TradeoffConfig(), offset=offset)
    rows = [json.loads(s) for s in (path / "samples.jsonl").read_text().splitlines()]
    for i, row in enumerate(rows):
        row["sample_id"] = f"dataset_{i % 7 + 1}/keyframe_{i // 7 + 1}/frame_{i + offset:06d}"
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    ids = [r["sample_id"] for r in rows]
    meta = json.loads((path / "result.json").read_text())
    meta.update(
        sample_ids=ids,
        sample_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        outcome_sha256=hashlib.sha256(data).hexdigest(),
    )
    (path / "samples.jsonl").write_bytes(data)
    (path / "result.json").write_text(json.dumps(meta))
    return path


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dropout": 1.0},
        {"dropout": float("nan")},
        {"weight_decay": -1.0},
        {"validate_every": 0},
        {"patience": True},
        {"min_steps": 101, "max_steps": 100},
    ],
)
def test_training_controls_are_explicit(kwargs):
    with pytest.raises(ValueError):
        OutcomeTrainingConfig(**kwargs)


def test_three_groups_preserve_calibration_and_parent_table(tmp_path):
    path = four_keyframe_table(tmp_path / "train")
    before = {p: p.read_bytes() for p in path.iterdir()}
    table = load_outcome_table(path, "train")
    cfg = balanced_config()
    _, old_cal, _ = split_training_calibration(table, cfg)
    fit, val, cal, definition = split_training_validation_calibration(table, cfg)
    groups = [set(r["sample_id"].rsplit("/", 1)[0] for r in t.records) for t in (fit, val, cal)]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    assert len(fit.records) == 14 and len(val.records) == len(cal.records) == 7
    assert [r["sample_id"] for r in old_cal.records] == definition["calibration_sample_ids"]
    assert not definition["validation_used_for_calibration"]
    assert before == {p: p.read_bytes() for p in before}
    two = load_outcome_table(grouped_table(tmp_path / "two", "train", TradeoffConfig()), "train")
    with pytest.raises(ValueError, match="three"):
        split_training_validation_calibration(two, cfg)
    with pytest.raises(ValueError, match="TRAIN"):
        validate_training_views(
            fit, replace(val, metadata={**val.metadata, "collection_role": "development"})
        )
    with pytest.raises(ValueError, match="disjoint"):
        validate_training_views(fit, fit)


def test_dropout_keeps_weight_layout_and_evaluation_deterministic():
    torch.manual_seed(0)
    old = OutcomePredictor(8, component_risk=True)
    model = OutcomePredictor(8, component_risk=True, dropout=0.3)
    assert set(old.state_dict()) == set(model.state_dict())
    x = torch.randn(32, 61)
    y = torch.zeros(32, 3, 12)
    mask = torch.ones(32, 3, dtype=torch.bool)
    model.fit_normalization(x, y, mask)
    with torch.no_grad():
        model.change_network[-1].weight.fill_(0.2)
    a, _ = model(x)
    b, _ = model(x)
    assert not torch.equal(a, b)
    model.eval()
    a, _ = model(x)
    b, _ = model(x)
    assert torch.equal(a, b)


def test_patience_restores_exact_best_state_and_stops(tmp_path, monkeypatch):
    table = load_outcome_table(four_keyframe_table(tmp_path / "train"), "train")
    cfg = balanced_config(width=8)
    fit, val, _, _ = split_training_validation_calibration(table, cfg)
    captured = []

    def fixed_score(model, table, config):
        captured.append({k: v.clone() for k, v in model.state_dict().items()})
        return {"score": 1.0, "regression": 0.5, "risk_log_loss": 0.5, "known_keyframes": 7.0}

    monkeypatch.setattr(outcome_learning, "validation_score", fixed_score)
    policy = OutcomeTrainingConfig(max_steps=100, min_steps=10, validate_every=10, patience=2)
    model, logs, selection = outcome_learning.fit_regularized_outcome_predictor(
        fit, val, cfg, policy
    )
    assert selection["best_step"] == 10 and selection["completed_steps"] == 30
    assert selection["stop_reason"] == "validation_patience" and selection["best_state_restored"]
    assert all(torch.equal(v, captured[0][k]) for k, v in model.state_dict().items())
    assert not model.training
    assert any("validation_score" in r for r in logs)


def test_validation_does_not_refit_normalization_or_change_mode(tmp_path):
    table = load_outcome_table(four_keyframe_table(tmp_path / "train"), "train")
    cfg = balanced_config(epochs=3, width=8)
    fit, val, _, _ = split_training_validation_calibration(table, cfg)
    model, _ = outcome_learning.fit_outcome_predictor(fit, cfg)
    model.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    score = validation_score(model, val, cfg)
    assert score["known_keyframes"] == 7 and model.training
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_regularized_pilot_reload_binds_validation_and_training_policy(tmp_path):
    train = four_keyframe_table(tmp_path / "train")
    dev = four_keyframe_table(tmp_path / "dev", "development", 100)
    cfg = balanced_config(width=8, bootstrap_repetitions=5)
    path = tmp_path / "metric.json"
    path.write_text(json.dumps(asdict(cfg)))
    policy = OutcomeTrainingConfig(max_steps=50, min_steps=10, validate_every=10, patience=2)
    pp = tmp_path / "training.json"
    pp.write_text(json.dumps(asdict(policy)))
    result = outcome_learning.run_outcome_pilot(
        training_table=train,
        development_table=dev,
        config_path=path,
        training_policy_path=pp,
        output_dir=tmp_path / "out",
        relabel_policy=True,
        budgets_ms=(1000.0,),
    )
    assert result["fit_frames"] == 14
    assert result["training_selection"]["mlp"]["best_state_restored"]
    assert not result["training_selection"]["mlp"]["development_used_for_selection"]
    kwargs = dict(
        expected_config=cfg,
        expected_action_identity={"checkpoint_sha256": "synthetic"},
        expected_candidate_hash="candidate",
    )
    model = load_outcome_predictor(tmp_path / "out/mlp.pt", **kwargs)
    assert model.change_network.dropout == 0.1 and not model.training
    for key, value in (("training_policy_hash", "unverified"), ("best_step", 0)):
        changed = torch.load(tmp_path / "out/mlp.pt", weights_only=True)
        changed["training_selection"][key] = value
        torch.save(changed, tmp_path / "selection.pt")
        with pytest.raises(ValueError, match="checkpoint selection"):
            load_outcome_predictor(tmp_path / "selection.pt", **kwargs)
    raw = torch.load(tmp_path / "out/mlp.pt", weights_only=True)
    raw["training_policy"]["dropout"] = 0.2
    torch.save(raw, tmp_path / "changed.pt")
    with pytest.raises(ValueError, match="policy identity"):
        load_outcome_predictor(tmp_path / "changed.pt", **kwargs)
    raw["training_policy"]["dropout"] = 0.1
    definition = raw["training_partition"]
    definition["validation_sample_ids"][0] = definition["fit_sample_ids"][0]
    definition["partition_sha256"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in definition.items() if k != "partition_sha256"}, sort_keys=True
        ).encode()
    ).hexdigest()
    torch.save(raw, tmp_path / "overlap.pt")
    with pytest.raises(ValueError, match="separation"):
        load_outcome_predictor(tmp_path / "overlap.pt", **kwargs)
