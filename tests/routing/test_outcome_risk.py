"""Risk discrimination, independent encoder selection and calibration boundaries."""

import json
from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from reliable_endo_gs.config.outcome_training import OutcomeTrainingConfig
from reliable_endo_gs.routing.outcome_predictor import OutcomePredictor, load_outcome_predictor
from reliable_endo_gs.training import outcome_learning
from reliable_endo_gs.training.outcome_calibration import (
    calibrate_joint_risk,
    fit_platt_risk,
    split_training_validation_calibration,
)
from reliable_endo_gs.training.outcome_risk import refit_conditional_risk, risk_ranking_loss

from .test_balanced_outcomes import balanced_config
from .test_outcome_regularization import four_keyframe_table


def test_historical_training_hash_and_explicit_v2_controls():
    assert (
        OutcomeTrainingConfig().config_hash
        == "5c524ab2335b49e10fe23e496e09052cf79eca03a98a94fafd46cc4d4c9200c1"
    )
    with pytest.raises(ValueError, match="v2"):
        OutcomeTrainingConfig(risk_ranking_weight=0.25)
    with pytest.raises(ValueError, match="boolean"):
        OutcomeTrainingConfig(schema="outcome_training.v2", independent_encoder_stopping=1)
    assert (
        OutcomeTrainingConfig(
            schema="outcome_training.v2",
            risk_ranking_weight=0.25,
            independent_encoder_stopping=True,
            calibration_method="platt",
        ).config_hash
        == "39eeb4e989161138bcede650046ece33f2f7541f11392b86d0a2906dd2bac07b"
    )


def test_ranking_separates_safe_unsafe_and_masks_unknown():
    logits = torch.zeros(3, 2, requires_grad=True)
    labels = torch.tensor([[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    mask = torch.tensor([[True, True], [True, True], [False, True]])
    loss = risk_ranking_loss(logits, labels, mask)
    loss.backward()
    assert logits.grad[0, 0] > 0 and logits.grad[1, 0] < 0
    assert logits.grad[2, 0] == 0 and torch.equal(logits.grad[:, 1], torch.zeros(3))
    better = torch.tensor([[-3.0, 0.0], [3.0, 0.0], [1000.0, 0.0]])
    assert risk_ranking_loss(better, labels, mask) < loss


def test_one_class_ranking_is_zero_with_finite_gradients():
    logits = torch.randn(5, 3, requires_grad=True)
    loss = risk_ranking_loss(
        logits, torch.ones_like(logits), torch.ones_like(logits, dtype=torch.bool)
    )
    loss.backward()
    assert loss == 0 and torch.equal(logits.grad, torch.zeros_like(logits))
    with pytest.raises(ValueError, match="binary"):
        risk_ranking_loss(logits, logits, torch.ones_like(logits, dtype=torch.bool))


def test_independent_stopping_restores_distinct_best_encoder_states(tmp_path, monkeypatch):
    table = outcome_learning.load_outcome_table(four_keyframe_table(tmp_path / "train"), "train")
    cfg = balanced_config(width=8)
    fit, val, _, _ = split_training_validation_calibration(table, cfg)
    captured = []
    values = [(3.0, 1.0), (2.0, 2.0), (1.0, 3.0), (2.0, 4.0), (3.0, 5.0)]

    def scores(model, table, config):
        regression, risk = values[min(len(captured), len(values) - 1)]
        captured.append({k: v.clone() for k, v in model.state_dict().items()})
        return {"score": regression + risk, "regression": regression, "risk_log_loss": risk}

    monkeypatch.setattr(outcome_learning, "validation_score", scores)
    policy = OutcomeTrainingConfig(
        schema="outcome_training.v2",
        max_steps=100,
        min_steps=10,
        validate_every=10,
        patience=2,
        independent_encoder_stopping=True,
        risk_ranking_weight=0.25,
    )
    model, logs, selection = outcome_learning.fit_regularized_outcome_predictor(
        fit, val, cfg, policy
    )
    assert selection["completed_steps"] == 50
    assert selection["encoder_selection"]["changes"]["best_step"] == 30
    assert selection["encoder_selection"]["risk"]["best_step"] == 10
    for key, value in model.state_dict().items():
        expected = captured[2 if key.startswith("change_network.") else 0][key]
        assert torch.equal(value, expected)
    assert logs[0]["risk_ranking"] > 0


def test_platt_is_monotone_detached_bounded_and_improves_known_log_loss():
    x = torch.tensor([-2.0, -1.0, 1.0, 2.0], requires_grad=True)
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    temperature, bias = fit_platt_risk(x, labels, torch.ones(4), regularization=0.01)
    assert 0.135 <= temperature <= 7.390 and -4 <= bias <= 4
    assert x.grad is None
    adjusted = (x.detach() / temperature + bias).sigmoid()
    assert torch.all(adjusted[1:] > adjusted[:-1])
    assert F.binary_cross_entropy_with_logits(
        x.detach() / temperature + bias, labels
    ) < F.binary_cross_entropy_with_logits(x.detach(), labels)
    with pytest.raises(ValueError, match="invalid"):
        fit_platt_risk(x, labels, -torch.ones(4), regularization=0.01)


def test_risk_pilot_binds_encoder_and_calibration_provenance(tmp_path):
    train = four_keyframe_table(tmp_path / "train")
    dev = four_keyframe_table(tmp_path / "dev", "development", 100)
    cfg = balanced_config(width=8, bootstrap_repetitions=5)
    policy = OutcomeTrainingConfig(
        schema="outcome_training.v2",
        max_steps=50,
        min_steps=10,
        validate_every=10,
        patience=2,
        independent_encoder_stopping=True,
        calibration_method="platt",
        risk_ranking_weight=0.25,
    )
    cp, pp = tmp_path / "metric.json", tmp_path / "training.json"
    cp.write_text(json.dumps(asdict(cfg)))
    pp.write_text(json.dumps(asdict(policy)))
    result = outcome_learning.run_outcome_pilot(
        training_table=train,
        development_table=dev,
        config_path=cp,
        training_policy_path=pp,
        output_dir=tmp_path / "out",
        relabel_policy=True,
        budgets_ms=(1000.0,),
    )
    assert result["training_risk_calibration"]["mlp"]["method"] == "platt"
    assert "false_unsafe_at_cutoff" in result["calibration"]["mlp"]["learned_stereo"]
    kwargs = dict(
        expected_config=cfg,
        expected_action_identity={"checkpoint_sha256": "synthetic"},
        expected_candidate_hash="candidate",
    )
    load_outcome_predictor(tmp_path / "out/mlp.pt", **kwargs)
    raw = torch.load(tmp_path / "out/mlp.pt", weights_only=True)
    raw["training_selection"]["encoder_selection"]["risk"]["best_step"] = 0
    torch.save(raw, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="encoder selection"):
        load_outcome_predictor(tmp_path / "bad.pt", **kwargs)
    raw = torch.load(tmp_path / "out/mlp.pt", weights_only=True)
    raw["training_risk_calibration"]["regularization"] = 100.0
    torch.save(raw, tmp_path / "bad-calibration.pt")
    with pytest.raises(ValueError, match="calibration regularization"):
        load_outcome_predictor(tmp_path / "bad-calibration.pt", **kwargs)


def test_conditional_features_are_detached_predictions_not_labels():
    cfg = balanced_config(width=8)
    model = OutcomePredictor(8, component_risk=True, risk_context_config=cfg)
    x = torch.randn(12, 61, requires_grad=True)
    model.fit_normalization(x, torch.zeros(12, 3, 12), torch.ones(12, 3, dtype=torch.bool))
    changes, joint, component = model.forward_details(x)
    assert model.risk_network[0].in_features == 97
    assert changes.shape == (12, 3, 12) and joint.shape == (12, 3)
    assert component is not None
    gradients = torch.autograd.grad(
        joint.sum() + component.sum(), tuple(model.change_network.parameters()), allow_unused=True
    )
    assert all(g is None for g in gradients) and x.grad is None


def test_refit_keeps_selected_gain_and_normalization_fixed(tmp_path):
    table = outcome_learning.load_outcome_table(four_keyframe_table(tmp_path / "train"), "train")
    cfg = balanced_config(width=8)
    fit, val, cal, _ = split_training_validation_calibration(table, cfg)
    policy = OutcomeTrainingConfig(
        schema="outcome_training.v2",
        max_steps=30,
        min_steps=10,
        validate_every=10,
        patience=1,
        independent_encoder_stopping=True,
        calibration_method="platt",
        risk_feature_mode="predicted_margins",
    )
    model = OutcomePredictor(8, component_risk=True, dropout=0.1, risk_context_config=cfg)
    model.fit_normalization(fit.features, fit.changes, fit.observed)
    before = {
        k: v.clone() for k, v in model.state_dict().items() if not k.startswith("risk_network.")
    }
    _, report = refit_conditional_risk(model, fit, val, cfg, policy)
    assert report["gain_unchanged"] and report["best_state_restored"]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    model.train()
    calibrate_joint_risk(model, cal, cfg, policy)
    assert model.training


def test_conditional_pilot_reload_requires_fixed_gain_refit(tmp_path):
    train = four_keyframe_table(tmp_path / "train")
    dev = four_keyframe_table(tmp_path / "dev", "development", 100)
    cfg = balanced_config(width=8, bootstrap_repetitions=5)
    policy = OutcomeTrainingConfig(
        schema="outcome_training.v2",
        max_steps=30,
        min_steps=10,
        validate_every=10,
        patience=1,
        independent_encoder_stopping=True,
        calibration_method="platt",
        risk_feature_mode="predicted_margins",
    )
    cp, pp = tmp_path / "metric.json", tmp_path / "training.json"
    cp.write_text(json.dumps(asdict(cfg)))
    pp.write_text(json.dumps(asdict(policy)))
    result = outcome_learning.run_outcome_pilot(
        training_table=train,
        development_table=dev,
        config_path=cp,
        training_policy_path=pp,
        output_dir=tmp_path / "out",
        relabel_policy=True,
        budgets_ms=(1000.0,),
    )
    assert result["training_selection"]["mlp"]["risk_refit"]["gain_unchanged"]
    kwargs = dict(
        expected_config=cfg,
        expected_action_identity={"checkpoint_sha256": "synthetic"},
        expected_candidate_hash="candidate",
    )
    model = load_outcome_predictor(tmp_path / "out/mlp.pt", **kwargs)
    assert model.risk_context_config == cfg and not model.training
    changed = torch.load(tmp_path / "out/mlp.pt", weights_only=True)
    changed["risk_feature_names"] = changed["feature_names"]
    torch.save(changed, tmp_path / "bad-features.pt")
    with pytest.raises(ValueError, match="risk feature provenance"):
        load_outcome_predictor(tmp_path / "bad-features.pt", **kwargs)
    raw = torch.load(tmp_path / "out/mlp.pt", weights_only=True)
    raw["training_selection"]["risk_refit"]["gain_unchanged"] = False
    torch.save(raw, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="conditional risk refit"):
        load_outcome_predictor(tmp_path / "bad.pt", **kwargs)
