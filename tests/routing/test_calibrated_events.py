"""Signed utility and local/global/any calibration survive native evaluation."""

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from reliable_endo_gs.actions.repair import Action
from reliable_endo_gs.config.intervention import load_intervention_config
from reliable_endo_gs.config.selector_refit import SelectorRefitConfig
from reliable_endo_gs.evaluation.gain_events import gain_event_metrics
from reliable_endo_gs.evaluation.intervention_selection import (
    _action_availability,
    _selection_outputs,
)
from reliable_endo_gs.evaluation.risk_events import probability_metrics, risk_event_metrics
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.calibrated_selector import (
    CalibratedSelector,
    load_calibrated_selector,
)
from reliable_endo_gs.routing.repairability import select_repairability
from reliable_endo_gs.routing.tokens import selector_feature_names
from reliable_endo_gs.training.selector_fitting import calibrate_selector
from reliable_endo_gs.training.selector_outcomes import SelectorOutcomes


def calibration_inputs():
    config = replace(
        load_intervention_config(Path("configs/routing/intervention_metric_signed_gain.yaml")),
        selector_feature_mode="statistics",
        selector_token_channels=0,
        normalization_frames=1,
        normalization_rows=32,
    )
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(8, len(selector_feature_names(config)), generator=generator)
    gain = torch.zeros(8, 3)
    gain[:, 1:] = torch.tensor([[0.2, -0.1], [-0.1, 0.2]]).repeat(4, 1)
    events = torch.zeros(8, 3, 2)
    events[:, 1, 0] = torch.arange(8) % 2
    events[:, 1, 1] = (torch.arange(8) // 2) % 2
    # Gaussian local harm has no positive class, while global and any do.
    events[:, 2, 1] = torch.arange(8) % 2
    harms = events.bool().any(-1).float()
    ids = tuple(f"dataset_1/keyframe_{5 + i // 4}/frame_data{i:06d}" for i in range(8))
    table = SelectorOutcomes(
        {"collection_role": "train"}, [], features, gain, gain * 0, harms, ids, events
    )
    head = BalancedSelector(8, config)
    head.normalization.observe(features, gain, gain * 0, sample_id=ids[0], role="final_train")
    return CalibratedSelector(head, config, {"model": "frozen"}, "mask"), table


def test_signed_and_event_calibration_are_supported_independently_and_evaluation_is_read_only():
    model, table = calibration_inputs()
    head_before = copy.deepcopy(model.head.state_dict())
    report = calibrate_selector(
        model, table, SelectorRefitConfig(), excluded_keyframes={"dataset_1/keyframe_1"}
    )
    assert model.artifact_schema == "calibrated_selector.v2"
    assert model.calibration_available.tolist() == [True, True]
    assert model.event_calibration_available.tolist() == [[True, True], [False, True]]
    unsupported = report["actions"]["GAUSSIAN"]["events"]["local"]
    assert unsupported["supported"] is False
    assert unsupported["output"] == "raw_identity_transform"
    assert report["cutoffs_tuned"] is False
    assert report["development_used_for_calibration"] is False
    assert all(
        torch.equal(value, head_before[key]) for key, value in model.head.state_dict().items()
    )

    # Affine E[utility] calibration does not relabel raw sign/magnitude outputs.
    model.gain_slope.fill_(0.5)
    model.gain_bias.fill_(0.03)
    before = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        raw_gain, _, logits, raw_parts = model.head.outputs_with_logits(table.features)
        gain, risks, events, parts = _selection_outputs(model, table.features)
    assert events.shape == (8, 3, 3) and parts is not None and raw_parts is not None
    torch.testing.assert_close(gain[:, 1:], raw_gain[:, 1:] * 0.5 + 0.03)
    torch.testing.assert_close(parts.expected_gain, raw_parts.expected_gain)
    torch.testing.assert_close(events[:, 2, 0], logits[:, 1, 0].sigmoid())
    torch.testing.assert_close(risks, events[:, :, 2])
    assert torch.equal(events[:, 0], torch.zeros(8, 3))
    metadata = model.prediction_metadata()
    assert metadata["gain_components_reconstruct"] == "raw_expected_gain_before_affine"
    assert metadata["risk_event_calibration_available"] == [[True, True, True], [False, True, True]]
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    # The evaluator's exact outputs feed both downstream metric paths completely.
    record = {
        "valid_regions": [True] * 8,
        "actual_label_available": [[True] * 3 for _ in range(8)],
        "actual_gains": table.global_gain.tolist(),
        "actual_harms": [[False] * 3 for _ in range(8)],
        "predicted_risk_events": events.tolist(),
        "predicted_gain_components": parts.to_report(),
        "prediction_metadata": metadata,
        "policies": {
            "selector": {"action": "STOP", "region": None},
            "oracle": {"action": "STOP", "region": None},
        },
        "counterfactual_metric_outcomes": [
            {"region_id": ri, "action": action, "metric_changes": [0.0] * 12}
            for ri in range(8)
            for action in ("STEREO", "GAUSSIAN")
        ],
    }
    assert (
        risk_event_metrics([record], model.config)["STEREO"]["candidates"]["any"]["observations"]
        == 8
    )
    gain_report = gain_event_metrics([record])["STEREO"]
    assert gain_report["known_candidates"] == 8
    assert gain_report["prediction_space"] == "raw_head_sign_and_magnitudes"


def test_event_calibration_rejects_missing_or_inconsistent_labels_and_any_without_support_abstains():
    model, table = calibration_inputs()
    for labels in (None, torch.zeros_like(table.risk_components)):
        with pytest.raises(ValueError, match="local/global labels consistent"):
            calibrate_selector(
                model,
                replace(table, risk_components=labels),
                SelectorRefitConfig(),
                excluded_keyframes={"dataset_1/keyframe_1"},
            )
    unknown_classes = replace(
        table,
        harms=torch.zeros_like(table.harms),
        risk_components=torch.zeros_like(table.risk_components),
    )
    calibrate_selector(
        model, unknown_classes, SelectorRefitConfig(), excluded_keyframes={"dataset_1/keyframe_1"}
    )
    assert not bool(model.calibration_available.any())
    decision = model.select(
        table.features,
        torch.tensor([0.0, 1.0, 1.0]).expand(8, -1),
        torch.ones(8, dtype=torch.bool),
        1000,
    )
    assert decision.action == Action.STOP


def test_grouped_calibration_artifact_round_trip_and_event_support_validation(tmp_path):
    model, table = calibration_inputs()
    report = calibrate_selector(
        model, table, SelectorRefitConfig(), excluded_keyframes={"dataset_1/keyframe_1"}
    )
    raw = {
        "schema": model.artifact_schema,
        "width": 8,
        "head_config": model.head.config.to_dict(),
        "decision_config": model.config.to_dict(),
        "action_identity": {"model": "frozen"},
        "target_mask_index_sha256": "mask",
        "source_sha256": "source",
        "development_used_for_fitting": False,
        "state": model.state_dict(),
        "calibration": report,
        "partition": {
            "parent_roles_changed": False,
            "fit_keyframes": ["dataset_1/keyframe_1", "dataset_1/keyframe_2"],
            "validation_keyframes": ["dataset_1/keyframe_3", "dataset_1/keyframe_4"],
            "calibration_keyframes": ["dataset_1/keyframe_5", "dataset_1/keyframe_6"],
        },
    }
    path = tmp_path / "calibrated.pt"
    torch.save(raw, path)
    binding = {
        "expected_action_identity": {"model": "frozen"},
        "expected_config": model.config,
        "expected_mask_identity": "mask",
        "expected_source_sha256": "source",
    }
    loaded = load_calibrated_selector(path, **binding)
    for actual, expected in zip(
        loaded.predict(table.features), model.predict(table.features), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    raw["calibration"]["actions"]["GAUSSIAN"]["events"]["local"]["supported"] = True
    torch.save(raw, path)
    with pytest.raises(ValueError, match="event calibration availability"):
        load_calibrated_selector(path, **binding)


def test_missing_evaluation_events_and_components_raise_explicit_errors():
    model, _ = calibration_inputs()
    record = {
        "valid_regions": [True],
        "predicted_risk_events": [],
        "predicted_gain_components": {
            name: [] for name in ("positive_probability", "positive_gain", "negative_cost")
        },
    }
    with pytest.raises(ValueError, match="predicted_risk_events"):
        risk_event_metrics([record], model.config)
    with pytest.raises(ValueError, match="predicted_gain_components.positive_probability"):
        gain_event_metrics([record])


def test_probability_diagnostics_measure_discrimination_and_actual_harm_without_retuning():
    good = probability_metrics([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
    assert good["auroc"] == 1
    assert good["brier"] == pytest.approx(0.025)
    assert good["ECE_10_bins"] == pytest.approx(0.15)
    assert good["harm_rate"] == 0.5
    tied = probability_metrics([0.5] * 4, [False, False, True, True])
    assert tied["auroc"] == 0.5
    assert probability_metrics([0.2], [False])["auroc"] is None


def test_native_evaluator_masks_nonexecutable_actions_before_selection(monkeypatch):
    state = SimpleNamespace(batch=SimpleNamespace(left=torch.zeros(1, 3, 2, 2)))
    region = object()
    calls = []

    def precondition(current, candidate, action):
        assert current is state and candidate is region
        calls.append(action)
        return action != Action.STEREO

    monkeypatch.setattr(
        "reliable_endo_gs.evaluation.intervention_selection.action_eligible", precondition
    )
    executable = _action_availability(state, (region,))
    assert calls == list(Action)
    assert executable.tolist() == [[True, False, True]]
    model, _ = calibration_inputs()
    decision = select_repairability(
        torch.tensor([[0.0, 100.0, -1.0]]),
        torch.zeros(1, 3),
        torch.tensor([[0.0, 1.0, 1.0]]),
        executable,
        1000,
        model.config,
    )
    assert decision.action == Action.STOP
    assert _action_availability(state, ()).shape == (0, 3)


def test_nonexecutable_risk_candidates_are_distinct_from_unobserved_gt():
    model, _ = calibration_inputs()
    record = {
        "valid_regions": [True],
        "predicted_risk_events": [[[0.0] * 3, [0.5] * 3, [0.5] * 3]],
        "actual_label_available": [[True, False, False]],
        "counterfactual_metric_outcomes": [
            {"region_id": 0, "action": "STEREO", "action_available": False, "metric_changes": None},
            {
                "region_id": 0,
                "action": "GAUSSIAN",
                "action_available": True,
                "metric_changes": None,
            },
        ],
    }
    report = risk_event_metrics([record], model.config)
    assert report["STEREO"]["nonexecutable_candidates"] == 1
    assert report["STEREO"]["unknown_candidates"] == 0
    assert report["GAUSSIAN"]["nonexecutable_candidates"] == 0
    assert report["GAUSSIAN"]["unknown_candidates"] == 1
