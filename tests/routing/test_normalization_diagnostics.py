"""Normalization drift remains observable without changing the frozen transform."""

import copy

import pytest
import torch

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.routing.normalization import TrainingNormalization


def test_diagnostics_preserve_fixed_32_snapshot_normalization_and_expose_drift():
    config = InterventionConfig(schema="intervention_selection.v2")
    normalization = TrainingNormalization(3, (2,), config)
    gains = torch.zeros(2, 3)
    for snapshot in range(32):
        features = torch.tensor([[float(snapshot), 7.0, 0.0], [float(snapshot + 1), 7.0, 1.0]])
        normalization.observe(
            features,
            gains,
            gains,
            sample_id=f"dataset_1/keyframe_1/frame_{snapshot:06d}",
            role="final_train",
        )
        assert bool(normalization.fitted) is (snapshot == 31)
    before = copy.deepcopy(normalization.state_dict())
    values = torch.tensor([[1000.0, 8.0, 1.0], [-1000.0, 7.0, 0.0]])
    report = normalization.diagnostics(values)
    assert report["snapshots_observed"] == 32
    assert report["active_features"] == 2
    assert report["clipped_values"] == 2
    assert report["clipped_values_per_feature"] == [2, 0, 0]
    assert report["clipped_fraction"] == pytest.approx(1 / 3)
    assert report["inactive_changed_values"] == 1
    assert report["outside_fit_range_values"] == 3
    assert torch.equal(normalization(values)[:, 1], torch.zeros(2))
    normalization.observe(values, gains, gains, sample_id="dataset_1/later", role="final_train")
    assert all(torch.equal(value, normalization.state_dict()[key]) for key, value in before.items())


def test_unfitted_diagnostics_do_not_report_clipping_or_mutate_fit_state():
    normalization = TrainingNormalization(2, (), InterventionConfig())
    values = torch.tensor([[1000.0, -1000.0]])
    report = normalization.diagnostics(values)
    assert report["fitted"] is False and report["clipped_values"] == 0
    assert report["outside_fit_range_values"] is None
    assert torch.equal(normalization(values), values)
    assert int(normalization.counts.sum()) == 0
    with pytest.raises(ValueError, match="finite"):
        normalization.diagnostics(torch.tensor([[float("nan"), 0.0]]))
