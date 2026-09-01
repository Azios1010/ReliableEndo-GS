"""Deterministic CPU smoke tests for bounded uncertainty training/artifacts."""

import torch

from reliable_endo_gs.training.uncertainty import (
    evaluate_uncertainty_head,
    load_uncertainty_checkpoint,
    save_uncertainty_checkpoint,
    train_uncertainty_step,
)
from reliable_endo_gs.uncertainty.calibration import TemperatureCalibrator
from reliable_endo_gs.uncertainty.head import LaplaceUncertaintyHead
from reliable_endo_gs.uncertainty.providers import UncertaintyProviderProvenance


def test_cpu_training_update_checkpoint_roundtrip_and_deterministic_eval(tmp_path) -> None:
    torch.manual_seed(7)
    head = LaplaceUncertaintyHead(2, hidden_channels=4, epsilon=1e-3)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.05)
    features = torch.randn(1, 2, 2, 3)
    predicted = torch.zeros(1, 1, 2, 3)
    target = torch.ones_like(predicted)
    valid = torch.ones_like(predicted, dtype=torch.bool)

    result = train_uncertainty_step(head, optimizer, features, predicted, target, valid)

    assert result.valid_count == 6
    assert result.gradient_norm > 0.0
    assert result.parameter_update_norm > 0.0
    first = evaluate_uncertainty_head(head, features)
    second = evaluate_uncertainty_head(head, features)
    assert torch.equal(first, second)

    provenance = UncertaintyProviderProvenance(
        provider_id="learned_laplace_head",
        architecture_id="laplace_uncertainty_head.v1",
        config_identity="synthetic_config",
        checkpoint_identity="synthetic_untrained",
        schema_version="calibrated_disparity_sigma.v1",
        calibration_id="laplace_nll_v1:val-sha",
        feature_schema="audited_pre_render_stereo_features.v1",
    )
    checkpoint = tmp_path / "uncertainty.pt"
    save_uncertainty_checkpoint(checkpoint, head, provenance, optimizer=optimizer, step=1)

    restored = LaplaceUncertaintyHead(2, hidden_channels=4, epsilon=1e-3)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.05)
    restored_provenance = load_uncertainty_checkpoint(
        checkpoint, restored, optimizer=restored_optimizer
    )
    assert restored_provenance == provenance
    assert torch.equal(first, evaluate_uncertainty_head(restored, features))


def test_training_smoke_output_can_enter_calibrated_boundary() -> None:
    torch.manual_seed(8)
    head = LaplaceUncertaintyHead(1, hidden_channels=2)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
    features = torch.ones(1, 1, 1, 2)
    predicted = torch.zeros(1, 1, 1, 2)
    target = torch.ones_like(predicted)
    valid = torch.ones_like(predicted, dtype=torch.bool)
    train_uncertainty_step(head, optimizer, features, predicted, target, valid)
    raw_sigma = evaluate_uncertainty_head(head, features)
    result = TemperatureCalibrator(temperature=2.0, fit_split_identity="val-sha").apply(
        raw_sigma, valid, provider_id="learned_laplace_head"
    )
    assert bool(torch.isfinite(result.sigma_d).all())
