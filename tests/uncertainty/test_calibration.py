"""Calibration boundary and learned-provider contract tests."""

import pytest
import torch

from reliable_endo_gs.contracts import StereoPrediction
from reliable_endo_gs.uncertainty.calibration import TemperatureCalibrator
from reliable_endo_gs.uncertainty.head import LaplaceUncertaintyHead
from reliable_endo_gs.uncertainty.providers import (
    UNCERTAINTY_FEATURE_SCHEMA,
    UNCERTAINTY_FEATURES_KEY,
    LearnedLaplaceSigmaProvider,
)


def test_temperature_fit_requires_nonempty_validation_identity_and_matching_dtype() -> None:
    sigma = torch.ones(1, 1, 1, 1)
    error = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    valid = torch.ones_like(sigma, dtype=torch.bool)

    with pytest.raises(TypeError, match="same dtype"):
        TemperatureCalibrator.fit(
            sigma, error, valid, split_role="validation", split_identity="val-sha"
        )
    with pytest.raises(ValueError, match="non-empty string"):
        TemperatureCalibrator.fit(sigma, sigma, valid, split_role="validation", split_identity="")


def test_learned_provider_uses_schema_tagged_prediction_diagnostics() -> None:
    torch.manual_seed(4)
    head = LaplaceUncertaintyHead(2, hidden_channels=3)
    calibrator = TemperatureCalibrator(temperature=1.5, fit_split_identity="val-sha")
    provider = LearnedLaplaceSigmaProvider(head, calibrator)
    features = torch.randn(1, 2, 2, 3)
    prediction = StereoPrediction(
        disparity=torch.ones(1, 1, 2, 3),
        valid_mask=torch.ones(1, 1, 2, 3, dtype=torch.bool),
        diagnostics={
            UNCERTAINTY_FEATURES_KEY: features,
            f"{UNCERTAINTY_FEATURES_KEY}.schema": UNCERTAINTY_FEATURE_SCHEMA,
        },
    )

    result = provider.predict(prediction)

    assert result.sigma_d.shape == prediction.disparity.shape
    assert result.provider_id == "learned_laplace_head"
    assert provider.provenance.checkpoint_identity == "synthetic_untrained"


def test_learned_provider_rejects_unaudited_feature_boundary() -> None:
    provider = LearnedLaplaceSigmaProvider(
        LaplaceUncertaintyHead(1),
        TemperatureCalibrator(temperature=1.0, fit_split_identity="val-sha"),
    )
    prediction = StereoPrediction(
        disparity=torch.ones(1, 1, 1, 1),
        valid_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
        diagnostics={UNCERTAINTY_FEATURES_KEY: torch.ones(1, 1, 1, 1)},
    )

    with pytest.raises(ValueError, match="schema"):
        provider.predict(prediction)


def test_temperature_fit_rejects_empty_nonfinite_and_negative_targets() -> None:
    sigma = torch.ones(1, 1, 1, 1)
    valid = torch.ones_like(sigma, dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one valid"):
        TemperatureCalibrator.fit(
            sigma, torch.zeros_like(sigma), ~valid, split_role="validation", split_identity="val"
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        TemperatureCalibrator.fit(
            sigma,
            torch.full_like(sigma, float("nan")),
            valid,
            split_role="validation",
            split_identity="val",
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        TemperatureCalibrator.fit(
            sigma,
            torch.full_like(sigma, -1.0),
            valid,
            split_role="validation",
            split_identity="val",
        )
