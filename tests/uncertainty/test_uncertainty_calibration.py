"""Validation-only calibration and deployable provider tests."""

import pytest
import torch

from reliable_endo_gs.contracts import StereoPrediction
from reliable_endo_gs.uncertainty.calibration import TemperatureCalibrator
from reliable_endo_gs.uncertainty.providers import CalibratedProxySigmaProvider
from reliable_endo_gs.uncertainty.proxies import FinalUpdateMagnitudeProxy
from reliable_endo_gs.uncertainty.records import RawUncertaintyScore


def test_temperature_fit_is_laplace_closed_form_and_rejects_test_split() -> None:
    sigma = torch.ones(1, 1, 1, 2)
    error = torch.ones(1, 1, 1, 2)
    valid = torch.ones_like(sigma, dtype=torch.bool)

    calibrator = TemperatureCalibrator.fit(
        sigma, error, valid, split_role="validation", split_identity="val-sha"
    )

    assert calibrator.temperature == pytest.approx(2.0**0.5)
    with pytest.raises(ValueError, match="validation data only"):
        TemperatureCalibrator.fit(sigma, error, valid, split_role="test", split_identity="test-sha")


def test_calibrated_proxy_provider_never_accepts_ground_truth() -> None:
    iterations = torch.tensor([[[[0.0]], [[1.0]]]])
    prediction = StereoPrediction(
        disparity=iterations[:, -1:],
        valid_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
        disparity_iterations=iterations,
    )
    calibrator = TemperatureCalibrator(temperature=2.0, fit_split_identity="val-sha")

    result = CalibratedProxySigmaProvider(FinalUpdateMagnitudeProxy(), calibrator).predict(
        prediction
    )

    assert torch.equal(result.sigma_d, torch.full((1, 1, 1, 1), 2.0))
    assert result.calibration_id.endswith("val-sha")


def test_zero_residual_calibration_uses_neutral_positive_temperature() -> None:
    sigma = torch.ones(1, 1, 1, 2)
    error = torch.zeros_like(sigma)
    valid = torch.ones_like(sigma, dtype=torch.bool)

    calibrator = TemperatureCalibrator.fit(
        sigma, error, valid, split_role="validation", split_identity="val-sha"
    )

    assert calibrator.temperature == pytest.approx(1.0)


def test_calibrated_proxy_rejects_negative_scores_instead_of_flooring_them() -> None:
    class NegativeProvider:
        def predict(self, prediction: StereoPrediction) -> RawUncertaintyScore:
            return RawUncertaintyScore(
                torch.full_like(prediction.disparity, -1.0),
                prediction.valid_mask,
                "negative",
            )

    prediction = StereoPrediction(
        disparity=torch.ones(1, 1, 1, 1),
        valid_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
    )
    calibrator = TemperatureCalibrator(temperature=1.0, fit_split_identity="val-sha")

    with pytest.raises(ValueError, match="non-negative"):
        CalibratedProxySigmaProvider(NegativeProvider(), calibrator).predict(prediction)


def test_calibrated_proxy_floors_only_exact_zero_score() -> None:
    class ZeroProvider:
        def predict(self, prediction: StereoPrediction) -> RawUncertaintyScore:
            return RawUncertaintyScore(
                torch.zeros_like(prediction.disparity),
                prediction.valid_mask,
                "zero",
            )

    prediction = StereoPrediction(
        disparity=torch.ones(1, 1, 1, 1),
        valid_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
    )
    calibrator = TemperatureCalibrator(temperature=2.0, fit_split_identity="val-sha")
    result = CalibratedProxySigmaProvider(ZeroProvider(), calibrator).predict(prediction)

    assert torch.equal(result.sigma_d, torch.full_like(prediction.disparity, 2e-4))


def test_temperature_calibrator_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        TemperatureCalibrator(temperature=0.0, fit_split_identity="val-sha")
