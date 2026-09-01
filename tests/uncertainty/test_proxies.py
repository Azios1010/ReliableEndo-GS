"""Synthetic CPU tests for deployable raw uncertainty proxies."""

import pytest
import torch

from reliable_endo_gs.contracts import StereoPrediction
from reliable_endo_gs.uncertainty.proxies import (
    FinalUpdateMagnitudeProxy,
    IterationDisagreementProxy,
    left_right_consistency_score,
)


def _prediction(iterations: torch.Tensor) -> StereoPrediction:
    return StereoPrediction(
        disparity=iterations[:, -1:],
        valid_mask=torch.ones_like(iterations[:, -1:], dtype=torch.bool),
        disparity_iterations=iterations,
    )


def test_iteration_proxies_return_raw_scores_with_prediction_mask() -> None:
    iterations = torch.tensor([[[[0.0, 1.0]], [[2.0, 3.0]], [[5.0, 7.0]]]])
    prediction = _prediction(iterations)

    update = FinalUpdateMagnitudeProxy().predict(prediction)
    disagreement = IterationDisagreementProxy(window=3).predict(prediction)

    assert torch.equal(update.score, torch.tensor([[[[3.0, 4.0]]]]))
    assert torch.allclose(disagreement.score, torch.tensor([[[[2.0548, 2.4944]]]]), atol=1e-4)
    assert update.estimator_id == "final_update_magnitude"
    assert torch.equal(update.valid_mask, prediction.valid_mask)


def test_iteration_proxy_requires_real_iteration_evidence() -> None:
    prediction = StereoPrediction(
        disparity=torch.ones(1, 1, 1, 2), valid_mask=torch.ones(1, 1, 1, 2, dtype=torch.bool)
    )
    with pytest.raises(ValueError, match="exposes none"):
        FinalUpdateMagnitudeProxy().predict(prediction)


def test_left_right_consistency_uses_x_minus_left_disparity_and_masks_out_of_view() -> None:
    left = torch.tensor([[[[1.0, 1.0, 1.0]]]])
    right = torch.tensor([[[[-1.0, -1.0, -1.0]]]])
    valid = torch.ones_like(left, dtype=torch.bool)

    result = left_right_consistency_score(left, right, valid, valid)

    assert torch.equal(result.valid_mask, torch.tensor([[[[False, True, True]]]]))
    assert torch.equal(result.score, torch.zeros_like(left))


def test_proxy_rejects_nonfinite_score_on_a_valid_pixel() -> None:
    iterations = torch.tensor([[[[0.0]], [[float("nan")]]]])
    prediction = _prediction(iterations)

    with pytest.raises(ValueError, match="finite"):
        FinalUpdateMagnitudeProxy().predict(prediction)


def test_left_right_consistency_validates_mask_dtype_and_disparity_dtype() -> None:
    disparity = torch.ones(1, 1, 1, 2)
    right = disparity.clone()
    valid = torch.ones_like(disparity, dtype=torch.bool)

    with pytest.raises(TypeError, match="torch.bool"):
        left_right_consistency_score(disparity, right, valid.to(torch.float32), valid)
    with pytest.raises(TypeError, match="floating"):
        left_right_consistency_score(disparity.to(torch.int64), right.to(torch.int64), valid, valid)
