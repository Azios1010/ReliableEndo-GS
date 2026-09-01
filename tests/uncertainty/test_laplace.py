"""Synthetic CPU tests for positive learned Laplace uncertainty."""

import math

import pytest
import torch

from reliable_endo_gs.uncertainty.head import LaplaceUncertaintyHead
from reliable_endo_gs.uncertainty.laplace import (
    laplace_nll,
    positive_laplace_scale,
    sigma_from_laplace_scale,
)


def test_laplace_nll_matches_analytic_value_and_backpropagates() -> None:
    predicted = torch.tensor([[[[0.0, 0.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 2.0]]]])
    scale = torch.tensor([[[[2.0, 2.0]]]], requires_grad=True)
    mask = torch.ones_like(predicted, dtype=torch.bool)

    loss = laplace_nll(predicted, target, scale, mask)
    loss.backward()

    assert torch.allclose(loss, torch.tensor(0.75 + math.log(2.0)))
    assert predicted.grad is not None and scale.grad is not None
    assert torch.equal(sigma_from_laplace_scale(scale.detach()), scale.detach() * math.sqrt(2.0))


def test_head_returns_positive_scale_and_expected_shape() -> None:
    head = LaplaceUncertaintyHead(3, hidden_channels=4, epsilon=0.01)
    output = head(torch.zeros(2, 3, 4, 5))

    assert output.scale_b.shape == (2, 1, 4, 5)
    assert bool((output.scale_b >= 0.01).all())
    with pytest.raises(ValueError, match="channels"):
        head(torch.zeros(1, 2, 4, 5))


def test_laplace_nll_rejects_nonfinite_valid_residual() -> None:
    predicted = torch.tensor([[[[float("nan")]]]])
    target = torch.zeros_like(predicted)
    scale = torch.ones_like(predicted)
    valid = torch.ones_like(predicted, dtype=torch.bool)

    with pytest.raises(ValueError, match="finite where valid"):
        laplace_nll(predicted, target, scale, valid)


def test_laplace_nll_ignores_nonfinite_invalid_residual() -> None:
    predicted = torch.tensor([[[[1.0, float("nan")]]]])
    target = torch.zeros_like(predicted)
    scale = torch.ones_like(predicted)
    valid = torch.tensor([[[[True, False]]]])

    assert torch.equal(laplace_nll(predicted, target, scale, valid), torch.tensor(1.0))


def test_laplace_nll_penalizes_under_and_overconfidence() -> None:
    predicted = torch.zeros(1, 1, 1, 1)
    target = torch.full_like(predicted, 2.0)
    valid = torch.ones_like(predicted, dtype=torch.bool)

    too_small = laplace_nll(predicted, target, torch.full_like(predicted, 0.5), valid)
    reasonable = laplace_nll(predicted, target, torch.full_like(predicted, 2.0), valid)
    too_large = laplace_nll(predicted, target, torch.full_like(predicted, 8.0), valid)

    assert reasonable < too_small
    assert reasonable < too_large


def test_laplace_nll_handles_zero_and_large_finite_residuals() -> None:
    predicted = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
    target = torch.tensor([[[[0.0, 1.0e100]]]], dtype=torch.float64)
    scale = torch.tensor([[[[1.0, 1.0e100]]]], dtype=torch.float64)
    valid = torch.ones_like(predicted, dtype=torch.bool)

    loss = laplace_nll(predicted, target, scale, valid)

    assert bool(torch.isfinite(loss))


def test_laplace_nll_rejects_empty_invalid_scale_shape_and_mask() -> None:
    predicted = torch.zeros(1, 1, 1, 1)
    target = torch.zeros_like(predicted)
    valid = torch.ones_like(predicted, dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one valid"):
        laplace_nll(predicted, target, torch.ones_like(predicted), ~valid)
    for candidate in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and strictly positive"):
            laplace_nll(predicted, target, torch.full_like(predicted, candidate), valid)
    with pytest.raises(ValueError, match="share shape"):
        laplace_nll(predicted, target[..., :0], torch.ones_like(predicted), valid)
    with pytest.raises(TypeError, match="torch.bool"):
        laplace_nll(predicted, target, torch.ones_like(predicted), valid.to(torch.float32))


def test_positive_scale_and_head_reject_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        positive_laplace_scale(torch.tensor([float("inf")]), epsilon=1e-4)
    head = LaplaceUncertaintyHead(1)
    with pytest.raises(ValueError, match="finite"):
        head(torch.tensor([[[[float("nan")]]]]))
