"""Plan 08 robust masked photometric loss tests."""

import pytest
import torch

from reliable_endo_gs.rendering.cross_view import (
    RobustLossConfig,
    aggregate_view_losses,
    cross_view_loss,
)


def test_l1_averages_valid_pixels_and_channels() -> None:
    predicted = torch.zeros((1, 3, 1, 2), dtype=torch.float64)
    target = torch.ones_like(predicted)
    mask = torch.tensor([[[[True, False]]]])
    result = cross_view_loss(predicted, target, mask, RobustLossConfig(kind="l1"))
    assert result.loss is not None
    assert result.loss.item() == pytest.approx(1.0)
    assert result.valid_count == 1
    assert result.element_count == 3


def test_charbonnier_known_value_and_nonfinite_masking() -> None:
    predicted = torch.zeros((1, 1, 1, 2), dtype=torch.float64)
    target = torch.tensor([[[[1.0, float("nan")]]]], dtype=torch.float64)
    mask = torch.ones((1, 1, 1, 2), dtype=torch.bool)
    result = cross_view_loss(
        predicted,
        target,
        mask,
        RobustLossConfig(kind="charbonnier", charbonnier_epsilon=0.5),
    )
    assert result.loss is not None
    assert result.loss.item() == pytest.approx((1.0 + 0.25) ** 0.5)
    assert result.nonfinite_count == 1


def test_empty_mask_is_unavailable_and_view_weights_are_explicit() -> None:
    image = torch.zeros((1, 1, 1, 1), dtype=torch.float64)
    empty = torch.zeros((1, 1, 1, 1), dtype=torch.bool)
    unavailable = cross_view_loss(image, image, empty, RobustLossConfig(kind="l1"))
    available = cross_view_loss(image, torch.ones_like(image), ~empty, RobustLossConfig(kind="l1"))
    result = aggregate_view_losses(unavailable, available, left_weight=3.0, right_weight=2.0)
    assert unavailable.loss is None
    assert result.total is not None
    assert result.total.item() == pytest.approx(2.0)


def test_loss_rejects_shape_and_mask_contract_errors() -> None:
    image = torch.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError, match="shape"):
        cross_view_loss(image, image, torch.ones((1, 1, 2, 1), dtype=torch.bool))
    with pytest.raises(TypeError, match="dtype torch.bool"):
        cross_view_loss(image, image, torch.ones((1, 1, 2, 2)))
