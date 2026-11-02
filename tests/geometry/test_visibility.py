"""Canonical right-view strip width and the visibility-masked support (geometry/visibility.py)."""

from __future__ import annotations

import pytest
import torch

from reliable_endo_gs.geometry.visibility import BORDER_COLUMNS, right_visible_support, strip_width


def _disparity(border_value: float, height: int = 4, width: int = 40) -> torch.Tensor:
    disparity = torch.full((1, 1, height, width), 5.0)
    disparity[..., -BORDER_COLUMNS:] = border_value
    return disparity


def test_strip_width_is_the_rounded_up_border_quantile() -> None:
    assert strip_width(_disparity(7.2)) == 8
    assert strip_width(_disparity(9.0)) == 9


def test_strip_width_requires_finite_border_values_and_the_batch_one_shape() -> None:
    with pytest.raises(ValueError):
        strip_width(_disparity(float("nan")))
    with pytest.raises(ValueError):
        strip_width(torch.ones(2, 1, 4, 40))


def test_visible_support_removes_exactly_the_strip_and_keeps_the_rest_and_the_input() -> None:
    support = torch.ones(1, 1, 4, 40, dtype=torch.bool)
    support[..., 0] = False
    visible = right_visible_support(support, _disparity(7.2))
    assert not visible[..., 40 - 8 :].any()
    assert torch.equal(visible[..., : 40 - 8], support[..., : 40 - 8])
    assert bool(support[..., -1].all()) and visible is not support


def test_visible_support_reads_the_disparity_without_gradient() -> None:
    disparity = _disparity(7.2).requires_grad_(True)
    visible = right_visible_support(torch.ones(1, 1, 4, 40, dtype=torch.bool), disparity)
    assert visible.dtype is torch.bool and not visible.requires_grad


def test_visible_support_rejects_a_strip_as_wide_as_the_image() -> None:
    with pytest.raises(ValueError):
        right_visible_support(torch.ones(1, 1, 4, 10, dtype=torch.bool), _disparity(30.0, width=10))
