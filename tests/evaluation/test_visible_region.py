"""Strip width and column crop of the visible-region metric (CPU, synthetic)."""

from __future__ import annotations

import pytest
import torch

from reliable_endo_gs.evaluation.visible_region import BORDER_COLUMNS, crop_columns, strip_width


def test_strip_width_uses_the_border_columns_only_and_rounds_up():
    disparity = torch.full((1, 1, 6, 40), 99.0)
    disparity[..., -BORDER_COLUMNS:] = 12.2
    assert strip_width(disparity) == 13  # interior values do not matter
    disparity[..., -BORDER_COLUMNS:] = torch.linspace(10.0, 20.0, BORDER_COLUMNS)
    assert 10 < strip_width(disparity) <= 20


def test_strip_width_ignores_non_finite_values_and_rejects_bad_input():
    disparity = torch.full((1, 1, 4, 30), 8.0)
    disparity[..., -3:] = float("nan")
    assert strip_width(disparity) == 8
    disparity[..., -BORDER_COLUMNS:] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        strip_width(disparity)
    with pytest.raises(ValueError, match="shape"):
        strip_width(torch.zeros(2, 1, 4, 30))


def test_crop_columns_removes_the_right_strip_only():
    image = torch.arange(2 * 3 * 4 * 10, dtype=torch.float32).reshape(2, 3, 4, 10)
    cropped = crop_columns(image, 3)
    assert cropped.shape == (2, 3, 4, 7) and torch.equal(cropped, image[..., :7])
    assert torch.equal(crop_columns(image, 0), image)
    for bad in (-1, 10, 11):
        with pytest.raises(ValueError, match="strip"):
            crop_columns(image, bad)
