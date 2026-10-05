"""Regression tests for A03 image-border neighbourhood semantics."""

import pytest
import torch

from reliable_endo_gs.geometry.surface import _shifted_neighbour


@pytest.mark.parametrize("dy", [-1, 0, 1])
@pytest.mark.parametrize("dx", [-1, 0, 1])
def test_shifted_neighbour_masks_every_border_without_wraparound(dy: int, dx: int) -> None:
    height, width = 3, 4
    values = torch.arange(height * width, dtype=torch.float32).view(1, height, width, 1)
    valid = torch.ones((1, height, width), dtype=torch.bool)
    shifted, mask = _shifted_neighbour(values, valid, dy, dx)

    expected_mask = torch.zeros((height, width), dtype=torch.bool)
    expected_values = torch.zeros((height, width), dtype=torch.float32)
    for y in range(height):
        for x in range(width):
            source_y, source_x = y + dy, x + dx
            if 0 <= source_y < height and 0 <= source_x < width:
                expected_mask[y, x] = True
                expected_values[y, x] = values[0, source_y, source_x, 0]

    assert torch.equal(mask[0], expected_mask)
    # Values outside support may be rolled, but they must never be admitted by
    # the mask; supported values must always come from the in-bounds source.
    torch.testing.assert_close(shifted[0, ..., 0][expected_mask], expected_values[expected_mask])


def test_shifted_neighbour_rejects_wrapped_corner_values() -> None:
    values = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    valid = torch.ones((1, 2, 2), dtype=torch.bool)
    shifted, mask = _shifted_neighbour(values, valid, 1, 1)
    assert mask.tolist() == [[[True, False], [False, False]]]
    assert shifted[0, 0, 0, 0].item() == 4.0
    assert not bool(mask[0, 1, 1])
