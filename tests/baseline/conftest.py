"""Synthetic, non-scientific upstream-shaped fixtures for adapter tests."""

from collections.abc import Mapping

import pytest
import torch


@pytest.fixture
def upstream_output() -> Mapping[str, object]:
    batch, height, width = 2, 3, 4
    count = height * width
    return {
        "lmain": {
            "img": torch.linspace(-1.0, 1.0, batch * 3 * height * width).reshape(
                batch, 3, height, width
            ),
            "mask": torch.tensor(
                [0.0, 0.49, 0.5, 1.0] * (batch * height), dtype=torch.float32
            ).reshape(batch, 1, height, width),
            "flow_pred": torch.arange(batch * height * width, dtype=torch.float32).reshape(
                batch, 1, height, width
            ),
            "xyz": torch.arange(batch * count * 3, dtype=torch.float32).reshape(batch, count, 3),
            "pts_valid": torch.ones(batch, count, dtype=torch.bool),
            "rot_maps": torch.nn.functional.normalize(torch.ones(batch, 4, height, width), dim=1),
            "scale_maps": torch.full((batch, 3, height, width), 0.01),
            "opacity_maps": torch.full((batch, 1, height, width), 0.75),
            "img_pred": torch.full((batch, 3, height, width), 0.25),
        }
    }
