"""RenderOutput structural contract tests."""

import pytest
import torch

from reliable_endo_gs.contracts import RenderOutput


def test_render_output_accepts_minimal_and_optional_forms() -> None:
    image = torch.zeros(2, 3, 4, 5)
    minimal = RenderOutput(image=image)
    full = RenderOutput(
        image=image,
        depth=torch.ones(2, 1, 4, 5),
        visibility=torch.ones(2, 6),
    )

    assert minimal.image is image
    assert minimal.depth is None
    assert full.batch_size == 2
    assert full.spatial_shape == (4, 5)


def test_render_output_rejects_invalid_image_and_depth() -> None:
    with pytest.raises(ValueError, match="image must have rank 4"):
        RenderOutput(image=torch.zeros(2, 4, 5))
    with pytest.raises(TypeError, match="image must have a floating-point dtype"):
        RenderOutput(image=torch.zeros(2, 3, 4, 5, dtype=torch.int64))
    with pytest.raises(ValueError, match="depth must have shape"):
        RenderOutput(image=torch.zeros(2, 3, 4, 5), depth=torch.zeros(2, 1, 3, 5))
    with pytest.raises(TypeError, match="depth must have the same dtype"):
        RenderOutput(
            image=torch.zeros(2, 3, 4, 5),
            depth=torch.zeros(2, 1, 4, 5, dtype=torch.float64),
        )


def test_render_output_rejects_visibility_batch_and_device_mismatch() -> None:
    with pytest.raises(ValueError, match="visibility batch dimension"):
        RenderOutput(image=torch.zeros(2, 3, 4, 5), visibility=torch.zeros(1, 6))
    with pytest.raises(ValueError, match="visibility must be on the same device"):
        RenderOutput(
            image=torch.zeros(2, 3, 4, 5),
            visibility=torch.zeros(2, 6, device="meta"),
        )
