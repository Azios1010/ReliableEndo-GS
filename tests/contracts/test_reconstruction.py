"""ReconstructionState structural and boundary tests."""

import pytest
import torch

from reliable_endo_gs.contracts import (
    GaussianField,
    ReconstructionState,
    RenderOutput,
    StereoBatch,
    StereoPrediction,
)


def _state(
    stereo_batch: StereoBatch,
    stereo_prediction: StereoPrediction,
    gaussian_field: GaussianField,
    render_output: RenderOutput,
    **overrides: object,
) -> ReconstructionState:
    values: dict[str, object] = {
        "batch": stereo_batch,
        "stereo": stereo_prediction,
        "gaussians": gaussian_field,
        "render_left": render_output,
    }
    values.update(overrides)
    return ReconstructionState(**values)  # type: ignore[arg-type]


def test_valid_reconstruction_state_is_phase1_only(
    reconstruction_state: ReconstructionState,
) -> None:
    assert reconstruction_state.render_right is None
    assert not hasattr(reconstruction_state, "router")
    assert not hasattr(reconstruction_state, "actions")
    assert not hasattr(reconstruction_state, "budget")
    with pytest.raises(TypeError):
        reconstruction_state.diagnostics["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        reconstruction_state.provenance["new"] = 1  # type: ignore[index]


def test_reconstruction_rejects_component_batch_and_spatial_mismatch(
    stereo_batch: StereoBatch,
    stereo_prediction: StereoPrediction,
    gaussian_field: GaussianField,
    render_output: RenderOutput,
) -> None:
    wrong_prediction = StereoPrediction(
        disparity=torch.ones(1, 1, 4, 5),
        valid_mask=torch.ones(1, 1, 4, 5, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="stereo batch dimension"):
        _state(stereo_batch, wrong_prediction, gaussian_field, render_output)

    wrong_render = RenderOutput(image=torch.zeros(2, 3, 3, 5))
    with pytest.raises(ValueError, match="render_left spatial shape"):
        _state(stereo_batch, stereo_prediction, gaussian_field, wrong_render)


def test_reconstruction_rejects_device_mismatch(
    stereo_batch: StereoBatch,
    stereo_prediction: StereoPrediction,
    render_output: RenderOutput,
) -> None:
    meta_field = GaussianField(
        means3d=torch.zeros(2, 6, 3, device="meta"),
        colors=torch.zeros(2, 6, 3, device="meta"),
        rotations=torch.zeros(2, 6, 4, device="meta"),
        scales=torch.ones(2, 6, 3, device="meta"),
        opacities=torch.ones(2, 6, 1, device="meta"),
        valid_mask=torch.ones(2, 6, dtype=torch.bool, device="meta"),
    )
    with pytest.raises(ValueError, match="gaussians must be on the same device"):
        _state(stereo_batch, stereo_prediction, meta_field, render_output)


def test_reconstruction_accepts_optional_right_render(
    stereo_batch: StereoBatch,
    stereo_prediction: StereoPrediction,
    gaussian_field: GaussianField,
    render_output: RenderOutput,
) -> None:
    state = _state(
        stereo_batch,
        stereo_prediction,
        gaussian_field,
        render_output,
        render_right=render_output,
    )
    assert state.render_right is render_output
