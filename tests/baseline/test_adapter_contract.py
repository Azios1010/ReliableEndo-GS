"""CPU-safe contract translations for actual upstream dictionary fields."""

from collections.abc import Mapping

import pytest
import torch

from reliable_endo_gs.baseline.adapter import BaselineAdapterError, EndoE2EGSAdapter
from reliable_endo_gs.contracts import CameraBatch, ReconstructionState, StereoBatch


def _left(upstream_output: Mapping[str, object]) -> Mapping[str, object]:
    value = upstream_output["lmain"]
    assert isinstance(value, Mapping)
    return value


def test_stereo_translation_preserves_disparity_dtype_device_and_values(
    upstream_output: Mapping[str, object],
) -> None:
    left = _left(upstream_output)
    disparity = left["flow_pred"]
    mask = left["mask"]
    assert isinstance(disparity, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

    prediction = EndoE2EGSAdapter.stereo_prediction(upstream_output)

    assert prediction.disparity is disparity
    assert prediction.disparity.dtype == disparity.dtype
    assert prediction.disparity.device == disparity.device
    assert torch.equal(prediction.valid_mask, mask >= 0.5)
    assert prediction.disparity_iterations is None
    assert prediction.sigma_d is None


def test_stereo_translation_accepts_directly_exposed_training_iterations(
    upstream_output: Mapping[str, object],
) -> None:
    left = _left(upstream_output)
    disparity = left["flow_pred"]
    assert isinstance(disparity, torch.Tensor)
    iterations = (disparity - 1.0, disparity)

    prediction = EndoE2EGSAdapter.stereo_prediction(
        upstream_output, disparity_iterations=iterations
    )

    assert isinstance(prediction.disparity_iterations, tuple)
    assert prediction.disparity_iterations[0] is iterations[0]


def test_gaussian_translation_matches_upstream_renderer_layout(
    upstream_output: Mapping[str, object],
) -> None:
    left = _left(upstream_output)
    image = left["img"]
    xyz = left["xyz"]
    assert isinstance(image, torch.Tensor)
    assert isinstance(xyz, torch.Tensor)

    field = EndoE2EGSAdapter.gaussian_field(upstream_output)

    expected_colors = (image.permute(0, 2, 3, 1).reshape(2, 12, 3) * 0.5) + 0.5
    assert field.means3d is xyz
    assert torch.equal(field.colors, expected_colors)
    assert field.rotations.shape == (2, 12, 4)
    assert field.scales.shape == (2, 12, 3)
    assert field.opacities.shape == (2, 12, 1)
    assert field.valid_mask.shape == (2, 12)
    assert field.colors.dtype == image.dtype
    assert field.colors.device == image.device
    assert field.cov_surface is None
    assert field.cov_center is None
    assert field.cov_effective is None


def test_render_translation_exposes_only_upstream_rgb(
    upstream_output: Mapping[str, object],
) -> None:
    image = _left(upstream_output)["img_pred"]
    assert isinstance(image, torch.Tensor)

    rendered = EndoE2EGSAdapter.render_output(upstream_output)

    assert rendered.image is image
    assert rendered.depth is None
    assert rendered.visibility is None


@pytest.mark.parametrize(
    ("method", "value", "message"),
    [
        (EndoE2EGSAdapter.stereo_prediction, {}, "missing required field 'lmain'"),
        (
            EndoE2EGSAdapter.stereo_prediction,
            {"lmain": {"flow_pred": "not-a-tensor", "mask": torch.ones(1)}},
            "flow_pred must be a torch.Tensor",
        ),
        (
            EndoE2EGSAdapter.render_output,
            {"lmain": {}},
            "missing required tensor 'img_pred'",
        ),
    ],
)
def test_invalid_upstream_structures_fail_clearly(
    method: object, value: Mapping[str, object], message: str
) -> None:
    assert callable(method)
    with pytest.raises(BaselineAdapterError, match=message):
        method(value)


def test_gaussian_translation_refuses_ambiguous_validity(
    upstream_output: Mapping[str, object],
) -> None:
    left = dict(_left(upstream_output))
    left["pts_valid"] = torch.ones(2, 12)
    with pytest.raises(BaselineAdapterError, match="must be boolean"):
        EndoE2EGSAdapter.gaussian_field({"lmain": left})


def test_composite_translation_returns_only_project_contracts(
    upstream_output: Mapping[str, object],
) -> None:
    cameras = CameraBatch(
        intrinsics=torch.eye(3).repeat(2, 1, 1),
        world_from_camera=torch.eye(4).repeat(2, 1, 1),
    )
    batch = StereoBatch(
        left=torch.zeros(2, 3, 3, 4),
        right=torch.zeros(2, 3, 3, 4),
        left_camera=cameras,
        right_camera=cameras,
        sample_ids=("synthetic-0", "synthetic-1"),
        sequence_ids=("synthetic", "synthetic"),
    )

    state = EndoE2EGSAdapter.reconstruction_state(
        batch,
        upstream_output,
        provenance={"upstream_commit": "186fa2b4a2159b28393492f6df1aa444b54391a8"},
    )

    assert isinstance(state, ReconstructionState)
    assert state.batch is batch
    assert state.render_right is None
    assert state.stereo.sigma_d is None
    assert state.gaussians.cov_center is None
