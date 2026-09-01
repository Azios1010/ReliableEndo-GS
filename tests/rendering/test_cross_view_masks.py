"""Plan 08 camera and mask algebra tests."""

import pytest
import torch

from reliable_endo_gs.contracts import CameraBatch
from reliable_endo_gs.rendering.cross_view import (
    CameraRequest,
    CameraView,
    MaskEvidence,
    build_cross_view_mask,
    make_camera_request,
)


def _camera(batch_size: int = 1) -> CameraBatch:
    intrinsics = torch.eye(3).repeat(batch_size, 1, 1)
    transform = torch.eye(4).repeat(batch_size, 1, 1)
    return CameraBatch(intrinsics, transform)


def test_left_and_right_camera_requests_are_explicit() -> None:
    camera = _camera()
    left = make_camera_request(camera, view=CameraView.LEFT, image_size=(2, 3))
    right = CameraRequest(camera, "right", (2, 3))
    assert left.view is CameraView.LEFT
    assert right.view is CameraView.RIGHT
    assert left.image_size == right.image_size


def test_mask_components_intersect_and_exclusions_are_reported() -> None:
    geometry = torch.ones((1, 1, 2, 3), dtype=torch.bool)
    stereo = geometry.clone()
    stereo[..., 0, 1] = False
    specular = torch.zeros_like(geometry)
    specular[..., 1, 2] = True
    result = build_cross_view_mask(
        geometry,
        MaskEvidence(stereo_valid=stereo, specularity=specular),
    )
    assert result.valid_count == 4
    assert not result.combined_mask[..., 0, 1]
    assert not result.combined_mask[..., 1, 2]
    assert result.diagnostics["optional_unavailable"]
    assert result.diagnostics["excluded_by_component"]["stereo_valid"] == 1  # type: ignore[index]


def test_missing_optional_masks_do_not_become_fake_evidence() -> None:
    geometry = torch.ones((1, 1, 1, 2), dtype=torch.bool)
    result = build_cross_view_mask(geometry)
    assert result.valid_count == 2
    assert "occlusion" in result.diagnostics["optional_unavailable"]  # type: ignore[operator]
    assert result.coverage == 1.0


def test_all_invalid_mask_and_shape_or_dtype_failures() -> None:
    geometry = torch.zeros((1, 1, 1, 2), dtype=torch.bool)
    result = build_cross_view_mask(geometry)
    assert result.valid_count == 0
    with pytest.raises(TypeError, match="dtype torch.bool"):
        build_cross_view_mask(torch.ones_like(geometry, dtype=torch.float32))
    with pytest.raises(ValueError, match="shape"):
        build_cross_view_mask(geometry, {"in_view": torch.ones((1, 1, 1, 1), dtype=torch.bool)})
