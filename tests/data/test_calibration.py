"""Calibration parsing and analytic image-coordinate transforms."""

from pathlib import Path

import pytest
import torch

from reliable_endo_gs.data.calibration import (
    CalibrationError,
    adjust_intrinsics_for_resize_and_crop,
    crop_intrinsics,
    parse_opencv_calibration,
    parse_scared_c_stereo_calibration,
    resize_intrinsics,
)


def test_parse_opencv_calibration_preserves_declared_matrix_precision(tmp_path: Path) -> None:
    path = tmp_path / "calibration.yaml"
    path.write_text(
        """%YAML:1.0
---
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1000, 0, 500, 0, 1000, 500, 0, 0, 1 ]
""",
        encoding="utf-8",
    )

    parsed = parse_opencv_calibration(path)

    matrix = parsed["camera_matrix"]
    assert isinstance(matrix, torch.Tensor)
    assert matrix.shape == (3, 3)
    assert matrix.dtype == torch.float64
    assert matrix[0, 0] == 1000.0


def test_scared_c_parser_keeps_endoscope_stereo_regime(scared_c_root: Path) -> None:
    keyframe = scared_c_root / "dataset_1" / "keyframe_1"
    calibration = parse_scared_c_stereo_calibration(
        keyframe / "endoscope_calibration.yaml",
        image_size=(4, 3),
        protocol="scared_c_endoscope_stereo_calibration_v1",
    )

    assert calibration.left_intrinsics[0, 0] == 100.0
    assert calibration.right_intrinsics[0, 0] == 110.0
    assert calibration.intrinsics_source == "endoscope_calibration.yaml:M1,M2"
    assert calibration.semantics["units"] == "UNRESOLVED"


def test_missing_required_stereo_matrix_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "M1: !!opencv-matrix {rows: 3, cols: 3, dt: f, data: [1,2,3,4,5,6,7,8,9]}\n",
        encoding="utf-8",
    )

    with pytest.raises(CalibrationError, match="M2"):
        parse_scared_c_stereo_calibration(
            path,
            image_size=(4, 3),
            protocol="scared_c_endoscope_stereo_calibration_v1",
        )


def test_resize_intrinsics_uses_independent_axis_scales() -> None:
    intrinsics = torch.tensor([[1000.0, 0.0, 500.0], [0.0, 800.0, 400.0], [0.0, 0.0, 1.0]])

    resized = resize_intrinsics(intrinsics, original_size=(1000, 800), target_size=(500, 400))

    assert torch.equal(
        resized,
        torch.tensor([[500.0, 0.0, 250.0], [0.0, 400.0, 200.0], [0.0, 0.0, 1.0]]),
    )
    assert torch.equal(
        intrinsics, torch.tensor([[1000.0, 0.0, 500.0], [0.0, 800.0, 400.0], [0.0, 0.0, 1.0]])
    )


def test_crop_intrinsics_translates_principal_point_only() -> None:
    intrinsics = torch.tensor([[1000.0, 0.0, 500.0], [0.0, 800.0, 400.0], [0.0, 0.0, 1.0]])

    cropped = crop_intrinsics(intrinsics, crop_offset=(100, 50))

    assert cropped[0, 0] == 1000.0
    assert cropped[1, 1] == 800.0
    assert cropped[0, 2] == 400.0
    assert cropped[1, 2] == 350.0


def test_crop_then_resize_uses_explicit_crop_size() -> None:
    intrinsics = torch.tensor([[100.0, 0.0, 50.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]])

    adjusted = adjust_intrinsics_for_resize_and_crop(
        intrinsics,
        original_size=(100, 100),
        target_size=(200, 100),
        crop_offset=(10, 20),
        crop_size=(50, 50),
    )

    assert adjusted[0, 0] == 400.0
    assert adjusted[1, 1] == 400.0
    assert adjusted[0, 2] == 160.0
    assert adjusted[1, 2] == 80.0
