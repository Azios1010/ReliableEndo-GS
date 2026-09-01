"""Centralized calibration parsing and analytic image-coordinate transforms.

This module parses the OpenCV YAML representation used by the inspected
SCARED-C sample.  It does not infer pose direction, camera axes, units, or
rectification from numeric values; those semantics remain explicit metadata
owned by the selected data protocol.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import torch
import yaml
from yaml.nodes import MappingNode

CalibrationValue: TypeAlias = torch.Tensor | int | float | str | None


class CalibrationError(ValueError):
    """Raised when a calibration file cannot satisfy the expected schema."""


class _OpenCVLoader(yaml.SafeLoader):
    """Local YAML loader so importing this module has no global side effects."""


def _construct_opencv_matrix(loader: yaml.Loader, node: yaml.Node) -> torch.Tensor:
    raw = loader.construct_mapping(cast(MappingNode, node), deep=True)
    if not isinstance(raw, dict):
        raise CalibrationError("OpenCV matrix node must be a mapping")
    rows = raw.get("rows")
    cols = raw.get("cols")
    values = raw.get("data")
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or isinstance(cols, bool)
        or not isinstance(cols, int)
        or not isinstance(values, list)
        or rows <= 0
        or cols <= 0
        or len(values) != rows * cols
    ):
        raise CalibrationError("OpenCV matrix must provide valid rows, cols, and data")
    dt = raw.get("dt", "d")
    dtype = (
        torch.float32 if isinstance(dt, str) and dt.casefold().startswith("f") else torch.float64
    )
    try:
        return torch.tensor(values, dtype=dtype).reshape(rows, cols)
    except (TypeError, ValueError, RuntimeError) as error:
        raise CalibrationError(f"OpenCV matrix data is not numeric: {error}") from error


_OpenCVLoader.add_constructor("tag:yaml.org,2002:opencv-matrix", _construct_opencv_matrix)


def parse_opencv_calibration(path: Path) -> dict[str, CalibrationValue]:
    """Parse an OpenCV YAML file without changing the process-wide YAML loader.

    Matrix values retain the precision advertised by ``dt`` (``f`` or ``d``).
    Scalar metadata such as image dimensions remains a Python scalar.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CalibrationError(
            f"unable to read calibration file ({type(error).__name__})"
        ) from error

    lines = text.splitlines()
    if lines and lines[0].strip().startswith("%YAML:"):
        lines = lines[1:]
    try:
        decoded: object = yaml.load("\n".join(lines), Loader=_OpenCVLoader)
    except yaml.YAMLError as error:
        raise CalibrationError(f"invalid OpenCV YAML: {error}") from error
    if not isinstance(decoded, dict):
        raise CalibrationError("calibration file must contain a mapping")

    result: dict[str, CalibrationValue] = {}
    for key, value in decoded.items():
        if not isinstance(key, str):
            raise CalibrationError("calibration keys must be strings")
        if value is None or isinstance(value, (str, int, float, torch.Tensor)):
            result[key] = value
        else:
            raise CalibrationError(f"unsupported calibration value for {key!r}")
    return result


def _required_matrix(
    values: dict[str, CalibrationValue], name: str, shape: tuple[int, ...]
) -> torch.Tensor:
    value = values.get(name)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        observed = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
        raise CalibrationError(
            f"calibration field {name!r} must have shape {shape}; got {observed}"
        )
    if not torch.is_floating_point(value):
        raise CalibrationError(f"calibration field {name!r} must be floating point")
    return value


def _optional_matrix(
    values: dict[str, CalibrationValue], name: str, shapes: tuple[tuple[int, ...], ...]
) -> torch.Tensor | None:
    value = values.get(name)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor) or tuple(value.shape) not in shapes:
        observed = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
        expected = " or ".join(str(shape) for shape in shapes)
        raise CalibrationError(
            f"calibration field {name!r} must have shape {expected}; got {observed}"
        )
    return value


@dataclass(frozen=True, slots=True)
class StereoCalibration:
    """Verifiedly parsed stereo calibration with unresolved semantic labels."""

    left_intrinsics: torch.Tensor
    right_intrinsics: torch.Tensor
    left_distortion: torch.Tensor | None
    right_distortion: torch.Tensor | None
    rotation: torch.Tensor | None
    translation: torch.Tensor | None
    image_size: tuple[int, int]
    intrinsics_source: str
    protocol: str
    semantics: dict[str, str]


def parse_scared_c_stereo_calibration(
    path: Path,
    *,
    image_size: tuple[int, int],
    protocol: str,
) -> StereoCalibration:
    """Parse the endoscope stereo calibration selected by the SCARED-C protocol.

    ``M1``/``M2`` are kept together as one calibration regime.  The separate
    COLMAP ``K`` file is intentionally not consulted here because the inspected
    sample provides only one COLMAP intrinsic matrix.
    """

    values = parse_opencv_calibration(path)
    return StereoCalibration(
        left_intrinsics=_required_matrix(values, "M1", (3, 3)),
        right_intrinsics=_required_matrix(values, "M2", (3, 3)),
        left_distortion=_optional_matrix(values, "D1", ((1, 5), (5,))),
        right_distortion=_optional_matrix(values, "D2", ((1, 5), (5,))),
        rotation=_optional_matrix(values, "R", ((3, 3),)),
        translation=_optional_matrix(values, "T", ((1, 3), (3,))),
        image_size=image_size,
        intrinsics_source="endoscope_calibration.yaml:M1,M2",
        protocol=protocol,
        semantics={
            "pose_direction": "UNRESOLVED",
            "camera_axes": "UNRESOLVED",
            "rectification": "UNRESOLVED",
            "units": "UNRESOLVED",
            "pixel_center": "UNRESOLVED",
            "stereo_extrinsics_application": "UNRESOLVED",
        },
    )


def _validate_intrinsics(intrinsics: torch.Tensor) -> None:
    if intrinsics.ndim < 2 or tuple(intrinsics.shape[-2:]) != (3, 3):
        raise ValueError(f"intrinsics must end with shape (3, 3); got {tuple(intrinsics.shape)}")
    if not torch.is_floating_point(intrinsics):
        raise TypeError(f"intrinsics must be floating point; got {intrinsics.dtype}")


def resize_intrinsics(
    intrinsics: torch.Tensor,
    *,
    original_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Apply analytic resize ``(width, height) -> (width, height)`` to ``K``."""

    _validate_intrinsics(intrinsics)
    original_width, original_height = original_size
    target_width, target_height = target_size
    if min(original_width, original_height, target_width, target_height) <= 0:
        raise ValueError("image sizes must be positive")
    sx = target_width / original_width
    sy = target_height / original_height
    result = intrinsics.clone()
    result[..., 0, 0] *= sx
    result[..., 1, 1] *= sy
    result[..., 0, 2] *= sx
    result[..., 1, 2] *= sy
    return result


def crop_intrinsics(intrinsics: torch.Tensor, *, crop_offset: tuple[int, int]) -> torch.Tensor:
    """Apply a pixel crop whose top-left offset is ``(x, y)`` to ``K``."""

    _validate_intrinsics(intrinsics)
    crop_x, crop_y = crop_offset
    if crop_x < 0 or crop_y < 0:
        raise ValueError("crop offsets must be non-negative")
    result = intrinsics.clone()
    result[..., 0, 2] -= crop_x
    result[..., 1, 2] -= crop_y
    return result


def adjust_intrinsics_for_resize_and_crop(
    intrinsics: torch.Tensor,
    original_size: tuple[int, int],
    target_size: tuple[int, int],
    crop_offset: tuple[int, int] = (0, 0),
    *,
    crop_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Apply crop then resize, using one explicit pixel-coordinate convention.

    Sizes are ``(width, height)``.  If ``crop_size`` is omitted, the resize
    scale is relative to ``original_size``; this supports a crop offset-only
    transform and avoids inventing a crop extent.  Callers that resize a
    finite crop pass its exact ``crop_size``.
    """

    cropped = crop_intrinsics(intrinsics, crop_offset=crop_offset)
    resize_source = original_size if crop_size is None else crop_size
    if crop_size is not None and min(crop_size) <= 0:
        raise ValueError("crop_size must be positive")
    return resize_intrinsics(cropped, original_size=resize_source, target_size=target_size)
