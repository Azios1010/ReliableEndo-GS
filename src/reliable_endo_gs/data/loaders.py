"""Lazy decoding of indexed SCARED-C samples into project contracts."""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from reliable_endo_gs.contracts.cameras import CameraBatch
from reliable_endo_gs.contracts.samples import StereoBatch
from reliable_endo_gs.data.calibration import (
    CalibrationError,
    adjust_intrinsics_for_resize_and_crop,
    parse_scared_c_stereo_calibration,
)
from reliable_endo_gs.data.index import ScaredCRecord
from reliable_endo_gs.data.tensors import rgb_array_to_tensor, xyz_array_to_tensor, xyz_valid_mask


class DataDecodeError(ValueError):
    """Raised when an indexed external member cannot be decoded safely."""


def _read_rgb(path: Path, *, dtype: torch.dtype) -> torch.Tensor:
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            array = np.array(image, copy=True)
    except (ImportError, OSError, ValueError) as error:
        raise DataDecodeError(f"unable to decode RGB image ({type(error).__name__})") from error
    try:
        return rgb_array_to_tensor(array, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise DataDecodeError(f"invalid RGB image: {error}") from error


def _read_xyz(path: Path, *, dtype: torch.dtype) -> torch.Tensor:
    try:
        import importlib

        tifffile: Any = importlib.import_module("tifffile")
        array = tifffile.imread(path)
    except (ImportError, OSError, ValueError) as error:
        raise DataDecodeError(f"unable to decode XYZ TIFF ({type(error).__name__})") from error
    try:
        return xyz_array_to_tensor(array, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise DataDecodeError(f"invalid XYZ TIFF: {error}") from error


def _resize_and_crop(
    image: torch.Tensor,
    xyz: torch.Tensor,
    *,
    original_size: tuple[int, int],
    target_size: tuple[int, int] | None,
    crop: tuple[int, int, int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Apply an explicit ``(x, y, width, height)`` crop and optional resize."""

    crop_x, crop_y = 0, 0
    crop_width, crop_height = original_size
    if crop is not None:
        crop_x, crop_y, crop_width, crop_height = crop
        if min(crop_x, crop_y, crop_width, crop_height) < 0:
            raise ValueError("crop coordinates and size must be non-negative")
        if crop_width == 0 or crop_height == 0:
            raise ValueError("crop size must be positive")
        original_width, original_height = original_size
        if crop_x + crop_width > original_width or crop_y + crop_height > original_height:
            raise ValueError("crop must be contained within the original image")
        image = image[:, crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
        xyz = xyz[:, crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]

    if target_size is None:
        return image, xyz, (crop_width, crop_height)
    target_width, target_height = target_size
    if min(target_width, target_height) <= 0:
        raise ValueError("target image size must be positive")
    image = functional.interpolate(
        image.unsqueeze(0), size=(target_height, target_width), mode="bilinear", align_corners=False
    ).squeeze(0)
    xyz = functional.interpolate(
        xyz.unsqueeze(0), size=(target_height, target_width), mode="nearest"
    ).squeeze(0)
    return image, xyz, (crop_width, crop_height)


def _camera(
    intrinsics: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> CameraBatch:
    """Create a camera-local contract without fabricating an external pose."""

    matrix = intrinsics.to(dtype=dtype).unsqueeze(0)
    identity = torch.eye(4, dtype=dtype).unsqueeze(0)
    return CameraBatch(
        intrinsics=matrix,
        world_from_camera=identity,
    )


def load_scared_c_sample(
    record: ScaredCRecord,
    *,
    dtype: torch.dtype = torch.float32,
    target_size: tuple[int, int] | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> StereoBatch:
    """Decode one indexed SCARED-C reference keyframe.

    ``target_size`` is ``(width, height)``.  Images use bilinear interpolation;
    XYZ maps use nearest-neighbour interpolation.  The latter is a data
    decoding choice, not a geometry conversion.  The source coordinate-map
    units and axes are preserved as ``UNRESOLVED`` metadata.
    """

    if not isinstance(record, ScaredCRecord):
        raise TypeError("record must be a ScaredCRecord returned by build_scared_c_index")
    left = _read_rgb(record.left_image_path, dtype=dtype)
    right = _read_rgb(record.right_image_path, dtype=dtype)
    left_xyz = _read_xyz(record.left_depth_path, dtype=dtype)
    right_xyz = _read_xyz(record.right_depth_path, dtype=dtype)
    if left.shape != right.shape:
        raise DataDecodeError(
            f"left/right image tensor shapes do not match: {left.shape} vs {right.shape}"
        )
    if left_xyz.shape != right_xyz.shape or tuple(left_xyz.shape[1:]) != tuple(left.shape[1:]):
        raise DataDecodeError("stereo image and XYZ tensor shapes do not match")

    height, width = int(left.shape[1]), int(left.shape[2])
    original_size = (width, height)
    left, left_xyz, source_size = _resize_and_crop(
        left, left_xyz, original_size=original_size, target_size=target_size, crop=crop
    )
    right, right_xyz, right_source_size = _resize_and_crop(
        right, right_xyz, original_size=original_size, target_size=target_size, crop=crop
    )
    if source_size != right_source_size:
        raise DataDecodeError("left/right crop sizes do not match")
    output_size = (int(left.shape[2]), int(left.shape[1]))

    try:
        calibration = parse_scared_c_stereo_calibration(
            record.endoscope_calibration_path,
            image_size=original_size,
            protocol=record.protocol,
        )
    except CalibrationError as error:
        raise DataDecodeError(f"invalid SCARED-C calibration: {error}") from error
    left_k = calibration.left_intrinsics
    right_k = calibration.right_intrinsics
    if crop is not None or target_size is not None:
        crop_offset = (crop[0], crop[1]) if crop is not None else (0, 0)
        left_k = adjust_intrinsics_for_resize_and_crop(
            left_k,
            original_size,
            output_size,
            crop_offset,
            crop_size=source_size,
        )
        right_k = adjust_intrinsics_for_resize_and_crop(
            right_k,
            original_size,
            output_size,
            crop_offset,
            crop_size=source_size,
        )
    left_camera = _camera(
        left_k,
        dtype=dtype,
    )
    right_camera = _camera(
        right_k,
        dtype=dtype,
    )

    left_xyz = left_xyz.unsqueeze(0)
    right_xyz = right_xyz.unsqueeze(0)
    return StereoBatch(
        left=left.unsqueeze(0),
        right=right.unsqueeze(0),
        left_camera=left_camera,
        right_camera=right_camera,
        sample_ids=(record.sample_id,),
        sequence_ids=(record.sequence_id,),
        gt_depth_xyz=left_xyz,
        gt_right_depth_xyz=right_xyz,
        masks={
            "left_depth_valid": xyz_valid_mask(left_xyz[0]).unsqueeze(0),
            "right_depth_valid": xyz_valid_mask(right_xyz[0]).unsqueeze(0),
        },
        metadata={
            "dataset_id": record.dataset_id,
            "protocol": record.protocol,
            "sequence_id": record.sequence_id,
            "keyframe_id": record.keyframe_id,
            "frame_id": record.frame_id,
            "calibration_source": "endoscope_calibration.yaml",
            "intrinsics_source": "endoscope_calibration.yaml:M1,M2",
            "pose_source": "frame_data.tar.gz:camera-pose (not applied; direction UNRESOLVED)",
            "pose_applied": False,
            "stereo_extrinsics_applied": False,
            "world_transform_semantics": "camera-local identity only; external world frame UNRESOLVED",
            "camera_convention": "UNRESOLVED",
            "camera_axes": "UNRESOLVED",
            "stereo_extrinsics": "present in endoscope_calibration.yaml; interpretation UNRESOLVED",
            "depth_source": "left_depth_map.tiff/right_depth_map.tiff",
            "depth_representation": "per-pixel XYZ coordinate map",
            "depth_units": "UNRESOLVED",
            "invalid_depth": "non-finite XYZ values; all three channels must be finite",
            "rectification": "UNRESOLVED",
            "pixel_center": "UNRESOLVED",
            "image_color_space": "RGB; source alpha discarded",
            "image_range": "float32 [0, 1]",
            "index_members": record.relative_members,
            "output_size_wh": output_size,
        },
    )
