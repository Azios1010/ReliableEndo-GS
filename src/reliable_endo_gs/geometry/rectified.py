"""Raw rectified disparity adapter; canonical depth/backprojection are reused."""

from dataclasses import dataclass

import torch

from reliable_endo_gs.contracts.deployment import RectifiedStereoCamera
from reliable_endo_gs.geometry.backprojection import BackprojectionResult, backproject_depth
from reliable_endo_gs.geometry.conventions import GeometryConvention
from reliable_endo_gs.geometry.disparity import DepthResult, disparity_to_depth


@dataclass(frozen=True, slots=True)
class MetricStereoGeometry:
    """Effective disparity/depth [B,1,H,W], left-camera centres [B,3,H,W]."""

    effective_disparity: torch.Tensor
    depth: DepthResult
    points: BackprojectionResult


def reconstruct_rectified(
    raw_disparity: torch.Tensor,
    camera: RectifiedStereoCamera,
    support: torch.Tensor,
    *,
    min_effective_disparity: float,
) -> MetricStereoGeometry:
    """Convert d=u_L-u_R (current-grid pixels) to metric optical-axis geometry.

    Correct principal points ONCE here. FP16/BF16 predictions are promoted to
    calibration precision, retaining gradients. Invalid geometry remains masked
    NaN and must not be forwarded unfiltered to a renderer.
    """
    if raw_disparity.ndim != 4 or raw_disparity.shape[1] != 1:
        raise ValueError("raw disparity must be [B,1,H,W]")
    if not raw_disparity.is_floating_point() or raw_disparity.device != camera.K_L.device:
        raise ValueError("disparity must be floating point on the camera device")
    if raw_disparity.shape[0] != camera.K_L.shape[0]:
        raise ValueError("disparity camera batch mismatch")
    offset = (camera.K_L[:, 0, 2] - camera.K_R[:, 0, 2]).view(-1, 1, 1, 1)
    effective = raw_disparity.to(camera.K_L.dtype) - offset
    convention = GeometryConvention()
    depth = disparity_to_depth(
        effective,
        camera.K_L[:, 0, 0],
        camera.baseline,
        support,
        min_abs_disparity=min_effective_disparity,
        convention=convention,
    )
    points = backproject_depth(depth.depth, camera.K_L, depth.valid_mask, convention=convention)
    return MetricStereoGeometry(effective, depth, points)


def crop_resize_camera(
    camera: RectifiedStereoCamera,
    *,
    left_crop_xy: tuple[int, int],
    right_crop_xy: tuple[int, int],
    crop_hw: tuple[int, int],
    output_hw: tuple[int, int],
) -> RectifiedStereoCamera:
    """Calibration for paired equal-size crops and align_corners=False resize.

    Pixel centres transform as u' = sx*(u-crop_x+0.5)-0.5.
    Image resampling is caller-owned; unequal vertical crops are rejected.
    """
    if min(*crop_hw, *output_hw) <= 0 or min(*left_crop_xy, *right_crop_xy) < 0:
        raise ValueError("positive sizes and nonnegative crop offsets required")
    sy, sx = output_hw[0] / crop_hw[0], output_hw[1] / crop_hw[1]
    matrices = []
    for k, (x, y) in ((camera.K_L, left_crop_xy), (camera.K_R, right_crop_xy)):
        affine = torch.eye(3, dtype=k.dtype, device=k.device)
        affine[0, 0], affine[1, 1] = sx, sy
        affine[0, 2], affine[1, 2] = sx * (0.5 - x) - 0.5, sy * (0.5 - y) - 0.5
        matrices.append(affine @ k)
    return RectifiedStereoCamera(matrices[0], matrices[1], camera.T_R_from_L, True)


def resized_raw_disparity(
    resampled_disparity: torch.Tensor,
    *,
    left_crop_x: int,
    right_crop_x: int,
    crop_width: int,
    output_width: int,
) -> torch.Tensor:
    """Convert already resampled raw disparity to the cropped/resized pixel units.

    Resampling support/masks is caller-owned: invalid disparity must not bleed
    into valid neighbours. This function changes values, not the spatial grid.
    """
    if min(crop_width, output_width) <= 0 or min(left_crop_x, right_crop_x) < 0:
        raise ValueError("invalid crop/resize dimensions")
    return (resampled_disparity - left_crop_x + right_crop_x) * (output_width / crop_width)
