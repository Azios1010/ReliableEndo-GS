"""Camera-frame pixel rays and metric-depth backprojection."""

from dataclasses import dataclass

import torch

from reliable_endo_gs.geometry.conventions import GeometryConvention
from reliable_endo_gs.geometry.disparity import _count, _nan_where_invalid
from reliable_endo_gs.geometry.pixels import pixel_grid


@dataclass(frozen=True, slots=True)
class RayResult:
    """Camera-frame unnormalised rays shaped ``[B, 3, H, W]``."""

    rays: torch.Tensor
    convention: GeometryConvention


@dataclass(frozen=True, slots=True)
class BackprojectionDiagnostics:
    """Counts for a camera-frame backprojection."""

    total_count: int
    input_masked_count: int
    nonfinite_depth_count: int
    nonpositive_depth_count: int
    valid_count: int
    nonfinite_center_count: int


@dataclass(frozen=True, slots=True)
class BackprojectionResult:
    """Camera-frame centers shaped ``[B, 3, H, W]`` and a ``[B, 1, H, W]`` mask."""

    centers: torch.Tensor
    rays: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: BackprojectionDiagnostics
    convention: GeometryConvention


def pixel_rays(
    intrinsics: torch.Tensor,
    *,
    height: int,
    width: int,
    convention: GeometryConvention,
) -> RayResult:
    """Return ``K^-1 [u, v, 1]^T`` rays in the declared left-camera frame."""

    _validate_intrinsics(intrinsics)
    if height < 1 or width < 1:
        raise ValueError(f"height and width must be positive; got ({height}, {width})")
    batch_size = intrinsics.shape[0]
    pixels = pixel_grid(
        batch_size=batch_size,
        height=height,
        width=width,
        device=intrinsics.device,
        dtype=intrinsics.dtype,
        convention=convention,
    )
    homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)
    vectors = homogeneous.reshape(batch_size, height * width, 3).transpose(1, 2)
    try:
        rays = torch.linalg.solve(intrinsics, vectors)
    except torch.linalg.LinAlgError as error:
        raise ValueError("intrinsics must be invertible") from error
    rays = rays.reshape(batch_size, 3, height, width)
    if not bool(torch.all(torch.isfinite(rays))):
        raise ValueError("intrinsics produce non-finite pixel rays")
    return RayResult(rays, convention)


def backproject_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    convention: GeometryConvention,
) -> BackprojectionResult:
    """Backproject metric optical-axis depth with ``mu = D K^-1 [u, v, 1]^T``.

    ``depth`` and ``valid_mask`` have shape ``[B, 1, H, W]``. Invalid
    centers are ``NaN`` and are identified by the returned mask.
    """

    _validate_depth_inputs(depth, valid_mask)
    _validate_intrinsics(intrinsics)
    if intrinsics.shape[0] != depth.shape[0]:
        raise ValueError(
            "intrinsics batch dimension must match depth; "
            f"expected {depth.shape[0]}, got {intrinsics.shape[0]}"
        )
    if intrinsics.device != depth.device:
        raise ValueError("intrinsics must be on the same device as depth")
    if intrinsics.dtype != depth.dtype:
        raise TypeError("intrinsics must have the same dtype as depth")

    rays = pixel_rays(
        intrinsics,
        height=depth.shape[2],
        width=depth.shape[3],
        convention=convention,
    ).rays
    finite = torch.isfinite(depth)
    positive = depth > 0
    output_mask = valid_mask & finite & positive
    safe_depth = torch.where(output_mask, depth, torch.zeros_like(depth))
    centers = safe_depth * rays
    centers = _nan_where_invalid(centers, output_mask.expand(-1, 3, -1, -1))
    diagnostics = BackprojectionDiagnostics(
        total_count=depth.numel(),
        input_masked_count=_count(~valid_mask),
        nonfinite_depth_count=_count(valid_mask & ~finite),
        nonpositive_depth_count=_count(valid_mask & finite & ~positive),
        valid_count=_count(output_mask),
        nonfinite_center_count=_count(output_mask.expand(-1, 3, -1, -1) & ~torch.isfinite(centers)),
    )
    return BackprojectionResult(centers, rays, output_mask, diagnostics, convention)


def _validate_intrinsics(intrinsics: torch.Tensor) -> None:
    if not isinstance(intrinsics, torch.Tensor):
        raise TypeError("intrinsics must be a torch.Tensor")
    if intrinsics.ndim != 3 or tuple(intrinsics.shape[-2:]) != (3, 3):
        raise ValueError(f"intrinsics must have shape [B, 3, 3]; got {tuple(intrinsics.shape)}")
    if intrinsics.shape[0] < 1:
        raise ValueError("intrinsics batch dimension must be positive")
    if not torch.is_floating_point(intrinsics):
        raise TypeError(f"intrinsics must be floating-point; got {intrinsics.dtype}")
    if not bool(torch.all(torch.isfinite(intrinsics))):
        raise ValueError("intrinsics must be finite")


def _validate_depth_inputs(depth: torch.Tensor, valid_mask: torch.Tensor) -> None:
    if not isinstance(depth, torch.Tensor):
        raise TypeError("depth must be a torch.Tensor")
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"depth must have shape [B, 1, H, W]; got {tuple(depth.shape)}")
    if tuple(valid_mask.shape) != tuple(depth.shape):
        raise ValueError(
            f"valid_mask must have shape {tuple(depth.shape)}; got {tuple(valid_mask.shape)}"
        )
    if not torch.is_floating_point(depth):
        raise TypeError(f"depth must be floating-point; got {depth.dtype}")
    if valid_mask.dtype != torch.bool:
        raise TypeError(f"valid_mask must have dtype torch.bool; got {valid_mask.dtype}")
    if valid_mask.device != depth.device:
        raise ValueError("valid_mask must be on the same device as depth")
