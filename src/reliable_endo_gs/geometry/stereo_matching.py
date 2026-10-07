"""Canonical rectified right-image sampling; integer centres, current-grid px."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _right_coordinates(
    disparity: torch.Tensor, origin_xy: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single owner of u_R=u_L-d in raw current-grid coordinates."""
    h, w = disparity.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(h, device=disparity.device),
        torch.arange(w, device=disparity.device),
        indexing="ij",
    )
    u = x.to(disparity.dtype)[None] + origin_xy[0] - disparity[:, 0]
    v = (y.to(disparity.dtype)[None] + origin_xy[1]).expand_as(u)
    return u, v


@torch.no_grad()
def project_left_support(mask: torch.Tensor, disparity: torch.Tensor) -> torch.Tensor:
    """Project a fixed left mask B1HW to right pixels, floor/ceil splatting.

    This conservative pre-action mask does not use GT or post-action disparity.
    It avoids treating a repair's right-view core as a left-view image rectangle.
    """
    if mask.shape != disparity.shape or mask.dtype != torch.bool or mask.device != disparity.device:
        raise ValueError("projection mask must be on-device bool matching disparity")
    h, w = disparity.shape[-2:]
    u, v = _right_coordinates(disparity, (0, 0))
    finite = torch.isfinite(u)
    safe = torch.where(finite, u, torch.zeros_like(u))
    output = torch.zeros_like(mask, dtype=torch.int32).flatten(2)
    for position in (safe.floor(), safe.ceil()):
        valid = mask[:, 0] & finite & (position >= 0) & (position < w)
        indices = (v.long() * w + position.clamp(0, w - 1).long()).flatten(1)[:, None]
        output.scatter_add_(2, indices, valid.flatten(1)[:, None].to(torch.int32))
    return (output > 0).reshape_as(mask)


def warp_right(
    right: torch.Tensor,
    disparity: torch.Tensor,
    *,
    origin_xy: tuple[int, int] = (0, 0),
    right_support: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample full right [B,C,H,W] at u_R=u_L-d for a [B,1,h,w] left ROI.

    Return warped [B,C,h,w] and bool [B,1,h,w] validity. Zero padding is never
    interpreted as valid correspondence. No crop-width disparity rescaling.
    """
    if (
        right.ndim != 4
        or disparity.ndim != 4
        or disparity.shape[1] != 1
        or right.shape[0] != disparity.shape[0]
    ):
        raise ValueError("right and disparity must be compatible BCHW tensors")
    if (
        right.device != disparity.device
        or not right.is_floating_point()
        or not disparity.is_floating_point()
    ):
        raise ValueError("matching tensors must be floating and share device")
    if min(origin_xy) < 0:
        raise ValueError("ROI origin must be nonnegative")
    h, w = disparity.shape[-2:]
    rh, rw = right.shape[-2:]
    u, v = _right_coordinates(disparity, origin_xy)
    finite = torch.isfinite(u) & torch.isfinite(v)
    valid = (finite & (u >= 0) & (u <= rw - 1) & (v >= 0) & (v <= rh - 1))[:, None]
    u, v = torch.where(finite, u, torch.zeros_like(u)), torch.where(finite, v, torch.zeros_like(v))
    grid = torch.stack((2 * (u + 0.5) / rw - 1, 2 * (v + 0.5) / rh - 1), -1).to(right.dtype)
    warped = F.grid_sample(right, grid, padding_mode="zeros", align_corners=False)
    if right_support is not None:
        if (
            right_support.shape != (right.shape[0], 1, rh, rw)
            or right_support.dtype != torch.bool
            or right_support.device != right.device
        ):
            raise ValueError("right support must be on-device bool [B,1,H,W]")
        sampled_support = F.grid_sample(
            right_support.to(right.dtype), grid, mode="bilinear", align_corners=False
        )
        valid = valid & (sampled_support >= 1 - 1e-6)
    return warped, valid


def correspondence_error(
    left: torch.Tensor,
    right: torch.Tensor,
    disparity: torch.Tensor,
    *,
    right_support: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB absolute residual [B,1,H,W]; invalid matches have penalty 1.

    This is a deployment proxy, not metric depth error or calibrated risk.
    """
    warped, valid = warp_right(right, disparity, right_support=right_support)
    if warped.shape != left.shape:
        raise ValueError("matching images must share the left disparity grid")
    error = (left.float() - warped.float()).abs().mean(1, keepdim=True)
    return torch.where(valid, error, torch.ones_like(error)), valid
