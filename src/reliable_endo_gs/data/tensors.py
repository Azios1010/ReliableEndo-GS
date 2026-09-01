"""Tensor conversion helpers for decoded external sample arrays."""

from typing import Any

import torch


def rgb_array_to_tensor(array: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert an ``[H, W, 3]``/``[H, W, 4]`` uint8 array to ``[3, H, W]``.

    RGB values are explicitly scaled from ``[0, 255]`` to ``[0, 1]``.  An
    alpha channel is discarded because the Plan 02 contract is RGB-only.
    """

    tensor = torch.as_tensor(array)
    if tensor.ndim != 3 or tensor.shape[2] not in (3, 4):
        raise ValueError(f"RGB input must have shape [H, W, 3/4]; got {tuple(tensor.shape)}")
    if tensor.dtype != torch.uint8:
        raise TypeError(f"RGB input must be uint8; got {tensor.dtype}")
    return tensor[..., :3].permute(2, 0, 1).to(dtype=dtype).div(255.0).contiguous()


def xyz_array_to_tensor(array: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert an ``[H, W, 3]`` floating XYZ map to ``[3, H, W]``."""

    tensor = torch.as_tensor(array)
    if tensor.ndim != 3 or tensor.shape[2] != 3:
        raise ValueError(f"XYZ input must have shape [H, W, 3]; got {tuple(tensor.shape)}")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"XYZ input must be floating point; got {tensor.dtype}")
    return tensor.to(dtype=dtype).permute(2, 0, 1).contiguous()


def xyz_valid_mask(xyz: torch.Tensor) -> torch.Tensor:
    """Return an explicit ``[1, H, W]`` mask requiring all XYZ channels finite."""

    if xyz.ndim != 3 or xyz.shape[0] != 3:
        raise ValueError(f"XYZ tensor must have shape [3, H, W]; got {tuple(xyz.shape)}")
    return torch.isfinite(xyz).all(dim=0, keepdim=True)
