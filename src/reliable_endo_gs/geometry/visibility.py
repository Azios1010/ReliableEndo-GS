"""Right-view columns that no Gaussian anchored in the left image can cover (canonical formula).

A pixel of the left image with disparity ``d = u_left - u_right`` lands ``d`` pixels to the left in the right view, so the last columns of the right
view, as many as the disparity at the right border, are not covered by any Gaussian anchored in the left image. The width of that strip is taken from
the disparity of the last ``BORDER_COLUMNS`` columns: its ``STRIP_QUANTILE`` quantile, rounded up. This is a numerical approximation of the geometric
border (the disparity varies along the border); both the right-view evaluation on the visible part and the visibility-masked photometric loss use it.
"""

from __future__ import annotations

import math

import torch

BORDER_COLUMNS = 8
STRIP_QUANTILE = 0.95


def strip_width(disparity: torch.Tensor) -> int:
    """Width in pixels of the right-view strip not visible from the left image; ``disparity`` is [1,1,H,W] in pixels (d = u_left - u_right)."""
    if disparity.ndim != 4 or disparity.shape[0] != 1 or disparity.shape[1] != 1:
        raise ValueError("disparity must have shape [1,1,H,W]")
    border = disparity[..., -BORDER_COLUMNS:]
    values = border[torch.isfinite(border)]
    if values.numel() == 0:
        raise ValueError("no finite disparity at the right border")
    return int(math.ceil(float(torch.quantile(values.float().flatten(), STRIP_QUANTILE))))


def right_visible_support(support: torch.Tensor, disparity: torch.Tensor) -> torch.Tensor:
    """``support`` [1,1,H,W] bool with the last ``strip_width(disparity)`` columns set to False; the disparity is read without gradient.

    At least one column must remain; the input is not modified.
    """
    if (
        support.dtype != torch.bool
        or support.ndim != 4
        or support.shape[0] != 1
        or support.shape[1] != 1
    ):
        raise ValueError("support must be a bool tensor of shape [1,1,H,W]")
    strip = strip_width(disparity.detach())
    width = support.shape[-1]
    if strip >= width:
        raise ValueError(f"strip of {strip} px does not fit an image of width {width}")
    visible = support.clone()
    if strip > 0:
        visible[..., width - strip :] = False
    return visible
