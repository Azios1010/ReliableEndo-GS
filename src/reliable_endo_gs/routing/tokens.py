"""Bounded masked pooling of existing student features; no GT or second backbone."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from reliable_endo_gs.actions.repair import RepairState
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.regions.grid import Region
from reliable_endo_gs.regions.masks import MaskRegion
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES


def selector_feature_names(config: InterventionConfig) -> tuple[str, ...]:
    """Versioned [R,F] layout, retaining the historical statistics prefix."""
    if config.selector_feature_mode == "statistics":
        return BALANCED_FEATURE_NAMES
    return BALANCED_FEATURE_NAMES + tuple(
        f"{kind}_{i}"
        for kind in ("local_student_mean", "local_student_max", "global_student_mean")
        for i in range(config.selector_token_channels)
    )


@torch.no_grad()
def mask_feature_tokens(
    state: RepairState, regions: tuple[Region, ...], channels: int
) -> torch.Tensor:
    """[R,3C]: masked local mean/max and global mean from stride-4 [1,C,h,w].

    Two k3/p1/s2 convolutions put native centers at (4*y,4*x). Sample only
    original write pixels at their exact native coordinates, with bilinear
    interpolation and border extension. There is no full-resolution C-channel
    interpolation, box pooling or inclusion of mask holes. Global support is
    sampled at native centers. The backbone's receptive field may span context.
    """
    evidence = getattr(state.base, "matching_evidence", {})
    native = evidence.get("student_features")
    h, w = state.batch.left.shape[-2:]
    if (
        type(channels) is not int
        or channels <= 0
        or not isinstance(native, torch.Tensor)
        or native.shape != (1, channels, (h + 3) // 4, (w + 3) // 4)
        or native.device != state.batch.left.device
        or native.requires_grad
        or not bool(torch.isfinite(native).all())
        or not regions
        or any(not isinstance(r, MaskRegion) or r.image_hw != (h, w) for r in regions)
    ):
        raise ValueError("mask tokens require detached matching student channels and real masks")
    values = native.float()
    global_support = state.batch.support[..., ::4, ::4]
    if not bool(global_support.any()):
        raise ValueError("mask tokens require observable global support")
    global_mean = torch.where(global_support, values, 0).sum((2, 3))[0] / global_support.sum()
    rows = []
    nh, nw = values.shape[-2:]
    for region in regions:
        assert isinstance(region, MaskRegion)
        flat = torch.tensor(region.flat_indices, device=values.device, dtype=torch.long)
        y, x = flat.div(w, rounding_mode="floor"), flat.remainder(w)
        grid = torch.stack(
            (2 * (x.float() / 4 + 0.5) / nw - 1, 2 * (y.float() / 4 + 0.5) / nh - 1), -1
        )[None, None]
        sampled = F.grid_sample(
            values, grid, mode="bilinear", padding_mode="border", align_corners=False
        )[0, :, 0]
        support = state.support.flatten()[flat]
        if not bool(support.any()):
            raise ValueError("mask token has no observable write pixels")
        mean = torch.where(support, sampled, 0).sum(1) / support.sum()
        maximum = sampled.masked_fill(~support, -torch.inf).amax(1)
        rows.append(torch.cat((mean, maximum, global_mean)))
    return torch.stack(rows).detach()
