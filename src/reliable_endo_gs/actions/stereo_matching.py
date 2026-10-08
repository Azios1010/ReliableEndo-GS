"""Correspondence-aware residual action; reuses canonical full-right sampling."""

from __future__ import annotations

from typing import Literal, overload

import torch
from torch import nn

from reliable_endo_gs.geometry.stereo_matching import warp_right


class MatchingStereoRepair(nn.Module):
    """Search around current disparity, then spatially predict a bounded update.

    Inputs: RGB BCHW, raw disparity B1HW (full-grid px), RGB render residual.
    Output: dimensionless B1hw update logits; RepairActions owns geometry/bounds.
    """

    def __init__(
        self, width: int, radius_px: int, step_px: int, temperature: float, hidden_width: int
    ) -> None:
        super().__init__()
        if (
            min(width, radius_px, step_px, hidden_width) < 1
            or radius_px % step_px
            or temperature <= 0
        ):
            raise ValueError("invalid matching head parameters")
        self.radius_px, self.step_px, self.temperature = radius_px, step_px, temperature
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
        )
        candidates = 2 * (radius_px // step_px) + 1
        self.decoder = nn.Sequential(
            nn.Conv2d(candidates + 7, hidden_width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_width, 1, 3, padding=1),
        )
        output = self.decoder[-1]
        if not isinstance(output, nn.Conv2d) or output.bias is None:
            raise TypeError("matching output requires convolution bias")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    @overload
    def forward_pair(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        disparity: torch.Tensor,
        residual: torch.Tensor,
        halo: tuple[int, int, int, int],
        support: torch.Tensor,
        principal_offset: torch.Tensor,
        *,
        roi_features: bool = False,
        return_validity: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def forward_pair(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        disparity: torch.Tensor,
        residual: torch.Tensor,
        halo: tuple[int, int, int, int],
        support: torch.Tensor,
        principal_offset: torch.Tensor,
        *,
        roi_features: bool = False,
        return_validity: Literal[True],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward_pair(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        disparity: torch.Tensor,
        residual: torch.Tensor,
        halo: tuple[int, int, int, int],
        support: torch.Tensor,
        principal_offset: torch.Tensor,
        *,
        roi_features: bool = False,
        return_validity: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Full input grids; output B1hw on the supplied halo, with explicit origin.

        When ``return_validity`` is true, return a second ``[B,1,h,w]`` mask
        indicating that at least one searched disparity candidate has valid
        right-view support.  A row with no valid candidate is deliberately
        kept distinct from a low-confidence/ambiguous row: its probability
        vector is all zeros and callers must abstain from applying a learned
        correction there.
        """
        y0, x0, y1, x1 = halo
        if roi_features:
            # Two 3x3 stride-1 convolutions have receptive radius 2. Expand
            # before convolution, then retain exact halo features. The right
            # strip keeps full width so every disparity candidate has context.
            # Padding back to the full grid preserves canonical warp origins.
            h, w = left.shape[-2:]
            ly0, lx0, ly1, lx1 = max(0, y0 - 2), max(0, x0 - 2), min(h, y1 + 2), min(w, x1 + 2)
            lf = self.features(left[..., ly0:ly1, lx0:lx1].float())
            lf = lf[..., y0 - ly0 : y1 - ly0, x0 - lx0 : x1 - lx0]
            rf = self.features(right[..., ly0:ly1, :].float())
            rf = torch.nn.functional.pad(rf, (0, 0, ly0, h - ly1))
        else:
            lf = self.features(left.float())[..., y0:y1, x0:x1]
            rf = self.features(right.float())
        d = disparity[..., y0:y1, x0:x1].float()
        scores, masks = [], []
        for offset in range(-self.radius_px, self.radius_px + 1, self.step_px):
            warped, valid = warp_right(rf, d + offset, origin_xy=(x0, y0), right_support=support)
            valid = valid & (d + offset - principal_offset > 0)
            score = -(lf - warped).square().mean(1, keepdim=True) / self.temperature
            scores.append(score)
            masks.append(valid)
        valid = torch.cat(masks, dim=1)
        score = torch.cat(scores, dim=1)
        evidence = valid.any(dim=1, keepdim=True)
        # Finite sentinel prevents all-invalid softmax NaNs; empty rows carry
        # zero candidate probabilities and are covered by sensor masks later.
        probability = score.masked_fill(~valid, -1e4).softmax(1) * valid.to(score.dtype)
        probability = probability / probability.sum(1, keepdim=True).clamp_min(1e-6)
        context = torch.cat(
            (
                probability,
                left[..., y0:y1, x0:x1].float(),
                residual[..., y0:y1, x0:x1].float(),
                d / left.shape[-1],
            ),
            1,
        )
        output = self.decoder(context)
        if not isinstance(output, torch.Tensor):
            raise TypeError("matching decoder must return a Tensor")
        if return_validity:
            return output, evidence
        return output
