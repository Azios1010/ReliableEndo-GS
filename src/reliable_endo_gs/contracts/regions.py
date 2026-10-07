"""Deployment target-ROI contract, independent of GT and sensor validity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True, slots=True)
class TargetROI:
    """Bool [1,1,H,W] LEFT output-grid ROI from input or RGB prediction.

    This is independent of sensor support, GT validity and error candidates.
    Source identity records the ROI producer; these declarations cannot prove
    how a third party made a mask, which must be audited separately.
    """

    pixels: torch.Tensor
    source: Literal["input_roi", "rgb_prediction"]
    source_identity: str

    def __post_init__(self) -> None:
        if self.source not in ("input_roi", "rgb_prediction"):
            raise ValueError("target ROI must originate from deployment input or RGB prediction")
        if not isinstance(self.source_identity, str) or not self.source_identity.strip():
            raise ValueError("target ROI requires producer identity")
        if (
            not isinstance(self.pixels, torch.Tensor)
            or self.pixels.ndim != 4
            or self.pixels.shape[:2] != (1, 1)
            or self.pixels.dtype != torch.bool
            or min(self.pixels.shape[-2:]) < 1
        ):
            raise ValueError("target ROI must be bool [1,1,H,W]")
        object.__setattr__(self, "pixels", self.pixels.detach().clone())
