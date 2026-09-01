"""Small learned positive Laplace-scale head over approved stereo features."""

from dataclasses import dataclass

import torch
from torch import nn

from reliable_endo_gs.contracts.common import require_floating, require_tensor
from reliable_endo_gs.uncertainty.laplace import positive_laplace_scale, sigma_from_laplace_scale


@dataclass(frozen=True, slots=True)
class LaplaceHeadOutput:
    """Raw learned scale and derived, still uncalibrated standard deviation."""

    scale_b: torch.Tensor
    raw_sigma_d: torch.Tensor


class LaplaceUncertaintyHead(nn.Module):
    """A 1x1-convolutional head that uses only caller-supplied stereo features.

    Features have shape ``[B, C, H, W]`` and must originate from an audited,
    pre-render baseline boundary.  No ground truth is accepted at inference.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 16, epsilon: float = 1e-4) -> None:
        super().__init__()
        if in_channels < 1 or hidden_channels < 1:
            raise ValueError("in_channels and hidden_channels must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be strictly positive")
        self.epsilon = epsilon
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> LaplaceHeadOutput:
        """Predict positive Laplace scales for ``[B, C, H, W]`` features."""

        require_tensor("features", features)
        if features.ndim != 4:
            raise ValueError(f"features must have shape [B, C, H, W]; got {tuple(features.shape)}")
        if features.shape[1] != self.network[0].in_channels:
            raise ValueError(
                f"features has {features.shape[1]} channels; expected {self.network[0].in_channels}"
            )
        require_floating("features", features)
        if features.numel() == 0:
            raise ValueError("features must be non-empty")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features must be finite")
        scale_b = positive_laplace_scale(self.network(features), self.epsilon)
        return LaplaceHeadOutput(scale_b=scale_b, raw_sigma_d=sigma_from_laplace_scale(scale_b))
