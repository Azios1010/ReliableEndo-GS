"""Opt-in multiscale stereo features and learned matching-cost context."""

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


class ParallelMomentsGroupNorm(nn.GroupNorm):
    """``GroupNorm(1, C)`` whose moments are reduced by a many-block kernel.

    Input/output: [N,C,*] -> [N,C,*]. Native single-group GroupNorm reduces one
    row of C*H*W values per sample in ONE CUDA block, so at batch one only one
    streaming multiprocessor works. This subclass computes the same mean and
    biased variance with ``torch.var_mean`` (split over many blocks), then
    applies the affine map as one fused ``x * scale + shift`` with
    ``scale = rsqrt(var + eps) * weight`` and ``shift = bias - mean * scale``.

    Numerical approximation (documented beside the formula): the accumulation
    order and the fused rounding differ from the native kernel, so results agree
    to float32 rounding (about 1e-6 relative at unit scale), not bitwise.
    ``shift = bias - mean * scale`` loses precision only when ``|mean| / std``
    reaches the thousands, far outside normalised feature maps. Parameters and
    ``state_dict`` keys are those of ``nn.GroupNorm``; only ``num_groups == 1``
    is supported. Low-precision inputs are reduced in float32.
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(1, num_channels, eps=eps, affine=affine, device=device, dtype=dtype)

    @classmethod
    def from_group_norm(cls, source: nn.GroupNorm) -> "ParallelMomentsGroupNorm":
        """Same parameters, device, dtype, mode and ``requires_grad`` as ``source``."""
        if source.num_groups != 1:
            raise ValueError("parallel-moments GroupNorm supports exactly one group")
        reference = source.weight if source.weight is not None else None
        converted = cls(
            source.num_channels,
            source.eps,
            source.affine,
            device=None if reference is None else reference.device,
            dtype=None if reference is None else reference.dtype,
        )
        converted.load_state_dict(source.state_dict())
        converted.train(source.training)
        if source.affine:
            converted.weight.requires_grad_(source.weight.requires_grad)
            converted.bias.requires_grad_(source.bias.requires_grad)
        return converted

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim < 2 or image.shape[1] != self.num_channels:
            raise ValueError("input must be [N,C,*] with the module's channel count")
        work = image if image.dtype in (torch.float32, torch.float64) else image.float()
        variance, mean = torch.var_mean(
            work, dim=tuple(range(1, work.ndim)), keepdim=True, unbiased=False
        )
        scale = torch.rsqrt(variance + self.eps)
        if self.affine:
            channel_shape = (1, -1) + (1,) * (work.ndim - 2)
            scale = scale * self.weight.view(channel_shape).to(work.dtype)
            shift = self.bias.view(channel_shape).to(work.dtype) - mean * scale
        else:
            shift = -mean * scale
        return cast(torch.Tensor, torch.addcmul(shift, work, scale).to(image.dtype))


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(1, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(1, channels),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.relu(image + self.body(image))


def _downsample(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, stride=2, padding=1, bias=False),
        nn.GroupNorm(1, outputs),
        nn.ReLU(),
    )


class MultiscaleStereoFeatures(nn.Module):
    """RGB [B,3,H,W] -> fused [B,C,ceil(H/4),ceil(W/4)] stereo features.

    Shared residual branches at strides 4, 8 and 16 use GroupNorm so batch-one
    training and later freezing do not depend on running BatchNorm statistics.
    Bilinear fusion retains the quarter-grid matching coordinate convention.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.stem = _downsample(3, width // 2)
        self.quarter = nn.Sequential(
            _downsample(width // 2, width), _ResidualBlock(width), _ResidualBlock(width)
        )
        self.eighth = nn.Sequential(
            _downsample(width, 2 * width),
            _ResidualBlock(2 * width),
            _ResidualBlock(2 * width),
        )
        self.sixteenth = nn.Sequential(_downsample(2 * width, 2 * width), _ResidualBlock(2 * width))
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * width, width, 1, bias=False),
            nn.GroupNorm(1, width),
            nn.ReLU(),
            _ResidualBlock(width),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Fuse contextual features while keeping matching on the stride-4 grid."""
        quarter = self.quarter(self.stem(image))
        eighth = self.eighth(quarter)
        sixteenth = self.sixteenth(eighth)
        return cast(
            torch.Tensor,
            self.fuse(
                torch.cat(
                    (
                        quarter,
                        F.interpolate(
                            eighth, quarter.shape[-2:], mode="bilinear", align_corners=False
                        ),
                        F.interpolate(
                            sixteenth, quarter.shape[-2:], mode="bilinear", align_corners=False
                        ),
                    ),
                    dim=1,
                )
            ),
        )


class StereoCostRegularizer(nn.Module):
    """Residual context for cost/matching validity [B,D,H/4,W/4].

    Invalid costs must be replaced before convolution; validity is a separate
    channel. The owner reapplies the hard epipolar mask before softmax. The
    zero-initialized final convolution starts with the canonical raw costs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(2, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 1, 3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, costs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Return finite residual costs [B,D,H/4,W/4], without an input -inf."""
        finite = torch.where(valid, costs, torch.zeros_like(costs))
        inputs = torch.stack((finite, valid.to(costs.dtype)), dim=1)
        return cast(torch.Tensor, self.body(inputs).squeeze(1))


class StereoContextDecoder(nn.Module):
    """RGB pair plus normalized coarse disparity [B,7,H,W] -> [B,1,H,W]."""

    def __init__(self, width: int) -> None:
        super().__init__()
        channels = max(8, width // 2)
        self.body = nn.Sequential(
            nn.Conv2d(7, channels, 3, padding=1),
            nn.ReLU(),
            _ResidualBlock(channels, 1),
            _ResidualBlock(channels, 2),
            _ResidualBlock(channels, 4),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict a contextual residual; the owner enforces effective disparity."""
        return cast(torch.Tensor, self.body(inputs))
