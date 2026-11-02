"""A00 boundary for native MVSplat and DepthSplat encoder/renderer pairs.

No learned layer, Gaussian parameter, or renderer is replaced. Native SH rendering
uses both source views (``left_source_only`` renders with the left source grid only). The field in the benchmark return is diagnostic only.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch
from torch import nn

from reliable_endo_gs.contracts import GaussianField, RenderOutput
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.evaluation.paired_models import FrozenModelPrediction
from reliable_endo_gs.geometry.conventions import GeometryConvention
from reliable_endo_gs.geometry.disparity import depth_to_disparity


@dataclass(frozen=True)
class PosedGaussianConfig:
    """Explicit optical-axis depth bounds in metres; no dataset-derived defaults."""

    near_m: float
    far_m: float
    global_step: int
    # Inference-time variant: render both cameras with the Gaussians of the left source grid only (default: both grids).
    left_source_only: bool = False

    def __post_init__(self) -> None:
        if not (isfinite(self.near_m) and isfinite(self.far_m)):
            raise ValueError("near/far must be finite metres")
        if not 0 < self.near_m < self.far_m or self.global_step < 0:
            raise ValueError("require 0 < near_m < far_m and nonnegative global_step")


def posed_context(
    batch: DeploymentStereoBatch, config: PosedGaussianConfig
) -> dict[str, torch.Tensor]:
    """Convert A00 RGB [1,3,H,W] and metre cameras to native [1,2,...].

    Native image grids use (u+0.5)/W, (v+0.5)/H; A00 uses integer pixel
    centers. Thus add 0.5 to principal points before normalizing K rows.
    World is the left camera; c2w_R = inverse(T_R_from_L). No resizing,
    baseline normalization, GT-based alignment, or image masking is applied.
    """
    if batch.left.shape[0] != 1:
        raise ValueError("benchmark requires batch size one")
    h, w = batch.left.shape[-2:]
    camera = batch.camera
    k = torch.stack((camera.K_L, camera.K_R), dim=1).clone()
    k[:, :, 0, 2] += 0.5
    k[:, :, 1, 2] += 0.5
    k[:, :, 0] /= w
    k[:, :, 1] /= h
    left_pose = torch.eye(4, dtype=k.dtype, device=k.device)[None]
    right_pose = torch.linalg.inv(camera.T_R_from_L)
    return {
        "image": torch.stack((batch.left, batch.right), dim=1),
        "intrinsics": k,
        "extrinsics": torch.stack((left_pose, right_pose), dim=1),
        "near": k.new_full((1, 2), config.near_m),
        "far": k.new_full((1, 2), config.far_m),
        "index": torch.tensor([[0, 1]], device=k.device),
    }


class PosedGaussianPredictor(nn.Module):
    """Native prediction with the benchmark's forward(sample) interface.

    Encoder: context [1,2,...] -> native Gaussians, plus visualization depth
    [1,2,H,W,1,1], scales [1,2HW,3], rotations [1,2HW,4] (xyzw).
    Decoder: native Gaussians + both input cameras -> color [1,2,3,H,W].
    Returned disparity [1,1,H,W] is derived from the encoder's plane-sweep
    optical-axis depth, not from a rendered depth or a Gaussian's range.
    Only the one-surface/one-Gaussian-per-pixel recipe is supported.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, config: PosedGaussianConfig):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

    def _encode(self, batch: DeploymentStereoBatch) -> tuple[Any, dict, dict]:
        context = posed_context(batch, self.config)
        dump: dict[str, torch.Tensor] = {}
        output = self.encoder(
            context, self.config.global_step, deterministic=True, visualization_dump=dump
        )
        gaussians = output["gaussians"] if isinstance(output, dict) else output
        h, w = batch.left.shape[-2:]
        if dump["depth"].shape != (1, 2, h, w, 1, 1):
            raise ValueError("native depth must be [1,2,H,W,1,1]; no implicit resampling")
        return gaussians, dump, context

    @staticmethod
    def _disparity(depths: torch.Tensor, batch: DeploymentStereoBatch) -> torch.Tensor:
        depth = depths[:, 0, :, :, 0, 0].unsqueeze(1).to(batch.camera.K_L.dtype)
        camera = batch.camera
        result = depth_to_disparity(
            depth,
            camera.K_L[:, 0, 0],
            camera.baseline,
            batch.support,
            min_depth=0.0,
            convention=GeometryConvention(),
        )
        offset = (camera.K_L[:, 0, 2] - camera.K_R[:, 0, 2]).view(1, 1, 1, 1)
        return result.disparity + offset

    def predict_disparity(self, batch: DeploymentStereoBatch) -> torch.Tensor:
        """Return raw d=u_L-u_R [1,1,H,W], including unequal-principal-point crops."""
        _, dump, _ = self._encode(batch)
        return self._disparity(dump["depth"], batch)

    def forward(self, sample: ScaredA00Sample) -> FrozenModelPrediction:
        """Render two input cameras at [1,3,H,W]; GT is never an encoder input."""
        if sample.supervision is not None:
            raise ValueError("predictor input must not contain ground truth")
        batch = sample.batch
        gaussians, dump, context = self._encode(batch)
        h, w = batch.left.shape[-2:]
        sources = 1 if self.config.left_source_only else 2
        if self.config.left_source_only:
            # Native Gaussians are view-major: the first h*w belong to the left source grid (see `indices` below).
            keep = h * w
            gaussians = type(gaussians)(
                means=gaussians.means[:, :keep],
                covariances=gaussians.covariances[:, :keep],
                harmonics=gaussians.harmonics[:, :keep],
                opacities=gaussians.opacities[:, :keep],
            )
        # Rectified A00 cameras have identity relative rotation. Keep the native
        # covariance/SH renderer; never re-render through our Gaussian prior.
        decoded = self.decoder(
            gaussians,
            context["extrinsics"],
            context["intrinsics"],
            context["near"],
            context["far"],
            (h, w),
        )
        if decoded.color.shape != (1, 2, 3, h, w):
            raise ValueError("native decoder color must be [1,2,3,H,W]")
        if not bool(torch.isfinite(decoded.color).all()):
            raise ValueError("native decoder produced nonfinite RGB")
        count = sources * h * w
        if gaussians.means.shape != (1, count, 3):
            raise ValueError(f"expected exactly {sources} source grid(s) of native Gaussians")
        # Flatten native SH coefficients into diagnostic field colors. This
        # field is not rendered, and its colors are not an RGB approximation.
        means = gaussians.means.float()
        field = GaussianField(
            means3d=means,
            colors=gaussians.harmonics.float().flatten(2),
            rotations=dump["rotations"].float()[:, :count][..., [3, 0, 1, 2]],
            scales=dump["scales"].float()[:, :count],
            opacities=gaussians.opacities.float().unsqueeze(-1),
            valid_mask=torch.isfinite(means).all(-1),
            cov_surface=gaussians.covariances.float(),
        )
        # Right-source pixels have no left-grid correspondence. -1 explicitly
        # excludes them from the optional source-indexed centre Chamfer metric.
        # Both source fields remain present in native RGB rendering.
        indices = torch.cat(
            (
                torch.arange(h * w, device=means.device),
                torch.full(((sources - 1) * h * w,), -1, device=means.device),
            ),
            dim=0,
        ).unsqueeze(0)
        return FrozenModelPrediction(
            self._disparity(dump["depth"], batch),
            field,
            indices,
            RenderOutput(decoded.color[:, 0]),
            RenderOutput(decoded.color[:, 1]),
        )
