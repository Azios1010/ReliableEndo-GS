"""Explicit A00 metric-geometry variant of the released Endo-E2E-GS network.

The learned upstream modules and sequence/RGB losses are unchanged. This is
not native parity: metric A00 geometry replaces per-image normalized depth,
integer pixel centers replace half-pixel backprojection, and the common A00
renderer supplies both calibrated target cameras without a GT inference mask.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from reliable_endo_gs.baseline.upstream import import_upstream_module
from reliable_endo_gs.contracts import GaussianField, RenderOutput
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch
from reliable_endo_gs.geometry.rectified import reconstruct_rectified
from reliable_endo_gs.rendering.field import FieldRenderRequest
from reliable_endo_gs.rendering.production import ProductionFieldRenderer


@dataclass(frozen=True)
class A00EndoOutput:
    """Disparity [B,1,H,W], optional scalar loss and metric [B,N,*] field."""

    disparity: torch.Tensor
    disparity_loss: torch.Tensor | None
    field: GaussianField | None
    scale_underflow_count: int = 0


def stabilize_zero_scales(scale: torch.Tensor) -> torch.Tensor:
    """Map exact zero standard deviations to FP32's smallest normal value.

    Input/output are [B,3,H,W]. Positive values, including subnormals, are
    unchanged after the existing FP32 conversion. Negatives/NaNs/infinities
    remain invalid and are rejected by the renderer. This opt-in numerical
    policy does not replace the upstream Softplus(beta=100)/upper clamp.
    The replacement squares to zero in FP32, just as the underflowed scale;
    native covariance/render/gradient parity must be validated before a run.
    """
    if not torch.is_floating_point(scale):
        raise TypeError("Gaussian scales must be floating point")
    value = scale.float()
    return torch.where(value == 0, value.new_tensor(torch.finfo(value.dtype).tiny), value)


class A00EndoModel(nn.Module):
    """Compose untouched learned modules with the canonical metric geometry.

    GT disparity/validity are separate training arguments, never deployment
    inputs. The module has no repair, teacher or routing dependency.
    """

    def __init__(
        self,
        native: nn.Module,
        sequence_loss: Callable[..., Any],
        *,
        train_iters: int = 3,
        val_iters: int = 3,
        scale_underflow_policy: str = "reject",
    ) -> None:
        super().__init__()
        self.native = native
        self.sequence_loss = sequence_loss
        self.train_iters = train_iters
        self.val_iters = val_iters
        if scale_underflow_policy not in ("reject", "zero_to_float32_tiny"):
            raise ValueError("unsupported baseline scale underflow policy")
        self.scale_underflow_policy = scale_underflow_policy

    def forward(
        self,
        batch: DeploymentStereoBatch,
        *,
        gt_disparity: torch.Tensor | None = None,
        gt_valid: torch.Tensor | None = None,
        with_gaussians: bool = False,
        mixed_precision: bool = False,
    ) -> A00EndoOutput:
        """Map RGB [B,3,H,W] to raw positive-sign A00 disparity and optional GS.

        Prediction signs are not flipped or clamped: upstream supervised
        sequence loss learns the declared d=u_left-u_right labels directly.
        Invalid metric predictions are excluded with their coverage reported.
        """
        if (gt_disparity is None) != (gt_valid is None):
            raise ValueError("training requires both GT disparity and validity")
        image = torch.cat((batch.left * 2 - 1, batch.right * 2 - 1), dim=0)
        with torch.autocast(image.device.type, enabled=mixed_precision):
            features = self.native.img_encoder(image)
            predictions = self.native.raft_stereo(
                features[2],
                iters=self.train_iters if gt_disparity is not None else self.val_iters,
                test_mode=gt_disparity is None,
            )
        disparity_loss = None
        if gt_disparity is not None:
            assert gt_valid is not None
            disparity_loss, _ = self.sequence_loss(predictions, gt_disparity, gt_valid)
            disparity = predictions[-1].float()
        else:
            disparity = predictions.float()
        if not bool(torch.isfinite(disparity).all()):
            raise ValueError("nonfinite Endo-E2E-GS disparity")
        field = None
        scale_underflow_count = 0
        if with_gaussians:
            geometry = reconstruct_rectified(
                disparity, batch.camera, batch.support, min_effective_disparity=1e-6
            )
            valid = geometry.depth.valid_mask
            # Only invalid slots are sanitized; no normalization or depth clipping.
            depth = torch.where(valid, geometry.depth.depth, 0)
            centers = torch.where(
                valid.expand_as(geometry.points.centers), geometry.points.centers, 0
            )
            rotation, scale, alpha = self.native.gs_parm_regresser(
                batch.left * 2 - 1, depth, features
            )
            if self.scale_underflow_policy == "zero_to_float32_tiny":
                scale_underflow_count = int(((scale == 0) & valid.expand_as(scale)).sum())
                scale = stabilize_zero_scales(scale)

            def flatten(value: torch.Tensor) -> torch.Tensor:
                return value.float().flatten(2).transpose(1, 2).contiguous()

            field = GaussianField(
                means3d=flatten(centers),
                colors=flatten(batch.left),
                rotations=flatten(rotation),
                scales=flatten(scale),
                opacities=flatten(alpha),
                valid_mask=valid.flatten(1),
            )
        return A00EndoOutput(disparity, disparity_loss, field, scale_underflow_count)


def build_a00_endo_model(
    upstream: Path,
    *,
    corr_implementation: str | None = None,
    scale_underflow_policy: str = "reject",
) -> A00EndoModel:
    """Load pinned modules lazily; construct fresh full network with no weights."""
    config_module = import_upstream_module("config.stereo_config", upstream)
    loader = config_module.ConfigStereo()
    loader.load(str(upstream / "config/stage2.yaml"))
    cfg = loader.get_cfg()
    cfg.defrost()
    cfg.stage1_ckpt = None
    cfg.restore_ckpt = None
    cfg.batch_size = 1
    # Upstream internal autocast follows the stage-specific call argument below.
    cfg.raft.mixed_precision = False
    if corr_implementation is not None:
        cfg.raft.corr_implementation = corr_implementation
    cfg.freeze()
    network = import_upstream_module(
        "lib.network", upstream, require_corr_sampler=cfg.raft.corr_implementation != "reg"
    )
    losses = import_upstream_module("lib.loss", upstream)
    return A00EndoModel(
        network.StereoEndoModel(cfg, with_gs_render=True),
        losses.sequence_loss,
        train_iters=int(cfg.raft.train_iters),
        val_iters=int(cfg.raft.val_iters),
        scale_underflow_policy=scale_underflow_policy,
    )


def render_a00_endo(
    output: A00EndoOutput,
    batch: DeploymentStereoBatch,
    renderer: ProductionFieldRenderer,
    *,
    view: str,
) -> RenderOutput:
    """Render the metric field at the original left/right calibrated camera."""
    if output.field is None:
        raise ValueError("Gaussian prediction is required for rendering")
    return renderer.render(
        FieldRenderRequest(output.field, batch.camera, tuple(batch.left.shape[-2:]), view)
    )
