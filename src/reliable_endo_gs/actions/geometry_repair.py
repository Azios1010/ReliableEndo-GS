"""Deployment-only geometry candidates; independent of routing and GT.

These deterministic candidates are exploratory actions, not calibrated risk
estimators. They change raw disparity on an explicit core mask and abstain
elsewhere. Canonical A03 construction remains the owner of depth/Gaussians.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml
from torch.nn import functional as F

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch
from reliable_endo_gs.geometry.stereo_matching import warp_right

if TYPE_CHECKING:
    from reliable_endo_gs.actions.repair import ActionResult, RepairState
    from reliable_endo_gs.reconstruction.model import ReferenceReconstructionModel
    from reliable_endo_gs.regions.grid import Region


@dataclass(frozen=True, slots=True)
class GeometryRepairConfig:
    """All distances in current image-grid pixels; thresholds are pilot choices."""

    schema: str = "geometry_repair_candidate.v1"
    mode: str = "correspondence"
    radius_px: int = 4
    window: int = 5
    max_delta_px: float = 4.0
    effective_fraction: float = 0.25
    minimum_effective_px: float = 0.25
    minimum_improvement: float = 0.02
    uniqueness_margin: float = 0.02
    reverse_tolerance_px: float = 1.0
    color_distance: float = 0.05
    minimum_neighbors: int = 5
    outlier_px: float = 1.0
    mad_multiplier: float = 2.5

    def __post_init__(self) -> None:
        if self.schema != "geometry_repair_candidate.v1" or self.mode not in (
            "correspondence",
            "support_median",
        ):
            raise ValueError("unsupported geometry candidate")
        for name in ("radius_px", "window", "minimum_neighbors"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.window % 2 != 1 or self.minimum_neighbors >= self.window**2:
            raise ValueError("candidate needs an odd window and attainable neighbor count")
        for name, value in asdict(self).items():
            if name in ("schema", "mode"):
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.effective_fraction >= 1:
            raise ValueError("effective disparity correction fraction must be below one")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def load_geometry_repair_config(path: Path) -> GeometryRepairConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) - set(GeometryRepairConfig.__dataclass_fields__):
        raise ValueError("unknown geometry candidate settings")
    return GeometryRepairConfig(**raw)


@dataclass(frozen=True, slots=True)
class GeometryRepairResult:
    """Disparity [B,1,H,W] raw pixels and bool accepted mask on the same grid."""

    disparity: torch.Tensor
    accepted: torch.Tensor


def _descriptor(image: torch.Tensor) -> torch.Tensor:
    """Eight 3x3 local binary comparisons plus RGB, all in [0,1].

    Sampling interpolates these bits at subpixel coordinates: the resulting
    cost is a soft descriptor L1 distance, not exact integer Hamming distance.
    """
    gray = image.float().mean(1, keepdim=True)
    patches = F.unfold(F.pad(gray, (1, 1, 1, 1), mode="replicate"), 3)
    patches = patches.reshape(image.shape[0], 9, *image.shape[-2:])
    comparisons = (patches >= gray).float()[:, (0, 1, 2, 3, 5, 6, 7, 8)]
    return torch.cat((comparisons, image.float()), 1)


def _cost(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    descriptor = (left[:, :8] - right[:, :8]).abs().mean(1, keepdim=True)
    rgb = (left[:, 8:] - right[:, 8:]).abs().clamp(max=0.2).mean(1, keepdim=True) / 0.2
    return 0.8 * descriptor + 0.2 * rgb


def _aggregate(cost: torch.Tensor, valid: torch.Tensor, window: int) -> torch.Tensor:
    fraction = F.avg_pool2d(valid.float(), window, 1, window // 2)
    average = F.avg_pool2d(cost * valid, window, 1, window // 2) / fraction.clamp_min(1e-6)
    return torch.where(valid & (fraction >= 0.8), average, torch.full_like(average, 2.0))


class GeometryRepair:
    """One deployment implementation for candidate intervention and later inference."""

    def __init__(self, config: GeometryRepairConfig) -> None:
        self.config = config

    @torch.no_grad()
    def execute(
        self, state: RepairState, region: Region, reference: ReferenceReconstructionModel
    ) -> ActionResult:
        """Frozen native intervention; the same apply() also serves CPU screening."""
        from reliable_endo_gs.actions.repair import Action, ActionResult, geometry_repair_outcome

        if any(p.requires_grad for p in reference.parameters()):
            raise ValueError("geometry candidate requires a frozen reference")
        result = self.apply(
            state.batch, state.base.target.disparity, region.mask(state.batch.left) & state.support
        )
        if not bool(result.accepted.any()):
            if state.base.render_right is None:
                raise ValueError("candidate requires both base renders")
            # Deterministic abstention: no head/rasterizer work or FP32 roundtrip.
            return ActionResult(
                Action.STEREO,
                state.base.gaussians,
                result.disparity,
                state.base.render_left,
                state.base.render_right,
                (),
            )
        return geometry_repair_outcome(state, region, result.disparity, reference)

    @torch.no_grad()
    def apply(
        self, batch: DeploymentStereoBatch, disparity: torch.Tensor, core: torch.Tensor
    ) -> GeometryRepairResult:
        """No teacher, GT, training state, rendering or parameter mutation."""
        if (
            disparity.shape != batch.support.shape
            or core.shape != disparity.shape
            or core.dtype != torch.bool
            or core.device != disparity.device
            or disparity.device != batch.left.device
            or not disparity.is_floating_point()
            or not bool(torch.isfinite(disparity).all())
        ):
            raise ValueError("geometry candidate needs finite disparity and on-device bool core")
        cfg = self.config
        offset = (batch.camera.K_L[:, 0, 2] - batch.camera.K_R[:, 0, 2])[:, None, None, None]
        effective = disparity - offset
        limit = (effective.clamp_min(0) * cfg.effective_fraction).clamp(max=cfg.max_delta_px)
        if cfg.mode == "correspondence":
            proposed, accepted = self._correspondence(batch, disparity, offset, limit)
        else:
            proposed, accepted = self._median(batch, disparity, limit)
        accepted &= core & batch.support & (proposed - offset >= cfg.minimum_effective_px)
        output = torch.where(accepted, proposed.to(disparity.dtype), disparity)
        return GeometryRepairResult(output, accepted & (output != disparity))

    def _correspondence(
        self,
        batch: DeploymentStereoBatch,
        disparity: torch.Tensor,
        offset: torch.Tensor,
        limit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        left, right = _descriptor(batch.left), _descriptor(batch.right)
        # Complete descriptor footprint is required, including sensor borders.
        sensor = F.avg_pool2d(batch.support.float(), 3, 1, 1) >= 1 - 1e-6
        shifts = range(-cfg.radius_px, cfg.radius_px + 1)
        costs = []
        for shift in shifts:
            candidate = disparity + shift
            warped, valid = warp_right(right, candidate, right_support=sensor)
            valid &= (
                sensor & (candidate - offset >= cfg.minimum_effective_px) & (abs(shift) <= limit)
            )
            costs.append(_aggregate(_cost(left, warped), valid, cfg.window))
        volume = torch.cat(costs, 1)
        best, index = volume.min(1, keepdim=True)
        peaks = volume.topk(2, dim=1, largest=False).values
        proposed = disparity + index.to(disparity.dtype) - cfg.radius_px
        matched_right, matched_valid = warp_right(right, proposed, right_support=sensor)
        reverse_costs = []
        for shift in shifts:
            # For the same right query u_R, independently search left u_L+j.
            shifted_left, valid = warp_right(
                left, torch.full_like(disparity, -shift), right_support=sensor
            )
            reverse_costs.append(
                _aggregate(_cost(shifted_left, matched_right), valid & matched_valid, cfg.window)
            )
        reverse = torch.cat(reverse_costs, 1)
        reverse_index = reverse.argmin(1, keepdim=True)
        reverse_peaks = reverse.topk(2, dim=1, largest=False).values
        accepted = (
            (best < 1.0)
            & (volume[:, cfg.radius_px : cfg.radius_px + 1] - best >= cfg.minimum_improvement)
            & (peaks[:, 1:2] - peaks[:, :1] >= cfg.uniqueness_margin)
            & ((reverse_index - cfg.radius_px).abs() <= cfg.reverse_tolerance_px)
            & (reverse_peaks[:, 1:2] - reverse_peaks[:, :1] >= cfg.uniqueness_margin)
        )
        return proposed, accepted

    def _median(
        self, batch: DeploymentStereoBatch, disparity: torch.Tensor, limit: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        b, _, h, w = disparity.shape
        gray = batch.left.float().mean(1, keepdim=True)
        neighbors = F.unfold(disparity.float(), cfg.window, padding=cfg.window // 2).reshape(
            b, -1, h, w
        )
        colors = F.unfold(gray, cfg.window, padding=cfg.window // 2).reshape_as(neighbors)
        valid = (
            F.unfold(batch.support.float(), cfg.window, padding=cfg.window // 2).reshape_as(
                neighbors
            )
            > 0.5
        )
        valid &= (colors - gray).abs() <= cfg.color_distance
        valid[:, cfg.window**2 // 2] = False  # do not validate an outlier with itself
        count = valid.sum(1, keepdim=True)
        ordered = neighbors.masked_fill(~valid, float("inf")).sort(1).values
        index = ((count - 1).clamp_min(0) // 2).long()
        median = ordered.gather(1, index)
        median = torch.where(count > 0, median, disparity)
        deviations = (neighbors - median).abs().masked_fill(~valid, float("inf")).sort(1).values
        mad = deviations.gather(1, index)
        proposed = disparity + (median - disparity).clamp(min=-limit, max=limit)
        _, fov = warp_right(batch.right, proposed, right_support=batch.support)
        accepted = (
            (count >= cfg.minimum_neighbors)
            & fov
            & (
                (disparity - median).abs()
                > torch.maximum(mad * cfg.mad_multiplier, mad.new_tensor(cfg.outlier_px))
            )
        )
        return proposed, accepted
