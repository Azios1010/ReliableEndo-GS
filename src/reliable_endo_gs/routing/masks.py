"""Bounded mask proposals from existing stereo and Gaussian/render evidence.

Proposals locate candidates; they are NOT action utility, calibrated risk or
foreground segmentation. A learned action selector still evaluates candidates.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import yaml
from torch.nn import functional as F

from reliable_endo_gs.actions.repair import RepairState
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.geometry.stereo_matching import correspondence_error, warp_right
from reliable_endo_gs.regions.grid import Region
from reliable_endo_gs.regions.masks import MaskRegion, TargetROI
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES, balanced_features


@dataclass(frozen=True, slots=True)
class MaskProposalConfig:
    """Explicit proposal scales and hard work bounds; require TRAIN calibration.

    RGB/warp scales use mean absolute RGB error in 0..1. Entropy and margin
    proxies already use 0..1. max_read_pixels bounds each left halo bounding
    box, not the full right feature pass, geometry rebuild or rasterization.
    """

    schema: str = "target_mask_proposals.v1"
    evidence: Literal["stereo", "gaussian", "combined"] = "combined"
    feature_evidence: Literal["stereo", "gaussian", "combined"] = "combined"
    warp_scale: float = 0.1
    render_scale: float = 0.1
    ambiguity_weight: float = 0.25
    threshold: float = 0.75
    cell_size: int = 8
    max_regions: int = 4
    max_pixels: int = 4096
    max_primitives: int = 1024
    max_read_pixels: int = 65536

    def __post_init__(self) -> None:
        if self.schema != "target_mask_proposals.v1" or self.evidence not in (
            "stereo",
            "gaussian",
            "combined",
        ):
            raise ValueError("unsupported mask proposal protocol/evidence")
        if self.feature_evidence not in ("stereo", "gaussian", "combined"):
            raise ValueError("unsupported mask feature ablation")
        for name in ("warp_scale", "render_scale", "ambiguity_weight", "threshold"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} must be finite")
        if min(self.warp_scale, self.render_scale) <= 0:
            raise ValueError("mask error scales must be positive")
        if not 0 <= self.threshold <= 1 or not 0 <= self.ambiguity_weight <= 1:
            raise ValueError("mask threshold and ambiguity weight must be in 0..1")
        for name in ("cell_size", "max_regions", "max_pixels", "max_primitives", "max_read_pixels"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def load_mask_config(path: Path) -> MaskProposalConfig:
    """Strict parsing: unknown fields cannot silently change mask identity."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - set(MaskProposalConfig.__dataclass_fields__):
        raise ValueError("mask config must contain only declared fields")
    return MaskProposalConfig(**raw)


@torch.no_grad()
def mask_features(
    state: RepairState,
    regions: tuple[Region, ...],
    config: InterventionConfig,
    proposal: MaskProposalConfig,
    *,
    scale_unit_m: float,
) -> torch.Tensor:
    """Canonical balanced local/global context with explicit source ablations.

    Common RGB, ROI position, support and camera calibration remain available.
    Ablation removes source columns, not the cost of producing the base model.
    Constant excluded columns are fitted inactive by TRAIN-only normalization.
    """
    features = balanced_features(state, regions, config, scale_unit_m=scale_unit_m)
    stereo_names = {
        "disparity_over_width",
        "disparity_std_over_width",
        "warp_error_mean",
        "warp_error_std",
        "warp_valid_fraction",
        "local_matching_margin",
        "local_matching_entropy_proxy",
        "disparity_relative_jump",
        "matching_candidate_support_fraction",
        "global_warp_error",
        "global_warp_support",
        "global_disparity_mean",
        "global_disparity_std",
        "depth_mean_m",
        "depth_std_m",
        "cost_entropy",
        "cost_margin",
        "feature_energy",
        "matching_evidence_available",
        *(f"patch_warp_{i}" for i in range(4)),
    }
    gaussian_names = {
        *(f"{view}_error_{c}" for view in ("left", "right") for c in ("r", "g", "b")),
        "opacity",
        "scale_x_m",
        "scale_y_m",
        "scale_z_m",
        "global_left_error",
        "global_right_error",
        "log_scale_x",
        "log_scale_y",
        "log_scale_z",
        "log_anisotropy",
        "primitive_density",
        "normal_x",
        "normal_y",
        "normal_z",
        "visible_fraction",
        "radius_over_diagonal",
        "visibility_available",
        *(f"patch_render_error_{i}" for i in range(4)),
    }
    excluded = stereo_names if proposal.feature_evidence == "gaussian" else gaussian_names
    if proposal.feature_evidence != "combined":
        columns = [i for i, name in enumerate(BALANCED_FEATURE_NAMES) if name in excluded]
        features[:, columns] = 0
    return features.detach()


@torch.no_grad()
def candidate_scores(state: RepairState, config: MaskProposalConfig) -> torch.Tensor:
    """Detached [1,1,H,W] proposal proxy using pre-action deployment data.

    Dual source uses max(stereo proxy, render proxy), not a claim that errors
    have equal scientific utility. Missing matching evidence is not confidence.
    Render-right error is sampled at u_R=u_L-d through the canonical sampler.
    """
    if state.batch.left.shape[0] != 1 or state.base.render_right is None:
        raise ValueError("mask proposals require batch one and both pre-action renders")
    disparity = state.base.target.disparity
    warp_error, warp_valid = correspondence_error(
        state.batch.left, state.batch.right, disparity, right_support=state.batch.support
    )
    stereo = (warp_error / config.warp_scale).clamp(0, 1)
    evidence = getattr(state.base, "matching_evidence", {})
    if "cost_entropy" in evidence and "cost_margin" in evidence:
        ambiguity = torch.maximum(evidence["cost_entropy"], 1 - evidence["cost_margin"])
        stereo = torch.maximum(stereo, config.ambiguity_weight * ambiguity.clamp(0, 1))
    stereo = torch.where(warp_valid, stereo, torch.zeros_like(stereo))
    left_error = (state.base.render_left.image - state.batch.left).abs().mean(1, keepdim=True)
    right_error = (state.base.render_right.image - state.batch.right).abs().mean(1, keepdim=True)
    right_error, valid = warp_right(right_error, disparity, right_support=state.batch.support)
    render = torch.maximum(
        left_error, torch.where(valid, right_error, torch.zeros_like(right_error))
    )
    render = (render / config.render_scale).clamp(0, 1)
    result = stereo if config.evidence == "stereo" else render
    if config.evidence == "combined":
        result = torch.maximum(stereo, render)
    return torch.where(
        state.support & torch.isfinite(result), result, torch.zeros_like(result)
    ).detach()


@torch.no_grad()
def propose_masks(
    state: RepairState, target: TargetROI, config: MaskProposalConfig, *, halo: int
) -> tuple[MaskRegion, ...]:
    """Score only target pixels; use coarse 8-connected components as grouping.

    Pooling may group separate nearby errors, but never adds good/background
    pixels to writes. Coarse connectivity crosses old quadrant boundaries.
    Pixel ranking, primitive and read-area budgets bound candidate collection.
    One compact coarse occupancy and bounded selected indices cross to CPU;
    this is still full-grid cheap evidence, not measured sparse acceleration.
    """
    scores = candidate_scores(state, config)
    if target.pixels.shape != scores.shape or target.pixels.device != scores.device:
        raise ValueError("target ROI must match the prediction output grid/device")
    active = target.pixels & state.support & (scores > config.threshold)
    cell = config.cell_size
    occupancy = F.max_pool2d(active.float(), cell, stride=cell, ceil_mode=True)[0, 0]
    ch, cw = occupancy.shape
    pending = set(map(tuple, occupancy.nonzero().cpu().tolist()))
    labels = torch.full((ch, cw), -1, dtype=torch.long)
    component_count = 0
    h, w = scores.shape[-2:]
    while pending:
        seed = min(pending)
        pending.remove(seed)
        cells, queue = [seed], [seed]
        for cy, cx in queue:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbour = (cy + dy, cx + dx)
                    if neighbour in pending:
                        pending.remove(neighbour)
                        queue.append(neighbour)
                        cells.append(neighbour)
        coords = torch.tensor(cells)
        labels[coords[:, 0], coords[:, 1]] = component_count
        component_count += 1
    if not component_count:
        return ()
    # Gather active pixels once, rather than rescanning the full image for
    # every component. Only compact coarse labels/priorities cross to CPU.
    active_indices = active.reshape(-1).nonzero().flatten()
    component_ids = labels.to(scores.device)[
        active_indices // w // cell, active_indices % w // cell
    ]
    priority = scores.reshape(-1)[active_indices]
    maxima = priority.new_zeros(component_count)
    maxima.scatter_reduce_(0, component_ids, priority, reduce="amax")
    priorities = maxima.cpu().tolist()
    component_order = sorted(range(component_count), key=lambda i: (-priorities[i], i))
    slots = state.base.prior.source_index[0]
    valid_slots = state.base.gaussians.valid_mask[0] & (slots >= 0)
    slot_counts = torch.zeros(h * w, device=scores.device, dtype=torch.long)
    slot_counts.scatter_add_(0, slots[valid_slots], torch.ones_like(slots[valid_slots]))
    result: list[MaskRegion] = []
    remaining_pixels, remaining_primitives = config.max_pixels, config.max_primitives
    for component_id in component_order:
        if len(result) == config.max_regions or remaining_pixels <= 0 or remaining_primitives <= 0:
            break
        indices = active_indices[component_ids == component_id]
        ordered = scores.reshape(-1)[indices].argsort(descending=True, stable=True)
        indices = indices[ordered[:remaining_pixels]]
        counts = slot_counts[indices]
        indices = indices[counts.cumsum(0) <= remaining_primitives]
        if not indices.numel() or not bool(slot_counts[indices].sum()):
            continue
        region = MaskRegion.from_indices(len(result), (h, w), tuple(indices.cpu().tolist()), halo)
        y0, x0, y1, x1 = region.halo
        if (y1 - y0) * (x1 - x0) > config.max_read_pixels:
            # Reject oversized read boxes; never fill/crop a mask silently.
            continue
        result.append(region)
        remaining_pixels -= region.pixel_count
        remaining_primitives -= int(slot_counts[indices].sum())
    return tuple(result)
