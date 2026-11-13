"""Automatic deployment candidate sampler; no tissue/GT segmentation claim."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from reliable_endo_gs.actions.repair import RepairState
from reliable_endo_gs.config.selector_study import SelectorStudyConfig
from reliable_endo_gs.contracts.regions import TargetROI
from reliable_endo_gs.regions.grid import grid_regions
from reliable_endo_gs.regions.masks import MaskRegion
from reliable_endo_gs.routing.masks import MaskProposalConfig


@dataclass(frozen=True, slots=True)
class SupportGridConfig:
    schema: str = "automatic_support_grid.v1"
    tile_hw: tuple[int, int] = (32, 32)

    def __post_init__(self) -> None:
        if (
            self.schema != "automatic_support_grid.v1"
            or len(self.tile_hw) != 2
            or any(type(v) is not int or v < 1 for v in self.tile_hw)
        ):
            raise ValueError("invalid automatic support grid")
        object.__setattr__(self, "tile_hw", tuple(self.tile_hw))


class AutomaticSupportGrid:
    """Target-mask provider and spatially spread bounded support-grid proposals.

    The input ROI is the complete image rectangle, not predicted tissue. Actual
    writes intersect pre-action sensor/prediction support [1,1,H,W]. A fixed
    spatial sampler is shared by actions, collection, inference and profiling.
    """

    def __init__(self, config: SupportGridConfig, limits: MaskProposalConfig) -> None:
        self.config, self.limits = config, limits
        self.protocol = {
            **asdict(config),
            "limits": asdict(limits),
            "roi": "full_input_rectangle",
            "support": "deployment_sensor_and_finite_prediction",
            "sampling": "nearest_unused_nonempty_tile_to_uniform_2D_strata_midpoints",
            "ground_truth_used": False,
            "producer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        self.identity = hashlib.sha256(
            json.dumps(self.protocol, sort_keys=True).encode()
        ).hexdigest()

    def require_samples(self, sample_ids: Sequence[str], image_hw: tuple[int, int]) -> None:
        """No per-sample files; the deterministic producer covers any input grid."""
        if not sample_ids or min(image_hw) < 1:
            raise ValueError("automatic support grid needs sample identities and a valid grid")

    def load(self, sample_id: str, image_hw: tuple[int, int], device: torch.device) -> TargetROI:
        """Full deployment input rectangle [1,1,H,W], independent of supervision."""
        self.require_samples((sample_id,), image_hw)
        return TargetROI(
            torch.ones((1, 1, *image_hw), dtype=torch.bool, device=device),
            "input_roi",
            "automatic_full_input_rectangle;producer_sha256=" + self.identity,
        )

    @torch.no_grad()
    def propose(self, state: RepairState, *, halo: int) -> tuple[MaskRegion, ...]:
        """Bounded immutable sparse masks; never fill support holes or score GT.

        Candidates exceeding a per-patch bound are rejected, not cropped. The
        total selected pixel/primitive limits apply across the whole frame.
        """
        h, w = state.batch.left.shape[-2:]
        support = state.support
        if support.shape != (1, 1, h, w) or support.dtype != torch.bool:
            raise ValueError("automatic support must be bool [1,1,H,W]")
        slots = state.base.prior.source_index[0]
        valid_slots = state.base.gaussians.valid_mask[0] & (slots >= 0)
        counts = torch.zeros(h * w, dtype=torch.long, device=support.device)
        counts.scatter_add_(0, slots[valid_slots], torch.ones_like(slots[valid_slots]))
        tiles = grid_regions((h, w), self.config.tile_hw, halo)
        # Two pooled grids cross to CPU once; never scan the full image for
        # each candidate or synchronize CUDA for every tile.
        pixels_grid = F.avg_pool2d(
            support.float(),
            self.config.tile_hw,
            stride=self.config.tile_hw,
            ceil_mode=True,
            divisor_override=1,
        )
        primitives_grid = F.avg_pool2d(
            counts.reshape(1, 1, h, w).float() * support,
            self.config.tile_hw,
            stride=self.config.tile_hw,
            ceil_mode=True,
            divisor_override=1,
        )
        compact = torch.stack((pixels_grid.flatten(), primitives_grid.flatten()), 1).cpu().tolist()
        candidates = []
        for tile, (pixels, primitives) in zip(tiles, compact, strict=True):
            y0, x0, y1, x1 = tile.halo
            if (
                not pixels
                or not 0 < primitives <= self.limits.max_primitives
                or pixels > self.limits.max_pixels
                or (y1 - y0) * (x1 - x0) > self.limits.max_read_pixels
            ):
                continue
            cy0, cx0, cy1, cx1 = tile.core
            candidates.append(
                (tile, int(pixels), int(primitives), (cy0 + cy1) / (2 * h), (cx0 + cx1) / (2 * w))
            )
        n = min(self.limits.max_regions, len(candidates))
        result: list[MaskRegion] = []
        remaining_pixels, remaining_primitives = self.limits.max_pixels, self.limits.max_primitives
        strata_rows = max(1, math.isqrt(n))
        strata_columns = math.ceil(n / strata_rows)
        for i in range(n):
            ay = (i // strata_columns + 0.5) / strata_rows
            ax = (i % strata_columns + 0.5) / strata_columns
            nearest = min(
                range(len(candidates)),
                key=lambda j: ((candidates[j][3] - ay) ** 2 + (candidates[j][4] - ax) ** 2, j),
            )
            tile, pixels, primitives, _, _ = candidates.pop(nearest)
            if pixels > remaining_pixels or primitives > remaining_primitives:
                continue
            y0, x0, y1, x1 = tile.core
            coords = support[0, 0, y0:y1, x0:x1].nonzero()
            indices = tuple(((coords[:, 0] + y0) * w + coords[:, 1] + x0).cpu().tolist())
            result.append(MaskRegion.from_indices(len(result), (h, w), indices, halo))
            remaining_pixels -= len(indices)
            remaining_primitives -= primitives
        return tuple(result)


def automatic_study_regions(
    config: SelectorStudyConfig,
) -> tuple[MaskProposalConfig, AutomaticSupportGrid]:
    """Build the single candidate owner without data/GT access."""
    limits = MaskProposalConfig(
        max_regions=config.max_regions,
        max_pixels=config.max_pixels,
        max_primitives=config.max_primitives,
        max_read_pixels=config.max_read_pixels,
    )
    return limits, AutomaticSupportGrid(SupportGridConfig(tile_hw=config.grid_tile_hw), limits)
