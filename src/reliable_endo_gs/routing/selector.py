"""Small utility MLP and a conservative one-intervention allocator.

The initial experiment selects at most one region/action per frame. Predicted
costs constrain allocation, not actual wall time; measured overruns must be
reported. No oracle outcomes or GT enter feature extraction or selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn

from reliable_endo_gs.actions.repair import Action, RepairState, primitive_core
from reliable_endo_gs.regions.grid import Region

FEATURE_NAMES = (
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "left_error_r",
    "left_error_g",
    "left_error_b",
    "right_error_r",
    "right_error_g",
    "right_error_b",
    "disparity_over_width",
    "disparity_std_over_width",
    "support_fraction",
    "opacity",
    "scale_x_m",
    "scale_y_m",
    "scale_z_m",
    "core_x",
    "core_y",
)


def region_features(state: RepairState, regions: tuple[Region, ...]) -> torch.Tensor:
    """Detached [R,18] deployment-only, pre-action summaries."""
    if state.base.render_right is None:
        raise ValueError("selector features require both pre-action renders")
    rows = []
    height, width = state.batch.left.shape[-2:]
    for region in regions:
        core = region.mask(state.batch.left)
        mask = core & state.support

        def mean(image: torch.Tensor, selected: torch.Tensor = mask) -> torch.Tensor:
            values = image[0, :, selected[0, 0]]
            return (
                values.float().mean(dim=1)
                if values.shape[1]
                else image.new_zeros(image.shape[1]).float()
            )

        disparity = state.base.target.disparity[mask].float() / width
        slots = primitive_core(state, region)
        opacity = state.base.gaussians.opacities[slots].float()
        scales = state.base.gaussians.scales[slots].float()
        y0, x0, y1, x1 = region.core
        geometry = state.batch.left.new_tensor(
            [
                float(disparity.mean()) if disparity.numel() else 0.0,
                float(disparity.std(unbiased=False)) if disparity.numel() else 0.0,
                float(mask.sum()) / int(core.sum()),
                float(opacity.mean()) if opacity.numel() else 0.0,
            ]
        ).float()
        scale = scales.mean(dim=0) if scales.numel() else geometry.new_zeros(3)
        xy = geometry.new_tensor([(x0 + x1) / (2 * width), (y0 + y1) / (2 * height)])
        rows.append(
            torch.cat(
                (
                    mean(state.batch.left),
                    mean((state.base.render_left.image - state.batch.left).abs()),
                    mean((state.base.render_right.image - state.batch.right).abs()),
                    geometry,
                    scale,
                    xy,
                )
            )
        )
    if not rows:
        raise ValueError("selector requires at least one region")
    return torch.stack(rows).detach()


class ActionSelector(nn.Module):
    """18 -> width -> width -> two predicted quality-minus-cost gains.

    STOP utility is canonically zero, so it cannot drift through regression.
    The two outputs correspond to STEREO and GAUSSIAN.
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("selector width must be positive")
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return [R,3] utilities in action enum order."""
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("selector features must be [R,18]")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("selector features must be finite")
        gains = self.network(features.float())
        return torch.cat((torch.zeros_like(gains[:, :1]), gains), dim=1)


@dataclass(frozen=True, slots=True)
class Decision:
    region_index: int | None
    action: Action
    predicted_utility: float
    estimated_ms: float


def select_action(
    utilities: torch.Tensor, costs_ms: torch.Tensor, valid: torch.Tensor, budget_ms: float
) -> Decision:
    """Choose one positive utility/cost intervention, else exact STOP."""
    if not isfinite(budget_ms) or budget_ms < 0:
        raise ValueError("budget_ms must be finite and nonnegative")
    if utilities.ndim != 2 or utilities.shape[1] != 3 or costs_ms.shape != utilities.shape:
        raise ValueError("utilities and costs must be [R,3]")
    if valid.shape != utilities.shape[:1] or valid.dtype != torch.bool:
        raise ValueError("valid must be bool [R]")
    if not bool(torch.isfinite(utilities).all() & torch.isfinite(costs_ms).all()):
        raise ValueError("utility/cost must be finite")
    if not bool((costs_ms >= 0).all()) or not bool((utilities[:, 0] == 0).all()):
        raise ValueError("costs cannot be negative and STOP utility must be zero")
    allowed = valid[:, None] & (costs_ms > 0) & (costs_ms <= budget_ms) & (utilities > 0)
    allowed[:, 0] = False
    if not bool(allowed.any()):
        return Decision(None, Action.STOP, 0.0, 0.0)
    scores = torch.where(allowed, utilities / costs_ms.clamp_min(1e-12), -torch.inf)
    index = int(scores.reshape(-1).argmax())
    row, action = divmod(index, 3)
    return Decision(
        row, Action(action), float(utilities[row, action]), float(costs_ms[row, action])
    )
