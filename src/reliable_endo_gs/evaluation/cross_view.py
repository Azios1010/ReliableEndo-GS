"""Development-only cross-view evaluation with explicit mask coverage."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from reliable_endo_gs.contracts import RenderOutput
from reliable_endo_gs.evaluation.rendering import MetricResult, evaluate_rendering


@dataclass(frozen=True, slots=True)
class MaskCoverage:
    """Coverage summary for one view's supervision mask."""

    valid_count: int
    total_count: int

    @property
    def fraction(self) -> float:
        return self.valid_count / self.total_count if self.total_count else 0.0


@dataclass(frozen=True, slots=True)
class CrossViewEvaluation:
    """Left, masked-right, and full-right metrics plus mask coverage."""

    left: dict[str, MetricResult]
    right_masked: dict[str, MetricResult]
    right_full: dict[str, MetricResult]
    right_coverage: MaskCoverage


def summarize_mask_coverage(mask: torch.Tensor) -> MaskCoverage:
    """Return valid/total counts for a boolean ``[B,1,H,W]`` mask."""

    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B, 1, H, W]")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have dtype torch.bool")
    return MaskCoverage(int(mask.sum().item()), int(mask.numel()))


def evaluate_cross_view(
    left_render: RenderOutput | torch.Tensor,
    left_target: torch.Tensor,
    right_render: RenderOutput | torch.Tensor,
    right_target: torch.Tensor,
    right_mask: torch.Tensor,
    *,
    left_mask: torch.Tensor | None = None,
    data_range: float = 1.0,
) -> CrossViewEvaluation:
    """Evaluate masked and full right-view metrics without mutating state."""

    right_image = right_render.image if isinstance(right_render, RenderOutput) else right_render
    left_image = left_render.image if isinstance(left_render, RenderOutput) else left_render
    coverage = summarize_mask_coverage(right_mask)
    return CrossViewEvaluation(
        left=evaluate_rendering(left_image, left_target, left_mask, data_range=data_range),
        right_masked=evaluate_rendering(
            right_image, right_target, right_mask, data_range=data_range
        ),
        right_full=evaluate_rendering(right_image, right_target, None, data_range=data_range),
        right_coverage=coverage,
    )


evaluate_cross_view_metrics = evaluate_cross_view


__all__ = [
    "CrossViewEvaluation",
    "MaskCoverage",
    "evaluate_cross_view",
    "evaluate_cross_view_metrics",
    "summarize_mask_coverage",
]
