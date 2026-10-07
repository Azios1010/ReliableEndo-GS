"""Offline fixed-denominator metric changes for the experimental outcome table."""

from __future__ import annotations

import math
from typing import Any

import torch

from reliable_endo_gs.actions.repair import ActionResult, RepairState
from reliable_endo_gs.evaluation.depth import depth_mae, depth_rmse
from reliable_endo_gs.geometry.rectified import reconstruct_rectified
from reliable_endo_gs.regions.grid import Region


def fixed_depth_metrics(
    disparity: torch.Tensor,
    gt_depth: torch.Tensor,
    fixed: torch.Tensor,
    state: RepairState,
) -> dict[str, float | None]:
    """MAE/RMSE/p95 in mm on immutable GT+base support [1,1,H,W].

    Lost support is an unavailable outcome, never a smaller denominator.
    torch.quantile uses linear interpolation, explicitly pinned here.
    """
    geometry = reconstruct_rectified(
        disparity, state.batch.camera, state.batch.support, min_effective_disparity=1e-6
    )
    if not bool(fixed.any()) or bool((fixed & ~geometry.depth.valid_mask).any()):
        return {"mae_mm": None, "rmse_mm": None, "p95_mm": None}
    predicted, target = geometry.depth.depth * 1000, gt_depth * 1000
    errors = (predicted - target).abs()[fixed]
    if not bool(torch.isfinite(errors).all()):
        return {"mae_mm": None, "rmse_mm": None, "p95_mm": None}
    return {
        "mae_mm": depth_mae(predicted, target, fixed).value,
        "rmse_mm": depth_rmse(predicted, target, fixed).value,
        "p95_mm": float(errors.float().quantile(0.95, interpolation="linear")),
    }


def depth_snapshot(
    outcome: ActionResult,
    stop: ActionResult,
    state: RepairState,
    gt_depth: torch.Tensor,
    gt_valid: torch.Tensor,
    region: Region,
) -> dict[str, dict[str, float | None]]:
    """Before/after must call this with the same STOP reference and region."""
    base = reconstruct_rectified(
        stop.disparity, state.batch.camera, state.support, min_effective_disparity=1e-6
    )
    fixed = gt_valid & base.depth.valid_mask & state.support
    return {
        "global": fixed_depth_metrics(outcome.disparity, gt_depth, fixed, state),
        "local": fixed_depth_metrics(
            outcome.disparity, gt_depth, fixed & region.mask(state.batch.left), state
        ),
    }


def metric_changes(
    before: dict[str, Any],
    after: dict[str, Any],
    depth_before: dict[str, dict[str, float | None]],
    depth_after: dict[str, dict[str, float | None]],
    outside_before: float,
    outside_after: float,
    local_before: float,
    local_after: float,
) -> list[float] | None:
    """Return COMPONENT_NAMES order; missing evidence cannot certify safety."""
    pairs: list[tuple[float | None, float | None]] = [
        (depth_before["global"]["mae_mm"], depth_after["global"]["mae_mm"]),
        (depth_before["local"]["mae_mm"], depth_after["local"]["mae_mm"]),
        (depth_before["global"]["rmse_mm"], depth_after["global"]["rmse_mm"]),
        (depth_before["local"]["rmse_mm"], depth_after["local"]["rmse_mm"]),
        (depth_before["global"]["p95_mm"], depth_after["global"]["p95_mm"]),
    ]
    # Higher is better for render/coverage; convert into positive improvement.
    pairs.extend(
        (after[key]["value"], before[key]["value"])
        for key in ("left_psnr", "right_psnr", "left_ssim", "right_ssim")
    )
    pairs.append((outside_before, outside_after))
    pairs.append((local_before, local_after))
    pairs.append((after["geometry_coverage"]["value"], before["geometry_coverage"]["value"]))
    if any(a is None or b is None for a, b in pairs):
        return None
    values = [float(a) - float(b) for a, b in pairs if a is not None and b is not None]
    return values if all(math.isfinite(v) for v in values) else None
