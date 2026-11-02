"""Constant-disparity-shift test: complementary crops move the disparity range while resolution and scale stay fixed.

Left crop offset ``a`` and right crop offset ``b`` satisfy a + b = 32 and shift D = a - b, so the new ground-truth disparity is
d' = d - D. A model that uses the stereo signal follows the shift (slope of mean prediction on mean truth near 1); a model that
has collapsed to a prior does not (slope near 0). Same definition as the validation-sequence study of the development phase.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample

SHIFTS: tuple[int, ...] = (-24, -16, -8, 0, 8, 16, 24)


def shifted_batch(
    sample: ScaredA00Sample,
    gt_disparity: torch.Tensor,
    gt_valid: torch.Tensor,
    shift: int,
    width: int = 608,
) -> tuple[DeploymentStereoBatch, torch.Tensor, torch.Tensor]:
    """Returns (batch, gt_disparity', gt_valid'); disparity [1,1,H,W], width columns."""
    if shift % 2 or abs(shift) > 32 - 8:
        raise ValueError("shift must be even and at most 24 in magnitude")
    a, b = (32 + shift) // 2, (32 - shift) // 2
    batch, camera = sample.batch, sample.batch.camera
    k_left, k_right = camera.K_L.clone(), camera.K_R.clone()
    k_left[:, 0, 2] -= a
    k_right[:, 0, 2] -= b
    new_camera = RectifiedStereoCamera(k_left, k_right, camera.T_R_from_L.clone(), True)
    left = batch.left[..., a : a + width].contiguous()
    right = batch.right[..., b : b + width].contiguous()
    support = batch.support[..., a : a + width].contiguous()
    disparity = gt_disparity[..., a : a + width] - float(shift)
    columns = torch.arange(width, device=disparity.device, dtype=disparity.dtype).view(
        1, 1, 1, width
    )
    valid = gt_valid[..., a : a + width] & (disparity >= 1.0) & (columns - disparity >= 0)
    return DeploymentStereoBatch(left, right, new_camera, support), disparity, valid


def shift_stats(
    pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, support: torch.Tensor
) -> dict[str, Any] | None:
    mask = valid & support & torch.isfinite(pred)
    count = int(mask.sum())
    if not count:
        return None
    error = (pred - gt)[mask]
    return {
        "n": count,
        "mean_pred": float(pred[mask].mean()),
        "mean_gt": float(gt[mask].mean()),
        "bias": float(error.mean()),
        "epe": float(error.abs().mean()),
    }


def shift_slope(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Least-squares slope of the per-shift mean prediction on the per-shift mean truth (rows with n > 0)."""
    by_shift: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("n"):
            by_shift.setdefault(int(row["shift"]), []).append((row["mean_pred"], row["mean_gt"]))
    shifts = sorted(by_shift)
    if len(shifts) < 3:
        raise ValueError("need at least three shifts with data")
    pred = np.array([np.mean([p for p, _ in by_shift[s]]) for s in shifts])
    truth = np.array([np.mean([t for _, t in by_shift[s]]) for s in shifts])
    slope = float(np.polyfit(truth, pred, 1)[0])
    return {
        "slope": slope,
        "shifts": shifts,
        "mean_pred": pred.tolist(),
        "mean_gt": truth.tolist(),
        "reading": "follows the shift"
        if slope >= 0.8
        else ("insensitive" if abs(slope) <= 0.2 else "intermediate"),
    }
