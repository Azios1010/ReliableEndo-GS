"""Pre-action, deployable uncertainty proxies from audited stereo evidence."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional

from reliable_endo_gs.contracts import StereoPrediction
from reliable_endo_gs.contracts.common import (
    require_bool,
    require_floating,
    require_same_device,
    require_same_dtype,
    require_tensor,
)
from reliable_endo_gs.uncertainty.records import RawUncertaintyScore


def _iteration_tensor(prediction: StereoPrediction) -> torch.Tensor:
    """Return iterations as ``[B, K, H, W]`` without inventing missing evidence."""

    iterations = prediction.disparity_iterations
    if iterations is None:
        raise ValueError("proxy requires disparity_iterations, but the prediction exposes none")
    if isinstance(iterations, torch.Tensor):
        if iterations.shape[1] < 2:
            raise ValueError("proxy requires at least two disparity iterations")
        return iterations
    if not isinstance(iterations, Sequence) or len(iterations) < 2:
        raise ValueError("proxy requires at least two disparity iterations")
    return torch.cat(tuple(iterations), dim=1)


class FinalUpdateMagnitudeProxy:
    """Raw score ``|d_K - d_{K-1}|`` from two final recurrent updates."""

    estimator_id = "final_update_magnitude"
    version = "1"

    def predict(self, prediction: StereoPrediction) -> RawUncertaintyScore:
        iterations = _iteration_tensor(prediction)
        score = (iterations[:, -1:] - iterations[:, -2:-1]).abs()
        return RawUncertaintyScore(score, prediction.valid_mask, self.estimator_id, self.version)


class IterationDisagreementProxy:
    """Raw standard deviation over a validation-configured final iteration window."""

    estimator_id = "iteration_disagreement"
    version = "1"

    def __init__(self, window: int = 4) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        self.window = window

    def predict(self, prediction: StereoPrediction) -> RawUncertaintyScore:
        iterations = _iteration_tensor(prediction)
        if iterations.shape[1] < self.window:
            raise ValueError(
                f"iteration_disagreement requires {self.window} iterations; got {iterations.shape[1]}"
            )
        score = iterations[:, -self.window :].std(dim=1, unbiased=False).unsqueeze(1)
        return RawUncertaintyScore(score, prediction.valid_mask, self.estimator_id, self.version)


def left_right_consistency_score(
    disparity_left_to_right: torch.Tensor,
    disparity_right_to_left: torch.Tensor,
    left_valid_mask: torch.Tensor,
    right_valid_mask: torch.Tensor,
) -> RawUncertaintyScore:
    """Return ``|d_lr(x) + d_rl(x - d_lr(x))|`` with out-of-view masking.

    Both disparities must use their native pixel coordinate's signed direction:
    left-to-right is sampled at ``x - d_lr`` in the right image and right-to-
    left is therefore added.  ``grid_sample`` bilinear interpolation is the
    declared numerical approximation; right validity uses nearest sampling.
    """

    require_tensor("disparity_left_to_right", disparity_left_to_right)
    require_tensor("disparity_right_to_left", disparity_right_to_left)
    require_tensor("left_valid_mask", left_valid_mask)
    require_tensor("right_valid_mask", right_valid_mask)
    if disparity_left_to_right.shape != disparity_right_to_left.shape:
        raise ValueError("left and right disparity maps must have the same shape")
    if disparity_left_to_right.ndim != 4 or disparity_left_to_right.shape[1] != 1:
        raise ValueError("disparities must have shape [B, 1, H, W]")
    require_floating("disparity_left_to_right", disparity_left_to_right)
    require_floating("disparity_right_to_left", disparity_right_to_left)
    require_same_dtype(
        "disparity_right_to_left",
        disparity_right_to_left,
        "disparity_left_to_right",
        disparity_left_to_right,
    )
    require_same_device(
        "disparity_right_to_left",
        disparity_right_to_left,
        "disparity_left_to_right",
        disparity_left_to_right,
    )
    if (
        left_valid_mask.shape != disparity_left_to_right.shape
        or right_valid_mask.shape != disparity_left_to_right.shape
    ):
        raise ValueError("valid masks must match the disparity shape")
    require_bool("left_valid_mask", left_valid_mask)
    require_bool("right_valid_mask", right_valid_mask)
    require_same_device(
        "left_valid_mask", left_valid_mask, "disparity_left_to_right", disparity_left_to_right
    )
    require_same_device(
        "right_valid_mask", right_valid_mask, "disparity_right_to_left", disparity_right_to_left
    )
    if not bool(torch.isfinite(disparity_left_to_right[left_valid_mask]).all()):
        raise ValueError("disparity_left_to_right must be finite where left_valid_mask is true")
    if not bool(torch.isfinite(disparity_right_to_left[right_valid_mask]).all()):
        raise ValueError("disparity_right_to_left must be finite where right_valid_mask is true")

    batch, _, height, width = disparity_left_to_right.shape
    x = torch.arange(
        width, device=disparity_left_to_right.device, dtype=disparity_left_to_right.dtype
    )
    y = torch.arange(
        height, device=disparity_left_to_right.device, dtype=disparity_left_to_right.dtype
    )
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    sample_x = grid_x.unsqueeze(0) - disparity_left_to_right[:, 0]
    sample_y = grid_y.expand(batch, -1, -1)
    normalizer_x = max(width - 1, 1)
    normalizer_y = max(height - 1, 1)
    grid = torch.stack(
        (2.0 * sample_x / normalizer_x - 1.0, 2.0 * sample_y / normalizer_y - 1.0), dim=-1
    )
    sampled_right = functional.grid_sample(
        disparity_right_to_left, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    sampled_valid = (
        functional.grid_sample(
            right_valid_mask.to(disparity_left_to_right.dtype),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        >= 0.5
    )
    in_view = (sample_x >= 0) & (sample_x <= width - 1)
    valid = left_valid_mask & sampled_valid & in_view.unsqueeze(1)
    score = torch.where(
        valid, (disparity_left_to_right + sampled_right).abs(), torch.zeros_like(sampled_right)
    )
    return RawUncertaintyScore(score, valid, "left_right_consistency", "1")


def photometric_residual_score(
    left_image: torch.Tensor,
    right_image: torch.Tensor,
    disparity_left_to_right: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> RawUncertaintyScore:
    """Compute normalized same-view photometric residual from predicted disparity.

    Images must use the pinned upstream ``[-1, 1]`` image space.  Positive
    left-reference disparity samples the rectified right image at
    ``x_right = x_left - d_left``.  The returned score is mean absolute RGB
    residual in normalized image units (nominal range ``[0, 2]``), and its
    validity mask excludes non-finite/negative disparity and out-of-view
    samples.  No ground-truth tensor is consumed.
    """

    for name, value in (
        ("left_image", left_image),
        ("right_image", right_image),
        ("disparity_left_to_right", disparity_left_to_right),
    ):
        require_tensor(name, value)
        require_floating(name, value)
    if left_image.ndim != 4 or left_image.shape[1] != 3:
        raise ValueError("images must have shape [B, 3, H, W]")
    if right_image.shape != left_image.shape:
        raise ValueError("left_image and right_image must have identical shapes")
    if disparity_left_to_right.shape != (
        left_image.shape[0],
        1,
        left_image.shape[2],
        left_image.shape[3],
    ):
        raise ValueError("disparity must have shape [B, 1, H, W] matching the images")
    require_same_device("right_image", right_image, "left_image", left_image)
    require_same_device(
        "disparity_left_to_right", disparity_left_to_right, "left_image", left_image
    )
    require_same_dtype("right_image", right_image, "left_image", left_image)
    require_same_dtype("disparity_left_to_right", disparity_left_to_right, "left_image", left_image)
    if valid_mask is not None:
        require_tensor("valid_mask", valid_mask)
        require_bool("valid_mask", valid_mask)
        if valid_mask.shape != disparity_left_to_right.shape:
            raise ValueError("valid_mask must match disparity shape")
        require_same_device("valid_mask", valid_mask, "left_image", left_image)

    batch, _, height, width = disparity_left_to_right.shape
    x = torch.arange(width, device=left_image.device, dtype=left_image.dtype)
    y = torch.arange(height, device=left_image.device, dtype=left_image.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    sample_x = grid_x.unsqueeze(0) - disparity_left_to_right[:, 0]
    sample_y = grid_y.expand(batch, -1, -1)
    normalizer_x = max(width - 1, 1)
    normalizer_y = max(height - 1, 1)
    grid = torch.stack(
        (
            2.0 * sample_x / normalizer_x - 1.0,
            2.0 * sample_y / normalizer_y - 1.0,
        ),
        dim=-1,
    )
    warped_right = functional.grid_sample(
        right_image,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    finite_inputs = torch.isfinite(left_image).all(dim=1, keepdim=True) & torch.isfinite(
        right_image
    ).all(dim=1, keepdim=True)
    valid = (
        valid_mask
        if valid_mask is not None
        else torch.ones_like(disparity_left_to_right, dtype=torch.bool)
    )
    valid = (
        valid
        & torch.isfinite(disparity_left_to_right)
        & (disparity_left_to_right > 0.0)
        & (sample_x.unsqueeze(1) >= 0.0)
        & (sample_x.unsqueeze(1) <= width - 1)
        & finite_inputs
    )
    residual = (left_image - warped_right).abs().mean(dim=1, keepdim=True)
    residual = torch.where(valid, residual, torch.zeros_like(residual))
    return RawUncertaintyScore(residual, valid, "photometric_residual", "1")
