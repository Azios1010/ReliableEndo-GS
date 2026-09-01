"""Canonical Laplace disparity likelihood."""

import torch

from reliable_endo_gs.contracts.common import (
    require_bool,
    require_floating,
    require_same_device,
    require_same_dtype,
    require_tensor,
)


def _validate_laplace_inputs(
    predicted_disparity: torch.Tensor,
    target_disparity: torch.Tensor,
    scale_b: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    """Validate coupled likelihood tensors before indexing or reduction."""

    require_tensor("predicted_disparity", predicted_disparity)
    require_tensor("target_disparity", target_disparity)
    require_tensor("scale_b", scale_b)
    require_tensor("valid_mask", valid_mask)
    if predicted_disparity.shape != target_disparity.shape:
        raise ValueError("predicted_disparity and target_disparity must share shape")
    if scale_b.shape != predicted_disparity.shape or valid_mask.shape != predicted_disparity.shape:
        raise ValueError(
            "predicted_disparity, target_disparity, scale_b, and valid_mask must share shape"
        )
    require_floating("predicted_disparity", predicted_disparity)
    require_floating("target_disparity", target_disparity)
    require_floating("scale_b", scale_b)
    require_bool("valid_mask", valid_mask)
    require_same_dtype(
        "target_disparity", target_disparity, "predicted_disparity", predicted_disparity
    )
    require_same_dtype("scale_b", scale_b, "predicted_disparity", predicted_disparity)
    require_same_device(
        "target_disparity", target_disparity, "predicted_disparity", predicted_disparity
    )
    require_same_device("scale_b", scale_b, "predicted_disparity", predicted_disparity)
    require_same_device("valid_mask", valid_mask, "predicted_disparity", predicted_disparity)


def laplace_nll(
    predicted_disparity: torch.Tensor,
    target_disparity: torch.Tensor,
    scale_b: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean valid Laplace NLL ``|d*-d|/b + log(b)``.

    Invalid values are removed before reduction.  Empty masks raise instead of
    silently returning a zero scientific loss.
    """

    _validate_laplace_inputs(predicted_disparity, target_disparity, scale_b, valid_mask)
    values = scale_b[valid_mask]
    if values.numel() == 0:
        raise ValueError("laplace_nll requires at least one valid pixel")
    if not bool(torch.isfinite(values).all()) or not bool((values > 0).all()):
        raise ValueError("scale_b must be finite and strictly positive where valid")
    predicted = predicted_disparity[valid_mask]
    target = target_disparity[valid_mask]
    if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("predicted_disparity and target_disparity must be finite where valid")
    residual = (target - predicted).abs()
    return (residual / values + values.log()).mean()
