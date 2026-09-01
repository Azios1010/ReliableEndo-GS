"""Diagnostic-only uncertainty targets derived from ground truth."""

from dataclasses import dataclass

import torch

from reliable_endo_gs.uncertainty.records import UncertaintyLeakageError, _validate_map_pair


@dataclass(frozen=True, slots=True)
class OracleUncertaintyTarget:
    """Absolute disparity residual, explicitly prohibited from inference.

    The record is intentionally not a ``RawUncertaintyScore`` or
    ``CalibratedDisparitySigma``: those types are deployable while this target
    depends on ground truth and is only valid for training or diagnostics.
    """

    absolute_error: torch.Tensor
    valid_mask: torch.Tensor
    target_id: str = "oracle_absolute_disparity_error"
    non_deployable: bool = True

    def __post_init__(self) -> None:
        _validate_map_pair(self.absolute_error, self.valid_mask, "absolute_error")
        valid_values = self.absolute_error[self.valid_mask]
        if not bool(torch.isfinite(valid_values).all()) or not bool((valid_values >= 0).all()):
            raise ValueError("oracle absolute_error must be finite and non-negative where valid")
        if not isinstance(self.target_id, str) or not self.target_id:
            raise ValueError("target_id must be a non-empty string")
        if self.non_deployable is not True:
            raise ValueError("oracle target must remain marked non_deployable")

    def as_inference_feature(self) -> torch.Tensor:
        """Always reject accidental oracle-to-inference conversion."""

        raise UncertaintyLeakageError(
            "ground-truth oracle targets cannot be used for deployable inference"
        )


def build_oracle_target(
    predicted_disparity: torch.Tensor,
    ground_truth_disparity: torch.Tensor,
    prediction_valid_mask: torch.Tensor,
    ground_truth_valid_mask: torch.Tensor,
) -> OracleUncertaintyTarget:
    """Build valid absolute disparity residuals without assigning invalid error.

    Ground truth is an explicit argument so callers cannot obtain this target
    from a deployable provider API.  Invalid pixels are excluded rather than
    converted into arbitrary large residuals.
    """

    _validate_map_pair(predicted_disparity, prediction_valid_mask, "predicted_disparity")
    _validate_map_pair(ground_truth_disparity, ground_truth_valid_mask, "ground_truth_disparity")
    if ground_truth_disparity.shape != predicted_disparity.shape:
        raise ValueError("ground_truth_disparity must match predicted_disparity")
    if ground_truth_disparity.dtype != predicted_disparity.dtype:
        raise TypeError("ground_truth_disparity must match predicted_disparity dtype")
    if ground_truth_disparity.device != predicted_disparity.device:
        raise ValueError("ground_truth_disparity must match predicted_disparity device")
    valid = prediction_valid_mask & ground_truth_valid_mask
    return OracleUncertaintyTarget((predicted_disparity - ground_truth_disparity).abs(), valid)
