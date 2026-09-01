"""Validation-only temperature calibration for deployable disparity sigma."""

import math
from dataclasses import dataclass

import torch

from reliable_endo_gs.contracts.common import (
    require_bool,
    require_floating,
    require_same_device,
    require_same_dtype,
    require_tensor,
)
from reliable_endo_gs.uncertainty.records import CalibratedDisparitySigma


@dataclass(frozen=True, slots=True)
class TemperatureCalibrator:
    """Immutable scalar map ``sigma'_d = tau sigma_d`` fit on validation only."""

    temperature: float
    fit_split_identity: str
    method: str = "laplace_nll_v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and strictly positive")
        if not self.fit_split_identity:
            raise ValueError("fit_split_identity must be non-empty")

    @classmethod
    def fit(
        cls,
        raw_sigma_d: torch.Tensor,
        absolute_error: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        split_role: str,
        split_identity: str,
    ) -> "TemperatureCalibrator":
        """Fit the closed-form Laplace-NLL temperature on validation pixels.

        Since ``b = sigma_d / sqrt(2)``, the optimum is
        ``tau = sqrt(2) mean(|error| / raw_sigma_d)``. Test split fitting is
        rejected; callers must preserve the immutable validation identity.
        """

        if split_role != "validation":
            raise ValueError("temperature calibration may be fit on validation data only")
        if not isinstance(split_identity, str) or not split_identity:
            raise ValueError("split_identity must be a non-empty string")
        require_tensor("raw_sigma_d", raw_sigma_d)
        require_tensor("absolute_error", absolute_error)
        require_tensor("valid_mask", valid_mask)
        if raw_sigma_d.ndim != 4 or raw_sigma_d.shape[1] != 1:
            raise ValueError(
                f"raw_sigma_d must have shape [B, 1, H, W]; got {tuple(raw_sigma_d.shape)}"
            )
        if raw_sigma_d.shape != absolute_error.shape or valid_mask.shape != raw_sigma_d.shape:
            raise ValueError("raw_sigma_d, absolute_error, and valid_mask must share shape")
        require_floating("raw_sigma_d", raw_sigma_d)
        require_floating("absolute_error", absolute_error)
        require_bool("valid_mask", valid_mask)
        require_same_dtype("absolute_error", absolute_error, "raw_sigma_d", raw_sigma_d)
        require_same_device("absolute_error", absolute_error, "raw_sigma_d", raw_sigma_d)
        require_same_device("valid_mask", valid_mask, "raw_sigma_d", raw_sigma_d)
        sigma = raw_sigma_d[valid_mask]
        error = absolute_error[valid_mask]
        if sigma.numel() == 0:
            raise ValueError("temperature calibration requires at least one valid pixel")
        if not bool(torch.isfinite(sigma).all()) or not bool((sigma > 0).all()):
            raise ValueError("raw_sigma_d must be finite and strictly positive where valid")
        if not bool(torch.isfinite(error).all()) or not bool((error >= 0).all()):
            raise ValueError("absolute_error must be finite and non-negative where valid")
        ratio_mean = float((error / sigma).mean().item())
        # A zero-residual validation set has no scale information. Preserve
        # the raw positive sigma with a neutral temperature instead of the
        # invalid non-positive optimum tau=0.
        temperature = 1.0 if ratio_mean == 0.0 else math.sqrt(2.0) * ratio_mean
        return cls(temperature=temperature, fit_split_identity=split_identity)

    def apply(
        self,
        raw_sigma_d: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        provider_id: str,
    ) -> CalibratedDisparitySigma:
        """Apply a frozen calibration without accepting ground-truth targets."""

        require_tensor("raw_sigma_d", raw_sigma_d)
        require_tensor("valid_mask", valid_mask)
        if raw_sigma_d.ndim != 4 or raw_sigma_d.shape[1] != 1:
            raise ValueError(
                f"raw_sigma_d must have shape [B, 1, H, W]; got {tuple(raw_sigma_d.shape)}"
            )
        if valid_mask.shape != raw_sigma_d.shape:
            raise ValueError("raw_sigma_d and valid_mask must share shape")
        require_floating("raw_sigma_d", raw_sigma_d)
        require_bool("valid_mask", valid_mask)
        require_same_device("valid_mask", valid_mask, "raw_sigma_d", raw_sigma_d)
        valid_sigma = raw_sigma_d[valid_mask]
        if valid_sigma.numel() == 0:
            raise ValueError("temperature calibration requires at least one valid pixel")
        if not bool(torch.isfinite(valid_sigma).all()) or not bool((valid_sigma > 0).all()):
            raise ValueError("raw_sigma_d must be finite and strictly positive where valid")
        return CalibratedDisparitySigma(
            sigma_d=raw_sigma_d * self.temperature,
            valid_mask=valid_mask,
            provider_id=provider_id,
            calibration_id=f"{self.method}:{self.fit_split_identity}",
        )
