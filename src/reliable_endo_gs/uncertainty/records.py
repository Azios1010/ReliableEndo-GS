"""Typed records that keep raw, calibrated, and oracle signals separate."""

from dataclasses import dataclass

import torch

from reliable_endo_gs.contracts.common import (
    require_bool,
    require_floating,
    require_same_device,
    require_same_dtype,
    require_shape,
    require_tensor,
)


class UncertaintyLeakageError(ValueError):
    """Raised when a non-deployable oracle record is used as inference input."""


def _validate_map_pair(values: torch.Tensor, valid_mask: torch.Tensor, name: str) -> None:
    require_tensor(name, values)
    require_tensor("valid_mask", valid_mask)
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B, 1, H, W]; got {tuple(values.shape)}")
    require_floating(name, values)
    require_shape("valid_mask", valid_mask, tuple(values.shape))
    require_bool("valid_mask", valid_mask)
    require_same_device("valid_mask", valid_mask, name, values)
    valid_values = values[valid_mask]
    if not bool(torch.isfinite(valid_values).all()):
        raise ValueError(f"{name} must be finite where valid_mask is true")


@dataclass(frozen=True, slots=True)
class RawUncertaintyScore:
    """A rank-only deployable score, deliberately not a probabilistic sigma.

    ``score`` and ``valid_mask`` have shape ``[B, 1, H, W]``.  Scores may be
    arbitrary real values because their interpretation is estimator-owned.
    """

    score: torch.Tensor
    valid_mask: torch.Tensor
    estimator_id: str
    version: str = "1"

    def __post_init__(self) -> None:
        _validate_map_pair(self.score, self.valid_mask, "score")
        if not self.estimator_id:
            raise ValueError("estimator_id must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")


@dataclass(frozen=True, slots=True)
class CalibratedDisparitySigma:
    """Canonical deployable disparity standard deviation ``sigma_d``.

    ``sigma_d`` is in disparity pixels and represents a standard deviation,
    rather than a raw score or Laplace scale.  Valid entries must be finite
    and strictly positive; invalid entries are ignored by consumers.
    """

    sigma_d: torch.Tensor
    valid_mask: torch.Tensor
    provider_id: str
    calibration_id: str

    def __post_init__(self) -> None:
        _validate_map_pair(self.sigma_d, self.valid_mask, "sigma_d")
        valid_sigma = self.sigma_d[self.valid_mask]
        if not bool(torch.isfinite(valid_sigma).all()) or not bool((valid_sigma > 0).all()):
            raise ValueError(
                "sigma_d must be finite and strictly positive where valid_mask is true"
            )
        if not self.provider_id or not self.calibration_id:
            raise ValueError("provider_id and calibration_id must be non-empty")


def require_matching_maps(
    first_name: str,
    first: torch.Tensor,
    second_name: str,
    second: torch.Tensor,
) -> None:
    """Require two floating uncertainty maps to share shape, device, and dtype."""

    require_shape(second_name, second, tuple(first.shape))
    require_floating(second_name, second)
    require_same_device(second_name, second, first_name, first)
    require_same_dtype(second_name, second, first_name, first)
