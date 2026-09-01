"""Numerical policies shared by the Plan 07 covariance calculations."""

from dataclasses import dataclass
from typing import cast

import torch


@dataclass(frozen=True, slots=True)
class CovarianceStabilityPolicy:
    """Explicit numerical tolerances for covariance validation.

    The policy does not add diagonal jitter.  Invalid primitives are reported
    through masks by the representation builders instead of being converted to
    apparently certain or artificially full-rank Gaussians.
    """

    quaternion_norm_epsilon: float = 1.0e-12
    symmetry_tolerance: float = 1.0e-6
    psd_tolerance: float = 1.0e-8
    kappa_range_tolerance: float = 1.0e-6


DEFAULT_COVARIANCE_STABILITY_POLICY = CovarianceStabilityPolicy()


def stable_logdet_spd(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(sign, logdet)`` for ``[..., 3, 3]`` covariance tensors.

    ``torch.linalg.slogdet`` avoids explicit matrix inversion and preserves
    gradients through the valid path.  Positive-definiteness and validity are
    checked by the caller because this helper is intentionally tensor-only.
    """

    if covariance.ndim < 2 or tuple(covariance.shape[-2:]) != (3, 3):
        raise ValueError(
            f"covariance must have trailing shape [3, 3]; got {tuple(covariance.shape)}"
        )
    return cast(tuple[torch.Tensor, torch.Tensor], torch.linalg.slogdet(covariance))


def covariance_outer(vector: torch.Tensor) -> torch.Tensor:
    """Return ``v v^T`` for ``[..., 3]`` vectors without changing dtype/device."""

    if vector.ndim < 1 or vector.shape[-1] != 3:
        raise ValueError(f"vector must have trailing shape [3]; got {tuple(vector.shape)}")
    return vector.unsqueeze(-1) * vector.unsqueeze(-2)
