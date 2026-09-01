"""Explicit boundary between a probabilistic representation and a renderer.

This module defines tensor and semantic requirements only.  It deliberately
does not call, emulate, or import the pinned CUDA renderer.
"""

from dataclasses import dataclass
from enum import Enum

import torch

from reliable_endo_gs.contracts.common import (
    require_bool,
    require_floating,
    require_shape,
    require_tensor,
)
from reliable_endo_gs.probabilistic_gs.marginalization import (
    ProbabilisticGaussianRepresentation,
    RepresentationProvenance,
    RepresentationVariant,
)


class CovarianceRequest(str, Enum):
    """Which covariance a future renderer adapter must consume."""

    NONE = "none"
    SURFACE = "surface"
    EFFECTIVE = "effective"


@dataclass(frozen=True, slots=True)
class RendererRequest:
    """Validated renderer input without a backend-specific Gaussian object.

    Tensor shapes are ``means3d [B,N,3]``, ``colors [B,N,C]`` when present,
    ``opacities [B,N,1]``, ``valid_mask [B,N]``, and ``covariance [B,N,3,3]``
    for a non-``NONE`` request.  ``covariance_source`` identifies the exact
    semantic field and prevents a renderer adapter from guessing between
    intrinsic support and effective marginalized covariance.
    """

    means3d: torch.Tensor
    opacities: torch.Tensor
    valid_mask: torch.Tensor
    covariance_source: CovarianceRequest
    covariance: torch.Tensor | None
    variant: RepresentationVariant
    frame: str
    length_unit: str
    colors: torch.Tensor | None = None
    provenance: RepresentationProvenance | None = None

    def __post_init__(self) -> None:
        means = require_tensor("means3d", self.means3d)
        if means.ndim != 3 or means.shape[-1] != 3:
            raise ValueError("means3d must have shape [B, N, 3]")
        require_floating("means3d", means)
        expected = (means.shape[0], means.shape[1])
        opacity = require_tensor("opacities", self.opacities)
        require_shape("opacities", opacity, expected + (1,))
        require_floating("opacities", opacity)
        valid = require_tensor("valid_mask", self.valid_mask)
        require_shape("valid_mask", valid, expected)
        require_bool("valid_mask", valid)
        if valid.device != means.device:
            raise ValueError("valid_mask must share device with means3d")
        if bool((valid & ~torch.isfinite(means).all(dim=-1)).any()):
            raise ValueError("means3d must be finite on valid primitives")
        for name, tensor in (("opacities", opacity),):
            if tensor.dtype != means.dtype or tensor.device != means.device:
                raise ValueError(f"{name} must share dtype/device with means3d")
            alpha_ok = torch.isfinite(tensor).all(dim=-1)
            alpha_ok = alpha_ok & (tensor >= 0).all(dim=-1) & (tensor <= 1).all(dim=-1)
            if bool((valid & ~alpha_ok).any()):
                raise ValueError(f"{name} must be finite and in [0, 1] on valid primitives")
        if not frame_is_explicit(self.frame, self.length_unit):
            raise ValueError("frame and length_unit must be explicit non-empty strings")
        source = CovarianceRequest(self.covariance_source)
        object.__setattr__(self, "covariance_source", source)
        variant = RepresentationVariant(self.variant)
        object.__setattr__(self, "variant", variant)
        if source is CovarianceRequest.NONE:
            if self.covariance is not None:
                raise ValueError("covariance must be None for covariance_source='none'")
        else:
            covariance = require_tensor("covariance", self.covariance)
            require_shape("covariance", covariance, expected + (3, 3))
            require_floating("covariance", covariance)
            if covariance.dtype != means.dtype or covariance.device != means.device:
                raise ValueError("covariance must share dtype/device with means3d")
            finite = torch.isfinite(covariance).all(dim=-1).all(dim=-1)
            if bool((valid & ~finite).any()):
                raise ValueError("covariance must be finite on valid primitives")
            symmetric = (covariance - covariance.transpose(-1, -2)).abs().amax(
                dim=(-1, -2)
            ) <= 1.0e-6
            clean = torch.where(
                torch.isfinite(covariance), covariance, torch.zeros_like(covariance)
            )
            minimum_eigenvalue = torch.linalg.eigvalsh(0.5 * (clean + clean.transpose(-1, -2)))[
                ..., 0
            ]
            if bool((valid & (~symmetric | (minimum_eigenvalue <= 0))).any()):
                raise ValueError(
                    "covariance must be symmetric positive definite on valid primitives"
                )
        if self.colors is not None:
            colors = require_tensor("colors", self.colors)
            if colors.ndim != 3 or colors.shape[:2] != expected:
                raise ValueError("colors must have shape [B, N, C]")
            require_floating("colors", colors)
            if colors.dtype != means.dtype or colors.device != means.device:
                raise ValueError("colors must share dtype/device with means3d")
        if self.provenance is not None:
            if self.provenance.frame and self.provenance.frame != self.frame:
                raise ValueError("provenance frame must match renderer frame")
            if self.provenance.length_unit and self.provenance.length_unit != self.length_unit:
                raise ValueError("provenance length_unit must match renderer length_unit")

    @classmethod
    def from_representation(
        cls,
        representation: ProbabilisticGaussianRepresentation,
        *,
        covariance_source: CovarianceRequest,
    ) -> "RendererRequest":
        """Select an explicit covariance field from a Plan 07 representation."""

        source = CovarianceRequest(covariance_source)
        covariance: torch.Tensor | None
        if source is CovarianceRequest.NONE:
            covariance = None
        elif source is CovarianceRequest.SURFACE:
            covariance = representation.cov_surface
        else:
            covariance = representation.cov_effective
        if source is not CovarianceRequest.NONE and covariance is None:
            raise ValueError(f"representation does not provide {source.value} covariance")
        opacity = (
            representation.base_opacity
            if source is not CovarianceRequest.EFFECTIVE
            else representation.effective_opacity
        )
        return cls(
            means3d=representation.means3d,
            opacities=opacity,
            valid_mask=representation.valid_mask,
            covariance_source=source,
            covariance=covariance,
            variant=representation.variant,
            frame=representation.frame,
            length_unit=representation.length_unit,
            colors=representation.colors,
            provenance=representation.provenance,
        )


def frame_is_explicit(frame: str, length_unit: str) -> bool:
    """Return whether renderer metadata has explicit frame and unit labels."""

    return bool(frame and length_unit)


__all__ = ["CovarianceRequest", "RendererRequest", "frame_is_explicit"]
