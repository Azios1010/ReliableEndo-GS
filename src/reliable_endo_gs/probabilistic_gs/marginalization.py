"""Two-covariance Gaussian representation and stable per-primitive marginalization."""

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

import torch

from reliable_endo_gs.contracts.common import (
    require_bool,
    require_floating,
    require_shape,
    require_tensor,
)
from reliable_endo_gs.probabilistic_gs.stability import (
    DEFAULT_COVARIANCE_STABILITY_POLICY,
    CovarianceStabilityPolicy,
    covariance_outer,
    stable_logdet_spd,
)
from reliable_endo_gs.probabilistic_gs.support_covariance import build_surface_covariance

MARGINALIZATION_SCHEMA_VERSION = "gaussian_marginalization.v1"
REPRESENTATION_SCHEMA_VERSION = "probabilistic_gaussian_representation.v1"


class RepresentationVariant(str, Enum):
    """Explicit representation variants supported by the local contract."""

    BASELINE = "baseline"
    COVARIANCE_ONLY = "covariance_only"
    OPACITY_ONLY = "opacity_only"
    FIXED_ISOTROPIC = "fixed_isotropic"
    STEREO_COVARIANCE_CORRECTED = "stereo_covariance_corrected"


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """Development availability and semantics for an ablation identity."""

    variant: RepresentationVariant
    available: bool
    covariance_source: str
    opacity_correction: bool
    note: str


_VARIANT_SPECS: tuple[VariantSpec, ...] = (
    VariantSpec(
        RepresentationVariant.BASELINE,
        True,
        "none",
        False,
        "Existing baseline Gaussian parameters; no uncertainty transformation.",
    ),
    VariantSpec(
        RepresentationVariant.COVARIANCE_ONLY,
        True,
        "effective",
        False,
        "Adds center covariance while retaining base opacity.",
    ),
    VariantSpec(
        RepresentationVariant.OPACITY_ONLY,
        False,
        "none",
        False,
        "Deferred: Plan 07 does not define a standalone opacity-only mapping.",
    ),
    VariantSpec(
        RepresentationVariant.FIXED_ISOTROPIC,
        False,
        "surface",
        False,
        "Deferred until a configured scientific fixed scale is approved.",
    ),
    VariantSpec(
        RepresentationVariant.STEREO_COVARIANCE_CORRECTED,
        True,
        "effective",
        True,
        "Adds observation-derived center covariance and determinant correction.",
    ),
)


def variant_specs() -> tuple[VariantSpec, ...]:
    """Return immutable, explicit development variant specifications."""

    return _VARIANT_SPECS


@dataclass(frozen=True, slots=True)
class RepresentationProvenance:
    """Machine-independent identity for a Plan 07 representation."""

    schema_version: str = REPRESENTATION_SCHEMA_VERSION
    surface_covariance_schema: str = "surface_covariance.v1"
    center_covariance_schema: str = "geometry_center_covariance.v1"
    marginalization_rule: str = "cov_surface_plus_cov_center.v1"
    opacity_correction_rule: str = "determinant_ratio.v1"
    opacity_correction: bool = False
    rank1_optimization: bool = False
    frame: str = ""
    length_unit: str = ""
    note: str = "CPU analytic/contract representation; production renderer parity pending."


@dataclass(frozen=True, slots=True)
class MarginalizationDiagnostics:
    """Counts and numerical decisions from covariance marginalization."""

    valid_count: int
    invalid_count: int
    nonfinite_count: int
    nonsymmetric_count: int
    nonpositive_surface_count: int
    nonpsd_center_count: int
    invalid_opacity_count: int
    range_violation_count: int
    zero_center_count: int
    used_rank1_optimization: bool


@dataclass(frozen=True, slots=True)
class MarginalizationResult:
    """Effective covariance, corrected opacity, and explicit validity."""

    cov_effective: torch.Tensor
    effective_opacity: torch.Tensor
    kappa: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: MarginalizationDiagnostics


@dataclass(frozen=True, slots=True)
class ProbabilisticGaussianRepresentation:
    """Semantically separated Gaussian representation for one batch.

    Tensors use ``[B, N, ...]``: ``means3d`` is ``[B,N,3]``, covariances are
    ``[B,N,3,3]``, opacities are ``[B,N,1]``, and ``valid_mask`` is ``[B,N]``.
    ``cov_surface`` is intrinsic support, ``cov_center`` is Plan 06 center
    uncertainty, and ``cov_effective`` is their marginal sum.  The latter two
    are never substituted for the former.
    """

    SCHEMA_NAME: ClassVar[str] = "probabilistic_gaussian_representation"
    SCHEMA_VERSION: ClassVar[str] = REPRESENTATION_SCHEMA_VERSION

    means3d: torch.Tensor
    base_opacity: torch.Tensor
    effective_opacity: torch.Tensor
    valid_mask: torch.Tensor
    frame: str
    length_unit: str
    variant: RepresentationVariant
    cov_surface: torch.Tensor | None = None
    cov_center: torch.Tensor | None = None
    cov_effective: torch.Tensor | None = None
    colors: torch.Tensor | None = None
    provenance: RepresentationProvenance = field(default_factory=RepresentationProvenance)
    # The native baseline path needs the original upstream parameterization.
    # These are intentionally carried through unchanged; surface covariance
    # construction remains owned by support_covariance.py.
    rotations: torch.Tensor | None = None
    scales: torch.Tensor | None = None

    def __post_init__(self) -> None:
        means = require_tensor("means3d", self.means3d)
        if means.ndim != 3 or means.shape[-1] != 3:
            raise ValueError(f"means3d must have shape [B, N, 3]; got {tuple(means.shape)}")
        require_floating("means3d", means)
        valid = require_tensor("valid_mask", self.valid_mask)
        require_shape("valid_mask", valid, tuple(means.shape[:2]))
        require_bool("valid_mask", valid)
        if valid.device != means.device:
            raise ValueError("valid_mask must share device with means3d")
        if bool((valid & ~torch.isfinite(means).all(dim=-1)).any()):
            raise ValueError("means3d must be finite on valid primitives")
        expected_opacity = (means.shape[0], means.shape[1], 1)
        for name, value in (
            ("base_opacity", self.base_opacity),
            ("effective_opacity", self.effective_opacity),
        ):
            tensor = require_tensor(name, value)
            require_shape(name, tensor, expected_opacity)
            require_floating(name, tensor)
            if tensor.dtype != means.dtype or tensor.device != means.device:
                raise ValueError(f"{name} must share dtype/device with means3d")
            alpha_ok = torch.isfinite(tensor).all(dim=-1)
            alpha_ok = alpha_ok & (tensor >= 0).all(dim=-1) & (tensor <= 1).all(dim=-1)
            if bool((valid & ~alpha_ok).any()):
                raise ValueError(f"{name} must be finite and in [0, 1] on valid primitives")
        if not self.frame or not self.length_unit:
            raise ValueError("frame and length_unit must be explicit non-empty strings")
        variant = RepresentationVariant(self.variant)
        object.__setattr__(self, "variant", variant)
        if self.cov_effective is not None and (self.cov_surface is None or self.cov_center is None):
            raise ValueError("cov_effective requires distinct cov_surface and cov_center")
        for name, covariance in (
            ("cov_surface", self.cov_surface),
            ("cov_center", self.cov_center),
            ("cov_effective", self.cov_effective),
        ):
            if covariance is None:
                continue
            tensor = require_tensor(name, covariance)
            require_shape(name, tensor, (means.shape[0], means.shape[1], 3, 3))
            require_floating(name, tensor)
            if tensor.dtype != means.dtype or tensor.device != means.device:
                raise ValueError(f"{name} must share dtype/device with means3d")
            covariance_finite = torch.isfinite(tensor).all(dim=-1).all(dim=-1)
            if bool((valid & ~covariance_finite).any()):
                raise ValueError(f"{name} must be finite on valid primitives")
        if self.colors is not None:
            colors = require_tensor("colors", self.colors)
            if colors.ndim != 3 or colors.shape[:2] != means.shape[:2]:
                raise ValueError("colors must have shape [B, N, C]")
            require_floating("colors", colors)
            if colors.dtype != means.dtype or colors.device != means.device:
                raise ValueError("colors must share dtype/device with means3d")
        if (self.rotations is None) != (self.scales is None):
            raise ValueError("rotations and scales must be provided together")
        if self.rotations is not None and self.scales is not None:
            rotations = require_tensor("rotations", self.rotations)
            scales = require_tensor("scales", self.scales)
            require_shape("rotations", rotations, (means.shape[0], means.shape[1], 4))
            require_shape("scales", scales, (means.shape[0], means.shape[1], 3))
            require_floating("rotations", rotations)
            require_floating("scales", scales)
            if rotations.dtype != means.dtype or rotations.device != means.device:
                raise ValueError("rotations must share dtype/device with means3d")
            if scales.dtype != means.dtype or scales.device != means.device:
                raise ValueError("scales must share dtype/device with means3d")
        if self.provenance.frame and self.provenance.frame != self.frame:
            raise ValueError("provenance frame must match representation frame")
        if self.provenance.length_unit and self.provenance.length_unit != self.length_unit:
            raise ValueError("provenance length_unit must match representation length_unit")

    @property
    def batch_size(self) -> int:
        """Return ``B``."""

        return int(self.means3d.shape[0])

    @property
    def gaussian_count(self) -> int:
        """Return ``N``."""

        return int(self.means3d.shape[1])


def _validate_covariance_inputs(
    cov_surface: torch.Tensor,
    cov_center: torch.Tensor,
    opacity: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    require_tensor("cov_surface", cov_surface)
    require_tensor("cov_center", cov_center)
    require_tensor("opacity", opacity)
    require_tensor("valid_mask", valid_mask)
    if cov_surface.ndim < 2 or tuple(cov_surface.shape[-2:]) != (3, 3):
        raise ValueError(
            f"cov_surface must have trailing shape [3, 3]; got {tuple(cov_surface.shape)}"
        )
    if cov_center.shape != cov_surface.shape:
        raise ValueError("cov_center must have the exact shape of cov_surface")
    expected_opacity = cov_surface.shape[:-2] + (1,)
    require_shape("opacity", opacity, tuple(expected_opacity))
    require_shape("valid_mask", valid_mask, tuple(cov_surface.shape[:-2]))
    require_floating("cov_surface", cov_surface)
    require_floating("cov_center", cov_center)
    require_floating("opacity", opacity)
    require_bool("valid_mask", valid_mask)
    if cov_center.dtype != cov_surface.dtype or opacity.dtype != cov_surface.dtype:
        raise TypeError("cov_surface, cov_center, and opacity must share dtype")
    if cov_center.device != cov_surface.device or opacity.device != cov_surface.device:
        raise ValueError("cov_surface, cov_center, opacity, and valid_mask must share device")
    if valid_mask.device != cov_surface.device:
        raise ValueError("cov_surface, cov_center, opacity, and valid_mask must share device")


def _covariance_validity(
    covariance: torch.Tensor,
    *,
    psd: bool,
    policy: CovarianceStabilityPolicy,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(covariance).all(dim=-1).all(dim=-1)
    clean = torch.where(torch.isfinite(covariance), covariance, torch.zeros_like(covariance))
    symmetric_error = (clean - clean.transpose(-1, -2)).abs().amax(dim=(-1, -2))
    symmetric = symmetric_error <= policy.symmetry_tolerance
    eigenvalues = torch.linalg.eigvalsh(0.5 * (clean + clean.transpose(-1, -2)))
    if psd:
        spectral = eigenvalues[..., 0] >= -policy.psd_tolerance
    else:
        spectral = eigenvalues[..., 0] > policy.psd_tolerance
    return finite & symmetric & spectral, finite, symmetric


def rank1_kappa(
    cov_surface: torch.Tensor,
    center_vector: torch.Tensor,
) -> torch.Tensor:
    """Compute ``(1 + v.T solve(Sigma_surf,v))**-1/2`` for ``[...,3]`` ``v``."""

    require_tensor("cov_surface", cov_surface)
    require_tensor("center_vector", center_vector)
    if tuple(cov_surface.shape[-2:]) != (3, 3):
        raise ValueError("cov_surface must have trailing shape [3, 3]")
    if center_vector.shape != cov_surface.shape[:-1]:
        raise ValueError("center_vector must have shape cov_surface.shape[:-1]")
    if not bool(torch.isfinite(cov_surface).all()) or not bool(torch.isfinite(center_vector).all()):
        raise ValueError("rank1_kappa inputs must be finite")
    eigenvalues = torch.linalg.eigvalsh(0.5 * (cov_surface + cov_surface.transpose(-1, -2)))
    if bool((eigenvalues[..., 0] <= 0).any()):
        raise ValueError("rank1_kappa requires positive-definite cov_surface")
    solved = torch.linalg.solve(cov_surface, center_vector.unsqueeze(-1)).squeeze(-1)
    quadratic = (center_vector * solved).sum(dim=-1)
    if bool((quadratic < 0).any()):
        raise ValueError("rank1 quadratic form must be non-negative")
    return torch.exp(-0.5 * torch.log1p(quadratic))


def marginalize(
    cov_surface: torch.Tensor,
    cov_center: torch.Tensor,
    opacity: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    surface_frame: str,
    center_frame: str,
    surface_unit: str,
    center_unit: str,
    policy: CovarianceStabilityPolicy = DEFAULT_COVARIANCE_STABILITY_POLICY,
    rank1_vector: torch.Tensor | None = None,
    use_rank1_optimization: bool = False,
) -> MarginalizationResult:
    """Return ``Sigma_eff``, ``kappa``, and corrected opacity.

    Covariances use ``[B,N,3,3]`` (or leading-dimension equivalent), opacity
    uses ``[B,N,1]``, and the mask uses ``[B,N]``.  Explicit frame and length
    unit strings are required and must match; no unresolved camera/unit bridge
    is inferred here.  Invalid primitives are masked and returned as NaN.
    """

    _validate_covariance_inputs(cov_surface, cov_center, opacity, valid_mask)
    if not surface_frame or not center_frame or not surface_unit or not center_unit:
        raise ValueError("surface/center frame and unit metadata are required")
    if surface_frame != center_frame:
        raise ValueError("surface and center covariance frames must match")
    if surface_unit != center_unit:
        raise ValueError("surface and center covariance length units must match")
    if use_rank1_optimization and rank1_vector is None:
        raise ValueError("rank1_vector is required when use_rank1_optimization=True")
    if rank1_vector is not None:
        require_tensor("rank1_vector", rank1_vector)
        if rank1_vector.shape != cov_surface.shape[:-1]:
            raise ValueError("rank1_vector must have shape cov_surface.shape[:-1]")
        if rank1_vector.dtype != cov_surface.dtype or rank1_vector.device != cov_surface.device:
            raise TypeError("rank1_vector must share dtype/device with cov_surface")

    surface_valid, surface_finite, surface_symmetric = _covariance_validity(
        cov_surface, psd=False, policy=policy
    )
    center_valid, center_finite, center_symmetric = _covariance_validity(
        cov_center, psd=True, policy=policy
    )
    alpha_finite = torch.isfinite(opacity).all(dim=-1)
    alpha_in_range = (opacity >= 0).all(dim=-1) & (opacity <= 1).all(dim=-1)
    alpha_valid = alpha_finite & alpha_in_range
    center_clean = torch.where(torch.isfinite(cov_center), cov_center, torch.zeros_like(cov_center))
    surface_clean = torch.where(
        torch.isfinite(cov_surface), cov_surface, torch.zeros_like(cov_surface)
    )
    zero_center = center_finite & (center_clean.abs().amax(dim=(-1, -2)) == 0)
    effective = surface_clean + center_clean
    effective = torch.where(zero_center.unsqueeze(-1).unsqueeze(-1), surface_clean, effective)
    effective_valid, _, _ = _covariance_validity(effective, psd=False, policy=policy)
    sign_surface, logdet_surface = stable_logdet_spd(surface_clean)
    sign_effective, logdet_effective = stable_logdet_spd(effective)
    determinant_valid = (sign_surface > 0) & (sign_effective > 0)
    base_valid = (
        valid_mask
        & surface_valid
        & center_valid
        & alpha_valid
        & effective_valid
        & determinant_valid
    )

    if use_rank1_optimization:
        assert rank1_vector is not None
        rank1_finite = torch.isfinite(rank1_vector).all(dim=-1)
        rank1_safe = torch.where(
            rank1_finite.unsqueeze(-1), rank1_vector, torch.zeros_like(rank1_vector)
        )
        identity = torch.eye(3, dtype=cov_surface.dtype, device=cov_surface.device)
        safe_surface = torch.where(
            surface_valid.unsqueeze(-1).unsqueeze(-1),
            surface_clean,
            identity.expand_as(surface_clean),
        )
        outer = covariance_outer(rank1_safe)
        difference = (center_clean - outer).abs().amax(dim=(-1, -2))
        if bool((difference[base_valid] > policy.symmetry_tolerance).any()):
            raise ValueError("rank1_vector does not represent cov_center on valid primitives")
        kappa = rank1_kappa(safe_surface, rank1_safe)
        base_valid = base_valid & rank1_finite
    else:
        kappa = torch.exp(0.5 * (logdet_surface - logdet_effective))
    kappa = torch.where(zero_center, torch.ones_like(kappa), kappa)
    finite_kappa = torch.isfinite(kappa) & (kappa > 0)
    range_violation = finite_kappa & (kappa > 1.0 + policy.kappa_range_tolerance)
    range_violation_for_valid = range_violation & base_valid
    if bool(range_violation_for_valid.any()):
        base_valid = base_valid & ~range_violation
    roundoff_above_one = (kappa > 1.0) & ~range_violation
    kappa = torch.where(roundoff_above_one, torch.ones_like(kappa), kappa)
    final_valid = base_valid & finite_kappa & ~range_violation

    output_covariance = torch.where(
        final_valid.unsqueeze(-1).unsqueeze(-1), effective, torch.full_like(effective, float("nan"))
    )
    corrected = opacity * kappa.unsqueeze(-1)
    corrected = torch.clamp(corrected, 0.0, 1.0)
    corrected = torch.where(
        final_valid.unsqueeze(-1), corrected, torch.full_like(corrected, float("nan"))
    )
    output_kappa = torch.where(final_valid, kappa, torch.full_like(kappa, float("nan")))
    diagnostics = MarginalizationDiagnostics(
        valid_count=int(final_valid.sum().item()),
        invalid_count=int((~final_valid).sum().item()),
        nonfinite_count=int((~surface_finite | ~center_finite).sum().item()),
        nonsymmetric_count=int((~surface_symmetric | ~center_symmetric).sum().item()),
        nonpositive_surface_count=int((~surface_valid).sum().item()),
        nonpsd_center_count=int((~center_valid).sum().item()),
        invalid_opacity_count=int((~alpha_valid).sum().item()),
        range_violation_count=int(range_violation_for_valid.sum().item()),
        zero_center_count=int((zero_center & valid_mask).sum().item()),
        used_rank1_optimization=use_rank1_optimization,
    )
    return MarginalizationResult(
        output_covariance, corrected, output_kappa, final_valid, diagnostics
    )


def build_probabilistic_representation(
    means3d: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    cov_center: torch.Tensor | None,
    valid_mask: torch.Tensor,
    *,
    frame: str,
    length_unit: str,
    variant: RepresentationVariant | str = RepresentationVariant.STEREO_COVARIANCE_CORRECTED,
    colors: torch.Tensor | None = None,
    policy: CovarianceStabilityPolicy = DEFAULT_COVARIANCE_STABILITY_POLICY,
    rank1_vector: torch.Tensor | None = None,
    use_rank1_optimization: bool = False,
) -> ProbabilisticGaussianRepresentation:
    """Build a representation from baseline parameters and Plan06 covariance.

    ``means3d``/``cov_center`` use the same explicit camera frame and metric
    unit.  The baseline variant intentionally retains no derived covariance;
    unsupported opacity-only and fixed-isotropic variants raise rather than
    inventing an experimental mapping.
    """

    require_tensor("means3d", means3d)
    require_tensor("rotations", rotations)
    require_tensor("scales", scales)
    require_tensor("opacities", opacities)
    require_tensor("valid_mask", valid_mask)
    if means3d.ndim != 3 or means3d.shape[-1] != 3:
        raise ValueError("means3d must have shape [B, N, 3]")
    expected = (means3d.shape[0], means3d.shape[1])
    if (
        rotations.shape != expected + (4,)
        or scales.shape != expected + (3,)
        or opacities.shape != expected + (1,)
    ):
        raise ValueError("rotations/scales/opacities do not match means3d primitive dimensions")
    require_shape("valid_mask", valid_mask, expected)
    require_bool("valid_mask", valid_mask)
    for name, tensor in (("rotations", rotations), ("scales", scales), ("opacities", opacities)):
        if tensor.dtype != means3d.dtype or tensor.device != means3d.device:
            raise TypeError(f"{name} must share dtype/device with means3d")
    if valid_mask.device != means3d.device:
        raise ValueError("valid_mask must share device with means3d")
    selected = RepresentationVariant(variant)
    if selected in (RepresentationVariant.OPACITY_ONLY, RepresentationVariant.FIXED_ISOTROPIC):
        raise NotImplementedError(
            f"{selected.value} is deferred because its semantics are not defined"
        )
    if not frame or not length_unit:
        raise ValueError("frame and length_unit must be explicit")
    if selected is RepresentationVariant.BASELINE:
        return ProbabilisticGaussianRepresentation(
            means3d=means3d,
            base_opacity=opacities,
            effective_opacity=opacities,
            valid_mask=valid_mask,
            frame=frame,
            length_unit=length_unit,
            variant=selected,
            colors=colors,
            rotations=rotations,
            scales=scales,
            provenance=RepresentationProvenance(
                frame=frame,
                length_unit=length_unit,
                marginalization_rule="not_applied",
                opacity_correction_rule="not_applied",
            ),
        )
    if cov_center is None:
        raise ValueError(f"{selected.value} requires cov_center")
    surface = build_surface_covariance(rotations, scales, valid_mask=valid_mask, policy=policy)
    result = marginalize(
        surface.cov_surface,
        cov_center,
        opacities,
        valid_mask & surface.valid_mask,
        surface_frame=frame,
        center_frame=frame,
        surface_unit=length_unit,
        center_unit=length_unit,
        policy=policy,
        rank1_vector=rank1_vector,
        use_rank1_optimization=use_rank1_optimization,
    )
    return ProbabilisticGaussianRepresentation(
        means3d=means3d,
        base_opacity=opacities,
        effective_opacity=opacities
        if selected is RepresentationVariant.COVARIANCE_ONLY
        else result.effective_opacity,
        valid_mask=result.valid_mask,
        frame=frame,
        length_unit=length_unit,
        variant=selected,
        cov_surface=surface.cov_surface,
        cov_center=cov_center,
        cov_effective=result.cov_effective,
        colors=colors,
        rotations=rotations,
        scales=scales,
        provenance=RepresentationProvenance(
            rank1_optimization=use_rank1_optimization,
            frame=frame,
            length_unit=length_unit,
            opacity_correction=selected is RepresentationVariant.STEREO_COVARIANCE_CORRECTED,
        ),
    )


__all__ = [
    "MARGINALIZATION_SCHEMA_VERSION",
    "REPRESENTATION_SCHEMA_VERSION",
    "MarginalizationDiagnostics",
    "MarginalizationResult",
    "ProbabilisticGaussianRepresentation",
    "RepresentationProvenance",
    "RepresentationVariant",
    "VariantSpec",
    "build_probabilistic_representation",
    "marginalize",
    "rank1_kappa",
    "variant_specs",
]
