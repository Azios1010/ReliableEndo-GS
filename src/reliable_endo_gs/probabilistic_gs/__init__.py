"""Probabilistic Gaussian representation primitives for Plan 07."""

from reliable_endo_gs.probabilistic_gs.marginalization import (
    MARGINALIZATION_SCHEMA_VERSION,
    REPRESENTATION_SCHEMA_VERSION,
    MarginalizationDiagnostics,
    MarginalizationResult,
    ProbabilisticGaussianRepresentation,
    RepresentationProvenance,
    RepresentationVariant,
    VariantSpec,
    build_probabilistic_representation,
    marginalize,
    rank1_kappa,
    variant_specs,
)
from reliable_endo_gs.probabilistic_gs.stability import (
    DEFAULT_COVARIANCE_STABILITY_POLICY,
    CovarianceStabilityPolicy,
    covariance_outer,
    stable_logdet_spd,
)
from reliable_endo_gs.probabilistic_gs.support_covariance import (
    SURFACE_COVARIANCE_SCHEMA_VERSION,
    SurfaceCovarianceDiagnostics,
    SurfaceCovarianceResult,
    build_surface_covariance,
    quaternion_to_rotation_matrix,
    surface_covariance,
)

__all__ = [
    "MARGINALIZATION_SCHEMA_VERSION",
    "REPRESENTATION_SCHEMA_VERSION",
    "SURFACE_COVARIANCE_SCHEMA_VERSION",
    "CovarianceStabilityPolicy",
    "DEFAULT_COVARIANCE_STABILITY_POLICY",
    "MarginalizationDiagnostics",
    "MarginalizationResult",
    "ProbabilisticGaussianRepresentation",
    "RepresentationProvenance",
    "RepresentationVariant",
    "SurfaceCovarianceDiagnostics",
    "SurfaceCovarianceResult",
    "VariantSpec",
    "build_probabilistic_representation",
    "build_surface_covariance",
    "covariance_outer",
    "marginalize",
    "quaternion_to_rotation_matrix",
    "rank1_kappa",
    "stable_logdet_spd",
    "surface_covariance",
    "variant_specs",
]
