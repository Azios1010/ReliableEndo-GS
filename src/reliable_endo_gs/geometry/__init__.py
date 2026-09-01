"""Canonical calibrated stereo geometry and first-order uncertainty propagation."""

from reliable_endo_gs.geometry.backprojection import (
    BackprojectionDiagnostics,
    BackprojectionResult,
    RayResult,
    backproject_depth,
    pixel_rays,
)
from reliable_endo_gs.geometry.conventions import (
    GEOMETRY_SCHEMA_VERSION,
    GeometryConvention,
    GeometryProvenance,
)
from reliable_endo_gs.geometry.covariance_propagation import (
    CenterCovarianceDiagnostics,
    CenterCovarianceResult,
    CenterJacobianResult,
    center_covariance_from_calibrated_sigma,
    center_covariance_from_disparity,
    center_jacobian_wrt_disparity,
    propagate_disparity_covariance,
)
from reliable_endo_gs.geometry.disparity import (
    DepthJacobianResult,
    DepthResult,
    DepthUncertaintyDiagnostics,
    DepthUncertaintyResult,
    DisparityResult,
    ScalarValidityDiagnostics,
    depth_jacobian_wrt_disparity,
    depth_to_disparity,
    depth_uncertainty_from_disparity,
    disparity_to_depth,
    focal_length_x_from_intrinsics,
)
from reliable_endo_gs.geometry.pixels import pixel_grid

__all__ = [
    "BackprojectionDiagnostics",
    "BackprojectionResult",
    "CenterCovarianceDiagnostics",
    "CenterCovarianceResult",
    "CenterJacobianResult",
    "DepthJacobianResult",
    "DepthResult",
    "DepthUncertaintyDiagnostics",
    "DepthUncertaintyResult",
    "DisparityResult",
    "GeometryConvention",
    "GeometryProvenance",
    "GEOMETRY_SCHEMA_VERSION",
    "RayResult",
    "ScalarValidityDiagnostics",
    "backproject_depth",
    "center_covariance_from_disparity",
    "center_covariance_from_calibrated_sigma",
    "center_jacobian_wrt_disparity",
    "depth_jacobian_wrt_disparity",
    "depth_to_disparity",
    "depth_uncertainty_from_disparity",
    "disparity_to_depth",
    "focal_length_x_from_intrinsics",
    "pixel_grid",
    "pixel_rays",
    "propagate_disparity_covariance",
]
