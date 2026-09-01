"""Explicit coordinate, unit, and pixel conventions for stereo geometry."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GeometryConvention:
    """Declare the meaning of all geometry tensors handled by this package.

    This initial canonical path is intentionally narrow: centers and
    covariances are in the left-camera frame, lengths are metres, and image
    indices denote pixel centres. No routine in :mod:`reliable_endo_gs.geometry`
    applies an implicit frame or unit conversion.
    """

    frame: str = "left_camera"
    axis_convention: str = "x_right_y_down_z_forward"
    length_unit: str = "m"
    focal_length_unit: str = "px"
    disparity_unit: str = "px"
    pixel_coordinate_convention: str = "integer_pixel_centers"

    def __post_init__(self) -> None:
        if self.frame != "left_camera":
            raise ValueError("frame must be 'left_camera' for the canonical geometry path")
        if self.axis_convention != "x_right_y_down_z_forward":
            raise ValueError("axis_convention must be 'x_right_y_down_z_forward'")
        if self.length_unit != "m":
            raise ValueError("length_unit must be 'm'; unit conversion is not implicit")
        if self.focal_length_unit != "px":
            raise ValueError("focal_length_unit must be 'px'")
        if self.disparity_unit != "px":
            raise ValueError("disparity_unit must be 'px'")
        if self.pixel_coordinate_convention != "integer_pixel_centers":
            raise ValueError("pixel_coordinate_convention must be 'integer_pixel_centers'")


GEOMETRY_SCHEMA_VERSION = "geometry_center_covariance.v1"


@dataclass(frozen=True, slots=True)
class GeometryProvenance:
    """Identity carried from a calibrated sigma provider into geometry outputs.

    The boundary is intentionally provider-neutral: both proxy and learned
    providers expose the same four attributes, while geometry records only the
    finalized ``sigma_d`` identity and the resolved camera convention.
    """

    provider_id: str
    calibration_id: str
    convention: GeometryConvention
    geometry_schema_version: str = GEOMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise ValueError("provider_id must be a non-empty string")
        if not isinstance(self.calibration_id, str) or not self.calibration_id:
            raise ValueError("calibration_id must be a non-empty string")
        if not isinstance(self.geometry_schema_version, str) or not self.geometry_schema_version:
            raise ValueError("geometry_schema_version must be a non-empty string")
        if not isinstance(self.convention, GeometryConvention):
            raise TypeError("convention must be a GeometryConvention")
