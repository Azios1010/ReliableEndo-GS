"""Geometry-derived Gaussian support priors.

This is the shared teacher/student constructor boundary for A03.  It consumes
raw rectified disparity plus first-party calibrated camera geometry, constructs
metric left-camera centres and local surface frames, and only then applies a
deterministic source sampling policy.  No teacher, student, uncertainty head,
repair operator or renderer is imported here.
"""

from dataclasses import dataclass, field
from math import isfinite
from typing import cast

import torch

from reliable_endo_gs.contracts.deployment import RectifiedStereoCamera
from reliable_endo_gs.gaussians.sampling import (
    SamplingPolicy,
    SamplingResult,
    fixed_grid_cells,
    gather_source,
    sample_fixed_grid_static,
    sample_grid,
)
from reliable_endo_gs.gaussians.support_covariance import (
    build_surface_covariance,
    quaternion_to_rotation_matrix,
)
from reliable_endo_gs.geometry.rectified import reconstruct_rectified
from reliable_endo_gs.geometry.surface import (
    SurfaceFramePolicy,
    estimate_surface_frame,
    estimate_surface_frame_on_grid,
)
from reliable_endo_gs.utils.inference_mode import compiled_when_enabled, static_grid_layout_enabled
from reliable_endo_gs.utils.readback import raise_first_failure, read_ints


@dataclass(frozen=True, slots=True)
class GeometryPriorPolicy:
    """A03 construction and sampling policy in explicit metric units."""

    surface: SurfaceFramePolicy = field(default_factory=SurfaceFramePolicy)
    sampling: SamplingPolicy = field(default_factory=SamplingPolicy)
    alpha_valid: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.surface, SurfaceFramePolicy):
            raise TypeError("surface must be a SurfaceFramePolicy")
        if not isinstance(self.sampling, SamplingPolicy):
            raise TypeError("sampling must be a SamplingPolicy")
        if not isinstance(self.alpha_valid, (float, int)) or isinstance(self.alpha_valid, bool):
            raise TypeError("alpha_valid must be a finite number")
        if not isfinite(float(self.alpha_valid)) or not 0 <= self.alpha_valid <= 1:
            raise ValueError("alpha_valid must be finite and in [0,1]")


@dataclass(frozen=True, slots=True)
class GeometryPriorDiagnostics:
    """Counts and rates that make drops/fallbacks visible to callers."""

    eligible_count: int
    selected_count: int
    valid_count: int
    fallback_count: int
    discontinuity_count: int
    tangent_fallback_count: int
    dropped_count: int

    @property
    def fallback_rate(self) -> float:
        """Fraction of valid emitted primitives using conservative support."""

        return self.fallback_count / max(self.valid_count, 1)

    @property
    def drop_rate(self) -> float:
        """Fraction of eligible source pixels omitted by sampling/capping."""

        return self.dropped_count / max(self.eligible_count, 1)


@dataclass(frozen=True, slots=True)
class GeometryPrior:
    """Padded, renderer-ready geometry prior with source correspondence.

    ``mu`` is ``[B,N,3]`` in left-camera metres.  ``R0`` is ``[B,N,3,3]``
    with columns ``[t1,t2,n]`` and ``rotations`` is its normalized ``[w,x,y,z]``
    quaternion representation.  ``S0`` contains positive metric standard
    deviations, ``alpha0`` is an explicit validity/coverage prior, and
    ``cov_surface`` is built by the canonical support-covariance owner.  Padded
    slots are finite placeholders with ``valid_mask=False``; they must be
    omitted by a renderer.
    """

    mu: torch.Tensor
    R0: torch.Tensor
    rotations: torch.Tensor
    S0: torch.Tensor
    alpha0: torch.Tensor
    rgb: torch.Tensor
    valid_mask: torch.Tensor
    source_index: torch.Tensor
    source_xy: torch.Tensor
    coverage: torch.Tensor
    cov_surface: torch.Tensor
    fallback_mask: torch.Tensor
    discontinuity_mask: torch.Tensor
    tangent_fallback_mask: torch.Tensor
    sampling: SamplingResult
    diagnostics: GeometryPriorDiagnostics
    source_features: torch.Tensor | None = None

    def __post_init__(self) -> None:
        mu = self.mu
        if not isinstance(mu, torch.Tensor) or mu.ndim != 3 or mu.shape[-1] != 3:
            raise ValueError("mu must have shape [B,N,3]")
        if not torch.is_floating_point(mu):
            raise TypeError("mu must be floating point")
        batch_size, count, _ = mu.shape
        for name, value, shape in (
            ("R0", self.R0, (batch_size, count, 3, 3)),
            ("rotations", self.rotations, (batch_size, count, 4)),
            ("S0", self.S0, (batch_size, count, 3)),
            ("alpha0", self.alpha0, (batch_size, count, 1)),
            ("rgb", self.rgb, (batch_size, count, 3)),
            ("cov_surface", self.cov_surface, (batch_size, count, 3, 3)),
        ):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must be floating point")
            if value.device != mu.device or value.dtype != mu.dtype:
                raise TypeError(f"{name} must share mu dtype/device")
        if self.valid_mask.shape != (batch_size, count) or self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B,N]")
        if self.source_index.shape != (batch_size, count) or self.source_index.dtype != torch.long:
            raise ValueError("source_index must be long [B,N]")
        if self.source_xy.shape != (batch_size, count, 2):
            raise ValueError("source_xy must be [B,N,2]")
        if self.coverage.shape != (batch_size, count, 1):
            raise ValueError("coverage must be [B,N,1]")
        for name, value in (
            ("valid_mask", self.valid_mask),
            ("source_index", self.source_index),
            ("source_xy", self.source_xy),
            ("coverage", self.coverage),
            ("fallback_mask", self.fallback_mask),
            ("discontinuity_mask", self.discontinuity_mask),
            ("tangent_fallback_mask", self.tangent_fallback_mask),
        ):
            if value.device != mu.device:
                raise ValueError(f"{name} must share mu device")
        for name, value in (
            ("fallback_mask", self.fallback_mask),
            ("discontinuity_mask", self.discontinuity_mask),
            ("tangent_fallback_mask", self.tangent_fallback_mask),
        ):
            if value.shape != (batch_size, count) or value.dtype != torch.bool:
                raise ValueError(f"{name} must be bool [B,N]")
        if self.source_features is not None:
            if self.source_features.ndim != 3 or self.source_features.shape[:2] != (
                batch_size,
                count,
            ):
                raise ValueError("source_features must be [B,N,C]")
            if self.source_features.device != mu.device:
                raise ValueError("source_features must share mu device")

    @property
    def means3d(self) -> torch.Tensor:
        """GaussianField-compatible mean alias."""

        return self.mu

    @property
    def scales(self) -> torch.Tensor:
        """GaussianField-compatible scale alias."""

        return self.S0

    @property
    def opacities(self) -> torch.Tensor:
        """GaussianField-compatible opacity alias."""

        return self.alpha0

    @property
    def colors(self) -> torch.Tensor:
        """GaussianField-compatible RGB alias."""

        return self.rgb

    @property
    def source_coordinates(self) -> torch.Tensor:
        """Explicit correspondence alias."""

        return self.source_xy

    @property
    def coverage_prior(self) -> torch.Tensor:
        """Alias for the sampled validity/coverage prior."""

        return self.coverage


def rotation_matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    """Convert ``[...,3,3]`` rotations to normalized ``[...,4]`` wxyz.

    The dominant-diagonal branch is discrete and deterministic.  All branch
    candidates use clamped denominators so invalid/padded matrices cannot
    create NaNs in the returned finite placeholder; callers should retain a
    validity mask for scientific use.
    """

    if not isinstance(rotation, torch.Tensor):
        raise TypeError("rotation must be a torch.Tensor")
    if rotation.ndim < 2 or tuple(rotation.shape[-2:]) != (3, 3):
        raise ValueError("rotation must have trailing shape [3,3]")
    if not torch.is_floating_point(rotation):
        raise TypeError("rotation must be floating point")
    raise_first_failure(
        [
            (
                torch.isfinite(rotation).all(),
                lambda: ValueError("rotation must contain finite values"),
            )
        ]
    )
    return _rotation_matrix_to_quaternion(rotation)


@compiled_when_enabled
def _rotation_matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    """The arithmetic of :func:`rotation_matrix_to_quaternion` as pure tensor code (inputs already validated)."""
    m00, m01, m02 = rotation[..., 0, 0], rotation[..., 0, 1], rotation[..., 0, 2]
    m10, m11, m12 = rotation[..., 1, 0], rotation[..., 1, 1], rotation[..., 1, 2]
    m20, m21, m22 = rotation[..., 2, 0], rotation[..., 2, 1], rotation[..., 2, 2]
    trace = m00 + m11 + m22
    q_abs = 0.5 * torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1 + trace,
                    1 + m00 - m11 - m22,
                    1 - m00 + m11 - m22,
                    1 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            min=1.0e-12,
        )
    )
    denominator = 4 * q_abs.clamp_min(1.0e-6)
    by_w = torch.stack(
        (
            q_abs[..., 0],
            (m21 - m12) / denominator[..., 0],
            (m02 - m20) / denominator[..., 0],
            (m10 - m01) / denominator[..., 0],
        ),
        dim=-1,
    )
    by_x = torch.stack(
        (
            (m21 - m12) / denominator[..., 1],
            q_abs[..., 1],
            (m01 + m10) / denominator[..., 1],
            (m02 + m20) / denominator[..., 1],
        ),
        dim=-1,
    )
    by_y = torch.stack(
        (
            (m02 - m20) / denominator[..., 2],
            (m01 + m10) / denominator[..., 2],
            q_abs[..., 2],
            (m12 + m21) / denominator[..., 2],
        ),
        dim=-1,
    )
    by_z = torch.stack(
        (
            (m10 - m01) / denominator[..., 3],
            (m02 + m20) / denominator[..., 3],
            (m12 + m21) / denominator[..., 3],
            q_abs[..., 3],
        ),
        dim=-1,
    )
    candidates = torch.stack((by_w, by_x, by_y, by_z), dim=-2)
    branch = torch.argmax(q_abs, dim=-1)
    quaternion = torch.gather(
        candidates,
        -2,
        branch.unsqueeze(-1).unsqueeze(-1).expand(*branch.shape, 1, 4),
    ).squeeze(-2)
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    return cast(torch.Tensor, quaternion / norm.clamp_min(1.0e-12))


def matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    """Alias for :func:`rotation_matrix_to_quaternion`."""

    return rotation_matrix_to_quaternion(rotation)


def _gather_image_tensor(value: torch.Tensor, sampling: SamplingResult) -> torch.Tensor:
    """Gather ``[B,H,W,*]`` values by a :class:`SamplingResult`."""

    if value.ndim < 3:
        raise ValueError("value must have image-layout rank >= 3")
    batch_size, height, width = value.shape[:3]
    if sampling.coverage.shape[0] != batch_size or sampling.coverage.shape[-2:] != (height, width):
        raise ValueError("value spatial shape must match sampling")
    flat = value.reshape(batch_size, height * width, *value.shape[3:])
    count = sampling.source_index.shape[1]
    if count == 0:
        return flat.new_empty((batch_size, 0, *value.shape[3:]))
    safe_index = sampling.source_index.clamp_min(0)
    index = safe_index.reshape(batch_size, count, *([1] * (flat.ndim - 2)))
    index = index.expand(batch_size, count, *flat.shape[2:])
    gathered = torch.gather(flat, 1, index)
    return torch.where(
        sampling.valid_mask.reshape(batch_size, count, *([1] * (flat.ndim - 2))),
        gathered,
        torch.zeros_like(gathered),
    )


def _coerce_policy(
    policy: GeometryPriorPolicy | None,
    surface_policy: SurfaceFramePolicy | None,
    sampling_policy: SamplingPolicy | None,
) -> GeometryPriorPolicy:
    if policy is not None and (surface_policy is not None or sampling_policy is not None):
        raise ValueError("pass either policy or surface_policy/sampling_policy, not both")
    if policy is not None:
        return policy
    return GeometryPriorPolicy(
        surface=surface_policy or SurfaceFramePolicy(),
        sampling=sampling_policy or SamplingPolicy(),
    )


def build_geometry_prior(
    raw_disparity: torch.Tensor,
    camera: RectifiedStereoCamera,
    support: torch.Tensor,
    rgb: torch.Tensor | None = None,
    *,
    min_effective_disparity: float = 1.0e-6,
    policy: GeometryPriorPolicy | None = None,
    surface_policy: SurfaceFramePolicy | None = None,
    sampling_policy: SamplingPolicy | None = None,
    source_features: torch.Tensor | None = None,
    features: torch.Tensor | None = None,
) -> GeometryPrior:
    """Construct metric Gaussian priors from raw rectified disparity.

    ``rgb`` and optional ``source_features`` are sampled with the exact same
    discrete correspondence as ``mu``.  The raw disparity-to-depth and
    backprojection conventions are delegated to :func:`reconstruct_rectified`;
    support, discontinuity and branch masks are never differentiated.
    """

    if not isinstance(camera, RectifiedStereoCamera):
        raise TypeError("camera must be a RectifiedStereoCamera")
    if source_features is not None and features is not None:
        raise ValueError("pass either source_features or features, not both")
    if features is not None:
        source_features = features
    resolved_policy = _coerce_policy(policy, surface_policy, sampling_policy)
    geometry = reconstruct_rectified(
        raw_disparity,
        camera,
        support,
        min_effective_disparity=min_effective_disparity,
    )
    points = geometry.points.centers
    batch_size, _, height, width = points.shape
    point_image = points.permute(0, 2, 3, 1)
    cells = (
        fixed_grid_cells(height, width, resolved_policy.sampling)
        if static_grid_layout_enabled()
        else None
    )
    if cells is None:
        surface = estimate_surface_frame(
            points,
            geometry.points.valid_mask,
            policy=resolved_policy.surface,
            layout="channel_first",
        )
        sampling = sample_grid(surface.valid_mask, policy=resolved_policy.sampling)
        mu = _gather_image_tensor(point_image, sampling)
        frame = _gather_image_tensor(surface.frame, sampling)
        scales = _gather_image_tensor(surface.scales, sampling)
        fallback = _gather_image_tensor(surface.fallback_mask.unsqueeze(-1), sampling).squeeze(-1)
        discontinuity = _gather_image_tensor(
            surface.discontinuity_mask.unsqueeze(-1), sampling
        ).squeeze(-1)
        tangent_fallback = _gather_image_tensor(
            surface.tangent_fallback_mask.unsqueeze(-1), sampling
        ).squeeze(-1)
    else:
        # Static grid layout (utils.inference_mode): slot k is grid cell k, surface frames only at the grid cells, no nonzero.
        # Holes are masked by ``valid`` below exactly like padded slots.
        y0, x0, rows, cols = cells
        grid_policy = resolved_policy.sampling
        grid = estimate_surface_frame_on_grid(
            points,
            geometry.points.valid_mask,
            stride=grid_policy.stride,
            offset_xy=grid_policy.offset_xy,
            policy=resolved_policy.surface,
            layout="channel_first",
        )
        count = rows * cols
        sampling = sample_fixed_grid_static(
            grid.valid_mask, (height, width), cells, policy=grid_policy
        )
        mu = point_image[:, y0 :: grid_policy.stride, x0 :: grid_policy.stride].reshape(
            batch_size, count, 3
        )
        frame = grid.frame.reshape(batch_size, count, 3, 3)
        scales = grid.scales.reshape(batch_size, count, 3)
        fallback = grid.fallback_mask.reshape(batch_size, count)
        discontinuity = grid.discontinuity_mask.reshape(batch_size, count)
        tangent_fallback = grid.tangent_fallback_mask.reshape(batch_size, count)
    valid = sampling.valid_mask
    # Surface frame values are finite for valid centres.  Zero/identity
    # placeholders keep padded fields finite and make omission explicit.
    dtype = points.dtype
    identity = torch.eye(3, dtype=dtype, device=points.device).expand(batch_size, -1, -1)
    identity = identity.unsqueeze(1).expand(-1, sampling.source_index.shape[1], -1, -1)
    frame = torch.where(valid.unsqueeze(-1).unsqueeze(-1), frame, identity)
    scales = torch.where(
        valid.unsqueeze(-1),
        scales,
        points.new_full(
            scales.shape,
            resolved_policy.surface.bounded_fallback_scale_m,
        ),
    )
    mu = torch.where(valid.unsqueeze(-1), mu, torch.zeros_like(mu))
    rotations = rotation_matrix_to_quaternion(frame)
    rotations = rotations / torch.linalg.vector_norm(rotations, dim=-1, keepdim=True).clamp_min(
        1.0e-12
    )
    covariance_result = build_surface_covariance(rotations, scales, valid_mask=valid)
    cov_surface = torch.where(
        valid.unsqueeze(-1).unsqueeze(-1),
        covariance_result.cov_surface,
        torch.zeros_like(covariance_result.cov_surface),
    )

    if rgb is None:
        rgb = torch.zeros((batch_size, 3, height, width), dtype=dtype, device=points.device)
    if not isinstance(rgb, torch.Tensor) or rgb.ndim != 4 or rgb.shape[:1] != (batch_size,):
        raise ValueError("rgb must have shape [B,3,H,W]")
    if rgb.shape[1] != 3 or rgb.shape[2:] != (height, width):
        raise ValueError("rgb must have shape [B,3,H,W] matching disparity")
    if rgb.device != points.device:
        raise ValueError("rgb must share disparity device")
    rgb_sample = gather_source(rgb.to(dtype), sampling)
    rgb_sample = torch.where(valid.unsqueeze(-1), rgb_sample, torch.zeros_like(rgb_sample))

    feature_sample: torch.Tensor | None = None
    if source_features is not None:
        if not isinstance(source_features, torch.Tensor) or source_features.ndim != 4:
            raise ValueError("source_features must be [B,C,H,W]")
        if source_features.shape[0] != batch_size or source_features.shape[2:] != (height, width):
            raise ValueError("source_features spatial/batch shape must match disparity")
        if source_features.device != points.device:
            raise ValueError("source_features must share disparity device")
        feature_sample = gather_source(source_features, sampling)
        feature_sample = torch.where(
            valid.unsqueeze(-1), feature_sample, torch.zeros_like(feature_sample)
        )

    coverage = valid.unsqueeze(-1).to(dtype)
    alpha0 = coverage * float(resolved_policy.alpha_valid)
    selected, eligible, fallback_count, discontinuity_count, tangent_fallback_count = read_ints(
        valid.sum(),
        sampling.eligible_mask.sum(),
        (fallback & valid).sum(),
        (discontinuity & valid).sum(),
        (tangent_fallback & valid).sum(),
    )
    diagnostics = GeometryPriorDiagnostics(
        eligible_count=eligible,
        selected_count=selected,
        valid_count=selected,
        fallback_count=fallback_count,
        discontinuity_count=discontinuity_count,
        tangent_fallback_count=tangent_fallback_count,
        dropped_count=max(eligible - selected, 0),
    )
    return GeometryPrior(
        mu=mu,
        R0=frame,
        rotations=rotations,
        S0=scales,
        alpha0=alpha0,
        rgb=rgb_sample,
        valid_mask=valid,
        source_index=sampling.source_index,
        source_xy=sampling.source_xy.to(dtype=dtype),
        coverage=coverage,
        cov_surface=cov_surface,
        fallback_mask=fallback & valid,
        discontinuity_mask=discontinuity & valid,
        tangent_fallback_mask=tangent_fallback & valid,
        sampling=sampling,
        diagnostics=diagnostics,
        source_features=feature_sample,
    )


def construct_geometry_prior(*args: object, **kwargs: object) -> GeometryPrior:
    """Alias for :func:`build_geometry_prior`."""

    return build_geometry_prior(*args, **kwargs)  # type: ignore[arg-type]


def geometry_prior_from_disparity(*args: object, **kwargs: object) -> GeometryPrior:
    """Alias for :func:`build_geometry_prior`."""

    return build_geometry_prior(*args, **kwargs)  # type: ignore[arg-type]


GeometryPriorConfig = GeometryPriorPolicy
build_priors = build_geometry_prior


__all__ = [
    "GeometryPrior",
    "GeometryPriorDiagnostics",
    "GeometryPriorPolicy",
    "GeometryPriorConfig",
    "build_geometry_prior",
    "build_priors",
    "construct_geometry_prior",
    "geometry_prior_from_disparity",
    "matrix_to_quaternion",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
]
