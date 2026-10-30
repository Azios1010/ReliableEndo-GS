"""Masked metric surface frames for the independent Gaussian constructor.

The routines in this module deliberately operate on camera-frame metric
points.  They do not infer a world frame and they do not use an eigenvector
decomposition to choose a Gaussian rotation.  A local normal is estimated
from a small, validity- and depth-jump-masked neighbourhood; the remaining
frame is a deterministic camera-relative gauge owned by this module.
"""

import functools
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import torch

from reliable_endo_gs.utils.inference_mode import compiled_when_enabled


@dataclass(frozen=True, slots=True)
class SurfaceFramePolicy:
    """Numerical and neighbourhood policy for :func:`estimate_surface_frame`.

    Lengths are metric metres.  ``neighborhood_radius`` controls the square
    pixel neighbourhood (the centre itself is excluded).  Depth-jump checks
    use optical-axis ``z`` in metres and are intentionally hard masks; the
    selected valid point arithmetic remains differentiable.
    """

    neighborhood_radius: int = 1
    depth_discontinuity_abs_m: float = 0.01
    depth_discontinuity_rel: float = 0.05
    min_neighbor_count: int = 2
    min_cross_area_m2: float = 1.0e-10
    tangent_epsilon: float = 1.0e-5
    scale_factor: float = 0.5
    min_tangent_scale_m: float = 1.0e-4
    max_tangent_scale_m: float = 0.25
    normal_thickness_m: float = 1.0e-3
    max_normal_thickness_m: float = 0.25
    fallback_scale_m: float = 5.0e-3

    def __post_init__(self) -> None:
        if self.neighborhood_radius < 1:
            raise ValueError("neighborhood_radius must be at least one")
        if self.min_neighbor_count < 2:
            raise ValueError("min_neighbor_count must be at least two")
        for name, value in (
            ("depth_discontinuity_abs_m", self.depth_discontinuity_abs_m),
            ("depth_discontinuity_rel", self.depth_discontinuity_rel),
            ("min_cross_area_m2", self.min_cross_area_m2),
            ("tangent_epsilon", self.tangent_epsilon),
            ("scale_factor", self.scale_factor),
            ("min_tangent_scale_m", self.min_tangent_scale_m),
            ("max_tangent_scale_m", self.max_tangent_scale_m),
            ("normal_thickness_m", self.normal_thickness_m),
            ("max_normal_thickness_m", self.max_normal_thickness_m),
            ("fallback_scale_m", self.fallback_scale_m),
        ):
            if not isinstance(value, (float, int)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a finite non-negative number")
            if not isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.scale_factor <= 0:
            raise ValueError("scale_factor must be positive")
        if self.min_tangent_scale_m <= 0:
            raise ValueError("min_tangent_scale_m must be positive")
        if self.max_tangent_scale_m <= 0:
            raise ValueError("max_tangent_scale_m must be positive")
        if self.max_tangent_scale_m < self.min_tangent_scale_m:
            raise ValueError("max_tangent_scale_m must be >= min_tangent_scale_m")
        if self.normal_thickness_m <= 0:
            raise ValueError("normal_thickness_m must be positive")
        if self.max_normal_thickness_m <= 0:
            raise ValueError("max_normal_thickness_m must be positive")
        if self.normal_thickness_m > self.max_normal_thickness_m:
            raise ValueError("normal_thickness_m must be <= max_normal_thickness_m")
        if self.fallback_scale_m <= 0:
            raise ValueError("fallback_scale_m must be positive")

    @property
    def bounded_fallback_scale_m(self) -> float:
        """Return the positive isotropic fallback scale in metric metres."""

        return min(
            max(self.fallback_scale_m, self.min_tangent_scale_m),
            self.max_tangent_scale_m,
            self.max_normal_thickness_m,
        )


@dataclass(frozen=True, slots=True)
class SurfaceFrameResult:
    """Surface-frame estimate in image layout.

    Vector fields have shape ``[B,H,W,3]`` and ``frame`` has shape
    ``[B,H,W,3,3]`` with columns ``[t1,t2,n]``.  ``valid_mask`` is the
    finite, positive-depth centre support supplied by the caller.  A valid
    centre can have ``fallback_mask=True`` when its neighbourhood cannot
    support a reliable normal; such a centre is still safe to represent with
    the conservative isotropic scales in ``scales``.
    """

    normals: torch.Tensor
    tangent_1: torch.Tensor
    tangent_2: torch.Tensor
    frame: torch.Tensor
    scales: torch.Tensor
    valid_mask: torch.Tensor
    normal_valid_mask: torch.Tensor
    fallback_mask: torch.Tensor
    discontinuity_mask: torch.Tensor
    degeneracy_mask: torch.Tensor
    tangent_fallback_mask: torch.Tensor
    neighbor_count: torch.Tensor
    accepted_neighbor_mask: torch.Tensor
    selected_pair: torch.Tensor

    @property
    def R0(self) -> torch.Tensor:
        """Alias for the frame matrix whose columns are ``[t1,t2,n]``."""

        return self.frame

    @property
    def normal(self) -> torch.Tensor:
        """Singular alias retained for callers that request one normal field."""

        return self.normals

    @property
    def tangent_fallback(self) -> torch.Tensor:
        """Alias for the hard tangent-axis branch mask."""

        return self.tangent_fallback_mask


_DEFAULT_SURFACE_FRAME_POLICY = SurfaceFramePolicy()


def _as_image_points(
    centers: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    layout: Literal["channel_first", "channel_last"] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Normalize ``[B,3,H,W]`` or ``[B,H,W,3]`` points to image layout."""

    if not isinstance(centers, torch.Tensor):
        raise TypeError("centers must be a torch.Tensor")
    if not torch.is_floating_point(centers):
        raise TypeError("centers must be floating point")
    if centers.ndim != 4:
        raise ValueError(f"centers must have rank four; got {tuple(centers.shape)}")
    channel_first_shape = centers.shape[1] == 3
    channel_last_shape = centers.shape[-1] == 3
    if layout is None and channel_first_shape and channel_last_shape:
        raise ValueError(
            "centers layout is ambiguous; pass layout='channel_first' or layout='channel_last'"
        )
    if layout not in (None, "channel_first", "channel_last"):
        raise ValueError(f"unsupported centers layout: {layout!r}")
    if layout == "channel_first" or (layout is None and channel_first_shape):
        if not channel_first_shape:
            raise ValueError("layout='channel_first' requires centers shape [B,3,H,W]")
        image_points = centers.permute(0, 2, 3, 1)
        channel_first = True
        expected_mask_shape = (centers.shape[0], 1, centers.shape[2], centers.shape[3])
    elif layout == "channel_last" or (layout is None and channel_last_shape):
        if not channel_last_shape:
            raise ValueError("layout='channel_last' requires centers shape [B,H,W,3]")
        image_points = centers
        channel_first = False
        expected_mask_shape = (centers.shape[0], 1, centers.shape[1], centers.shape[2])
    else:
        raise ValueError(
            f"centers must have shape [B,3,H,W] or [B,H,W,3]; got {tuple(centers.shape)}"
        )
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must have dtype torch.bool")
    if valid_mask.ndim == 3:
        mask = valid_mask
    elif valid_mask.shape == expected_mask_shape:
        mask = valid_mask[:, 0]
    else:
        raise ValueError(
            "valid_mask must have shape [B,H,W] or [B,1,H,W] matching centers; "
            f"got {tuple(valid_mask.shape)}"
        )
    if mask.shape != image_points.shape[:3]:
        raise ValueError("valid_mask spatial shape must match centers")
    if mask.device != centers.device:
        raise ValueError("valid_mask must share centers device")
    return image_points, mask, channel_first


def _shifted_neighbour(
    points: torch.Tensor,
    valid: torch.Tensor,
    dy: int,
    dx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a zero-padded neighbour at ``(dy, dx)`` without wraparound."""

    # A roll is used only for data movement.  The explicit boundary mask
    # below prevents the usual wraparound from entering the neighbourhood.
    shifted_points = torch.roll(points, shifts=(-dy, -dx), dims=(1, 2))
    shifted_valid = torch.roll(valid, shifts=(-dy, -dx), dims=(1, 2))
    height, width = points.shape[1:3]
    boundary = torch.ones((height, width), dtype=torch.bool, device=points.device)
    # ``torch.roll(..., shifts=(-dy, -dx))`` maps output pixel ``(y, x)``
    # to source pixel ``(y + dy, x + dx)``.  Therefore a positive offset is
    # invalid at the *last* rows/columns and a negative offset is invalid at
    # the *first* rows/columns.  Keeping this mapping explicit prevents a
    # wrapped neighbour from crossing an image border.
    if dy > 0:
        boundary[-dy:] = False
    elif dy < 0:
        boundary[:-dy] = False
    if dx > 0:
        boundary[:, -dx:] = False
    elif dx < 0:
        boundary[:, :-dx] = False
    shifted_valid = shifted_valid & boundary.unsqueeze(0)
    return shifted_points, shifted_valid


def _orient_toward_camera(normal: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Orient normals toward the camera with a deterministic tie rule."""

    dot = (normal * centers).sum(dim=-1)
    flip = dot > 0
    tie = dot.abs() <= 1.0e-12
    # Exact camera-ray tangencies are rare, but a fixed z/x/y lexicographic
    # sign prevents frame flicker at the branch.  For the declared camera
    # convention a positive z normal is the away-from-camera tie choice.
    z_positive = normal[..., 2] > 0
    z_tied = normal[..., 2].abs() <= 1.0e-12
    x_positive = normal[..., 0] > 0
    x_tied = normal[..., 0].abs() <= 1.0e-12
    y_positive = normal[..., 1] > 0
    tie_flip = z_positive | (z_tied & x_positive) | (z_tied & x_tied & y_positive)
    flip = flip | (tie & tie_flip)
    return torch.where(flip.unsqueeze(-1), -normal, normal)


def _camera_relative_gauge(
    normal: torch.Tensor,
    *,
    tangent_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct ``[t1,t2,n]`` with an explicit least-aligned-axis fallback."""

    axes = torch.eye(3, dtype=normal.dtype, device=normal.device)
    camera_x = axes[0].expand(*normal.shape[:-1], 3)
    projection = camera_x - (camera_x * normal).sum(dim=-1, keepdim=True) * normal
    projection_norm = torch.linalg.vector_norm(projection, dim=-1)
    axis_alignment = normal.abs()
    least_aligned_index = torch.argmin(axis_alignment, dim=-1)
    chosen_axis = axes[least_aligned_index]
    fallback_projection = chosen_axis - (chosen_axis * normal).sum(dim=-1, keepdim=True) * normal
    use_fallback = projection_norm < tangent_epsilon
    raw_tangent = torch.where(use_fallback.unsqueeze(-1), fallback_projection, projection)
    tangent_norm = torch.linalg.vector_norm(raw_tangent, dim=-1, keepdim=True)
    safe_tangent_norm = torch.where(tangent_norm > 0, tangent_norm, torch.ones_like(tangent_norm))
    tangent_1 = raw_tangent / safe_tangent_norm
    tangent_2 = torch.linalg.cross(normal, tangent_1, dim=-1)
    tangent_2_norm = torch.linalg.vector_norm(tangent_2, dim=-1, keepdim=True)
    tangent_2 = tangent_2 / torch.where(
        tangent_2_norm > 0, tangent_2_norm, torch.ones_like(tangent_2_norm)
    )
    return tangent_1, tangent_2, use_fallback, least_aligned_index


@functools.lru_cache(maxsize=8)
def _pair_table(pairs: tuple[tuple[int, int], ...], device: torch.device) -> torch.Tensor:
    """Immutable [P,2] neighbour-pair table, built once per (pairs, device).

    Creating it per call was a pageable host-to-device copy that also waited for queued kernels.
    """
    return torch.tensor(pairs, dtype=torch.long, device=device)


def _neighbour_offsets(policy: SurfaceFramePolicy) -> list[tuple[int, int]]:
    """Neighbour ``(dy, dx)`` offsets in the fixed order that the pair enumeration and ``selected_pair`` refer to."""

    return [
        (dy, dx)
        for distance in range(1, policy.neighborhood_radius + 1)
        for dy in range(-distance, distance + 1)
        for dx in range(-distance, distance + 1)
        if (dy != 0 or dx != 0) and max(abs(dy), abs(dx)) == distance
    ]


def _surface_frame_from_neighbours(
    safe_points: torch.Tensor,
    center_valid: torch.Tensor,
    neighbour_points: torch.Tensor,
    neighbour_valid: torch.Tensor,
    resolved_policy: SurfaceFramePolicy,
) -> SurfaceFrameResult:
    """Per-centre surface-frame arithmetic shared by the dense and the grid evaluation (single owner).

    ``safe_points`` [B,A,C,3] finite centres, ``center_valid`` [B,A,C], ``neighbour_points`` [B,A,C,K,3] and
    ``neighbour_valid`` [B,A,C,K] in ``estimate_surface_frame`` offset order. Every centre is independent of every other.
    """

    offset_count = neighbour_points.shape[3]
    pair_table = _pair_table(
        tuple(
            (first, second)
            for first in range(offset_count)
            for second in range(first + 1, offset_count)
        ),
        safe_points.device,
    )
    return SurfaceFrameResult(
        *_surface_frame_tensors(
            safe_points,
            center_valid,
            neighbour_points,
            neighbour_valid,
            pair_table,
            resolved_policy,
        )
    )


@compiled_when_enabled
def _surface_frame_tensors(
    safe_points: torch.Tensor,
    center_valid: torch.Tensor,
    neighbour_points: torch.Tensor,
    neighbour_valid: torch.Tensor,
    pair_table: torch.Tensor,
    resolved_policy: SurfaceFramePolicy,
) -> tuple[torch.Tensor, ...]:
    """The arithmetic of :func:`_surface_frame_from_neighbours` as pure tensor code; fields in ``SurfaceFrameResult`` order."""

    batch_size, height, width, _ = safe_points.shape
    offset_count = neighbour_points.shape[3]

    centre_z = safe_points[..., 2].unsqueeze(-1)
    neighbour_z = neighbour_points[..., 2]
    jump_limit = (
        resolved_policy.depth_discontinuity_abs_m
        + resolved_policy.depth_discontinuity_rel * centre_z.abs()
    )
    finite_jump = torch.isfinite(neighbour_z) & torch.isfinite(centre_z)
    depth_continuous = finite_jump & ((neighbour_z - centre_z).abs() <= jump_limit)
    accepted = neighbour_valid & depth_continuous & center_valid.unsqueeze(-1)
    neighbour_count = accepted.sum(dim=-1)
    discontinuity_mask = center_valid & (neighbour_valid & ~depth_continuous).any(dim=-1)

    neighbour_vectors = neighbour_points - safe_points.unsqueeze(3)
    pair_crosses: list[torch.Tensor] = []
    pair_validity: list[torch.Tensor] = []
    for first in range(offset_count):
        for second in range(first + 1, offset_count):
            pair_crosses.append(
                torch.linalg.cross(
                    neighbour_vectors[..., first, :],
                    neighbour_vectors[..., second, :],
                    dim=-1,
                )
            )
            pair_validity.append(accepted[..., first] & accepted[..., second])
    crosses = torch.stack(pair_crosses, dim=3)
    pairs_valid = torch.stack(pair_validity, dim=3)
    areas = torch.linalg.vector_norm(crosses, dim=-1)
    scores = torch.where(pairs_valid, areas, torch.full_like(areas, -1.0))
    best_pair_index = torch.argmax(scores, dim=-1)
    best_area = torch.gather(areas, -1, best_pair_index.unsqueeze(-1)).squeeze(-1)
    selected_cross = torch.gather(
        crosses,
        3,
        best_pair_index.unsqueeze(-1).unsqueeze(-1).expand(batch_size, height, width, 1, 3),
    ).squeeze(3)
    normal_valid = (
        center_valid
        & (neighbour_count >= resolved_policy.min_neighbor_count)
        & (best_area >= resolved_policy.min_cross_area_m2)
    )
    degeneracy_mask = center_valid & ~normal_valid

    selected_norm = torch.linalg.vector_norm(selected_cross, dim=-1, keepdim=True)
    safe_selected_norm = torch.where(
        selected_norm > 0, selected_norm, torch.ones_like(selected_norm)
    )
    cross_normal = selected_cross / safe_selected_norm
    centre_norm = torch.linalg.vector_norm(safe_points, dim=-1, keepdim=True)
    safe_centre_norm = torch.where(centre_norm > 0, centre_norm, torch.ones_like(centre_norm))
    ray_normal = -safe_points / safe_centre_norm
    canonical_normal = torch.zeros_like(safe_points)
    canonical_normal[..., 2] = -1.0
    ray_normal = torch.where((centre_norm > 0).expand_as(ray_normal), ray_normal, canonical_normal)
    normal = torch.where(normal_valid.unsqueeze(-1), cross_normal, ray_normal)
    normal = _orient_toward_camera(normal, safe_points)
    normal = torch.where(center_valid.unsqueeze(-1), normal, canonical_normal)

    tangent_1, tangent_2, tangent_fallback, _ = _camera_relative_gauge(
        normal, tangent_epsilon=resolved_policy.tangent_epsilon
    )
    tangent_fallback = tangent_fallback & center_valid
    identity = torch.eye(3, dtype=safe_points.dtype, device=safe_points.device)
    frame = torch.stack((tangent_1, tangent_2, normal), dim=-1)
    frame = torch.where(center_valid.unsqueeze(-1).unsqueeze(-1), frame, identity)

    distances = torch.linalg.vector_norm(neighbour_vectors, dim=-1)
    distance_sum = torch.where(accepted, distances, torch.zeros_like(distances)).sum(dim=-1)
    safe_count = neighbour_count.clamp_min(1).to(safe_points.dtype)
    local_spacing = distance_sum / safe_count
    tangent_scale = (local_spacing * resolved_policy.scale_factor).clamp(
        min=resolved_policy.min_tangent_scale_m, max=resolved_policy.max_tangent_scale_m
    )
    normal_scale = safe_points.new_full(tangent_scale.shape, resolved_policy.normal_thickness_m)
    regular_scales = torch.stack((tangent_scale, tangent_scale, normal_scale), dim=-1)
    bounded_fallback_scale = resolved_policy.bounded_fallback_scale_m
    fallback_scales = safe_points.new_full((*tangent_scale.shape, 3), bounded_fallback_scale)
    scales = torch.where(
        normal_valid.unsqueeze(-1),
        regular_scales,
        fallback_scales,
    )
    scales = torch.where(center_valid.unsqueeze(-1), scales, fallback_scales)

    selected_pair = torch.empty(
        (*best_pair_index.shape, 2), dtype=torch.long, device=safe_points.device
    )
    selected_pair.copy_(pair_table[best_pair_index])
    selected_pair = torch.where(
        center_valid.unsqueeze(-1), selected_pair, torch.full_like(selected_pair, -1)
    )

    return (
        normal,
        tangent_1,
        tangent_2,
        frame,
        scales,
        center_valid,
        normal_valid,
        center_valid & ~normal_valid,
        discontinuity_mask,
        degeneracy_mask,
        tangent_fallback,
        neighbour_count,
        accepted,
        selected_pair,
    )


def estimate_surface_frame(
    centers: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    policy: SurfaceFramePolicy | None = None,
    layout: Literal["channel_first", "channel_last"] | None = None,
) -> SurfaceFrameResult:
    """Estimate masked camera-relative surface frames from metric points.

    ``centers`` are left-camera points in metres.  The centre validity mask is
    hard and non-differentiable by design.  For valid centres, selected point
    differences and cross products retain autograd connectivity.  A normal is
    selected from the largest non-degenerate pairwise neighbourhood cross
    product; if no such pair survives the depth-jump mask, a camera-facing
    ray normal and bounded isotropic scale are returned with ``fallback_mask``
    set.  Invalid centres receive finite placeholders and ``valid_mask=False``.
    """

    resolved_policy = policy or _DEFAULT_SURFACE_FRAME_POLICY
    image_points, input_valid, _ = _as_image_points(centers, valid_mask, layout=layout)
    finite = torch.isfinite(image_points).all(dim=-1)
    center_valid = input_valid & finite & (image_points[..., 2] > 0)
    safe_points = torch.where(finite.unsqueeze(-1), image_points, torch.zeros_like(image_points))

    neighbours: list[torch.Tensor] = []
    neighbour_validity: list[torch.Tensor] = []
    for dy, dx in _neighbour_offsets(resolved_policy):
        neighbour, neighbour_valid = _shifted_neighbour(safe_points, center_valid, dy, dx)
        neighbours.append(neighbour)
        neighbour_validity.append(neighbour_valid)
    neighbour_points = torch.stack(neighbours, dim=3)
    neighbour_valid = torch.stack(neighbour_validity, dim=3)
    return _surface_frame_from_neighbours(
        safe_points, center_valid, neighbour_points, neighbour_valid, resolved_policy
    )


def estimate_surface_frame_on_grid(
    centers: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    stride: int,
    offset_xy: tuple[int, int] = (0, 0),
    policy: SurfaceFramePolicy | None = None,
    layout: Literal["channel_first", "channel_last"] | None = None,
) -> SurfaceFrameResult:
    """:func:`estimate_surface_frame` evaluated only at the centres ``(y0 + stride*i, x0 + stride*j)``.

    ``y0 = offset_xy[1] % stride`` and ``x0 = offset_xy[0] % stride`` are the first row and column of the regular grid of
    ``SamplingPolicy(mode="fixed")``. Every field has shape ``[B,rows,cols,...]`` and equals the dense result sliced at the
    same centres: neighbours are strided views of a zero-padded copy of the points (padding is invalid, like the border mask of
    the dense roll), and the arithmetic is the shared :func:`_surface_frame_from_neighbours`. Used when only the grid centres
    are consumed, so the other ``stride**2 - 1`` of every ``stride**2`` pixels are not evaluated.
    """

    if stride < 1:
        raise ValueError("stride must be a positive integer")
    resolved_policy = policy or _DEFAULT_SURFACE_FRAME_POLICY
    image_points, input_valid, _ = _as_image_points(centers, valid_mask, layout=layout)
    finite = torch.isfinite(image_points).all(dim=-1)
    center_valid = input_valid & finite & (image_points[..., 2] > 0)
    safe_points = torch.where(finite.unsqueeze(-1), image_points, torch.zeros_like(image_points))
    batch_size, height, width, _ = safe_points.shape
    radius = resolved_policy.neighborhood_radius
    y0, x0 = offset_xy[1] % stride, offset_xy[0] % stride
    rows, cols = len(range(y0, height, stride)), len(range(x0, width, stride))
    if rows == 0 or cols == 0:
        raise ValueError("the grid has no cells inside the image")

    padded_points = safe_points.new_zeros((batch_size, height + 2 * radius, width + 2 * radius, 3))
    padded_points[:, radius : radius + height, radius : radius + width] = safe_points
    padded_valid = center_valid.new_zeros((batch_size, height + 2 * radius, width + 2 * radius))
    padded_valid[:, radius : radius + height, radius : radius + width] = center_valid

    def at(padded: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        top, left = radius + y0 + dy, radius + x0 + dx
        return padded[
            :,
            top : top + stride * (rows - 1) + 1 : stride,
            left : left + stride * (cols - 1) + 1 : stride,
        ]

    offsets = _neighbour_offsets(resolved_policy)
    neighbour_points = torch.stack([at(padded_points, dy, dx) for dy, dx in offsets], dim=3)
    neighbour_valid = torch.stack([at(padded_valid, dy, dx) for dy, dx in offsets], dim=3)
    return _surface_frame_from_neighbours(
        at(padded_points, 0, 0),
        at(padded_valid, 0, 0),
        neighbour_points,
        neighbour_valid,
        resolved_policy,
    )


def estimate_surface_normals(
    centers: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    policy: SurfaceFramePolicy | None = None,
    layout: Literal["channel_first", "channel_last"] | None = None,
) -> SurfaceFrameResult:
    """Compatibility name for :func:`estimate_surface_frame`."""

    return estimate_surface_frame(centers, valid_mask, policy=policy, layout=layout)


SurfacePolicy = SurfaceFramePolicy
SurfaceFrame = SurfaceFrameResult


__all__ = [
    "SurfaceFramePolicy",
    "SurfaceFrameResult",
    "SurfacePolicy",
    "SurfaceFrame",
    "estimate_surface_frame",
    "estimate_surface_frame_on_grid",
    "estimate_surface_normals",
]
