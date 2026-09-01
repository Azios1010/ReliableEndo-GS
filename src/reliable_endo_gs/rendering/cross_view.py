"""Cross-view camera, masking, loss, and development-renderer contracts.

The production renderer remains behind :mod:`reliable_endo_gs.rendering.protocol`.
The reference renderer in this module is intentionally small and CPU-safe: it
exists to exercise camera selection, tensor flow, masking, and autograd only.
It is not a Graphdeco renderer and provides no production-parity evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

import torch

from reliable_endo_gs.contracts import CameraBatch, RenderOutput
from reliable_endo_gs.rendering.protocol import RendererRequest


class CameraView(str, Enum):
    """The two explicitly selectable cameras in a stereo request."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class CameraRequest:
    """A renderer request's explicit camera and image geometry.

    ``CameraBatch.world_from_camera`` is camera-to-world according to the
    shared camera contract.  The reference renderer therefore applies its
    inverse when projecting world-frame Gaussian centers.
    """

    camera: CameraBatch
    view: CameraView | str
    image_size: tuple[int, int]
    schema_version: str = "cross_view_camera_request.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.camera, CameraBatch):
            raise TypeError("camera must be a CameraBatch")
        view = CameraView(self.view)
        object.__setattr__(self, "view", view)
        if (
            len(self.image_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in self.image_size
            )
            or any(value < 1 for value in self.image_size)
        ):
            raise ValueError("image_size must be a pair of positive integers")
        if not self.schema_version:
            raise ValueError("schema_version must be non-empty")


CrossViewCameraRequest = CameraRequest


def make_camera_request(
    camera: CameraBatch,
    *,
    view: CameraView | str,
    image_size: tuple[int, int],
) -> CameraRequest:
    """Build an explicit left or right renderer camera request."""

    return CameraRequest(camera=camera, view=view, image_size=image_size)


@dataclass(frozen=True, slots=True)
class MaskEvidence:
    """Optional, explicitly named cross-view mask components.

    ``geometry_valid`` is required either here or through the ``state``
    argument of :func:`build_cross_view_mask`.  Other fields are optional
    evidence.  An absent optional component is reported as unavailable and is
    not silently fabricated from image values.
    """

    geometry_valid: torch.Tensor | None = None
    stereo_valid: torch.Tensor | None = None
    in_view: torch.Tensor | None = None
    renderer_visibility: torch.Tensor | None = None
    left_right_valid: torch.Tensor | None = None
    depth_valid: torch.Tensor | None = None
    occlusion: torch.Tensor | None = None
    specularity: torch.Tensor | None = None
    tool: torch.Tensor | None = None
    saturation: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class CrossViewMaskResult:
    """Mask algebra result with coverage and explicit unavailable evidence."""

    component_masks: Mapping[str, torch.Tensor | None]
    combined_mask: torch.Tensor
    valid_count: int
    total_count: int
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_masks", MappingProxyType(dict(self.component_masks)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def coverage(self) -> float:
        """Return valid-pixel coverage, or zero for an empty image."""

        return self.valid_count / self.total_count if self.total_count else 0.0


def build_cross_view_mask(
    state: object | torch.Tensor,
    evidence: MaskEvidence | Mapping[str, torch.Tensor | None] | None = None,
) -> CrossViewMaskResult:
    """Combine required geometry validity and supplied cross-view evidence.

    Positive components are intersected; exclusion components are inverted
    before intersection.  The canonical required component is
    ``geometry_valid``.  ``state`` may be that tensor or an object/mapping with
    a ``geometry_valid``/``valid_mask`` field shaped ``[B,1,H,W]``.
    """

    state_mask = _extract_state_mask(state)
    values: dict[str, torch.Tensor | None] = {}
    if isinstance(evidence, MaskEvidence):
        values.update(
            {
                name: getattr(evidence, name)
                for name in (
                    "geometry_valid",
                    "stereo_valid",
                    "in_view",
                    "renderer_visibility",
                    "left_right_valid",
                    "depth_valid",
                    "occlusion",
                    "specularity",
                    "tool",
                    "saturation",
                )
            }
        )
    elif evidence is None:
        values = {}
    elif isinstance(evidence, Mapping):
        values = dict(evidence)
    else:
        raise TypeError("evidence must be MaskEvidence, a mapping, or None")

    geometry = values.pop("geometry_valid", None)
    if geometry is None:
        geometry = state_mask
    elif state_mask is not None:
        geometry = _validate_mask("geometry_valid", geometry, state_mask) & state_mask
    if geometry is None:
        raise ValueError("geometry_valid must be supplied by state or evidence")

    expected = tuple(geometry.shape)
    _validate_mask("geometry_valid", geometry, geometry)
    components: dict[str, torch.Tensor | None] = {"geometry_valid": geometry}
    components.update(values)
    for name, mask in components.items():
        if mask is not None:
            _validate_mask(name, mask, geometry)

    positive_names = (
        "geometry_valid",
        "stereo_valid",
        "in_view",
        "renderer_visibility",
        "left_right_valid",
        "depth_valid",
    )
    exclusion_names = ("occlusion", "specularity", "tool", "saturation")
    combined = geometry.clone()
    excluded_by: dict[str, int] = {}
    unavailable: list[str] = []
    for name in positive_names[1:]:
        mask = components.get(name)
        if mask is None:
            unavailable.append(name)
            continue
        excluded_by[name] = int((combined & ~mask).sum().item())
        combined = combined & mask
    for name in exclusion_names:
        mask = components.get(name)
        if mask is None:
            unavailable.append(name)
            continue
        excluded_by[name] = int((combined & mask).sum().item())
        combined = combined & ~mask

    diagnostics: dict[str, object] = {
        "required_components": ("geometry_valid",),
        "optional_unavailable": tuple(unavailable),
        "excluded_by_component": MappingProxyType(excluded_by),
        "valid_count": int(combined.sum().item()),
        "total_count": int(combined.numel()),
    }
    if tuple(combined.shape) != expected:
        raise AssertionError("cross-view mask composition changed shape")
    return CrossViewMaskResult(
        component_masks=components,
        combined_mask=combined,
        valid_count=int(combined.sum().item()),
        total_count=int(combined.numel()),
        diagnostics=diagnostics,
    )


def _extract_state_mask(state: object | torch.Tensor) -> torch.Tensor | None:
    if isinstance(state, torch.Tensor):
        return state
    if isinstance(state, Mapping):
        for name in ("geometry_valid", "valid_mask"):
            value = state.get(name)
            if value is not None:
                return value if isinstance(value, torch.Tensor) else None
        return None
    for name in ("geometry_valid", "valid_mask"):
        value = getattr(state, name, None)
        if value is not None:
            return value if isinstance(value, torch.Tensor) else None
    return None


def _validate_mask(name: str, mask: object, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if mask.shape != reference.shape:
        raise ValueError(
            f"{name} must have shape {tuple(reference.shape)}; got {tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")
    if mask.device != reference.device:
        raise ValueError(f"{name} must share device with geometry_valid")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B, 1, H, W]")
    return mask


class RobustLossKind(str, Enum):
    """Supported development photometric residual functions."""

    L1 = "l1"
    CHARBONNIER = "charbonnier"


@dataclass(frozen=True, slots=True)
class RobustLossConfig:
    """Pure loss settings with explicit pixel/channel/view reduction."""

    kind: RobustLossKind | str = RobustLossKind.CHARBONNIER
    charbonnier_epsilon: float = 1.0e-3
    left_weight: float = 1.0
    right_weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", RobustLossKind(self.kind))
        if self.charbonnier_epsilon <= 0:
            raise ValueError("charbonnier_epsilon must be strictly positive")
        for name, value in (("left_weight", self.left_weight), ("right_weight", self.right_weight)):
            if value < 0 or not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class MaskedLossResult:
    """One masked loss; ``loss=None`` explicitly denotes no supervision."""

    loss: torch.Tensor | None
    valid_count: int
    element_count: int
    total_count: int
    nonfinite_count: int
    skipped: bool
    definition: str

    @property
    def available(self) -> bool:
        return self.loss is not None and not self.skipped


@dataclass(frozen=True, slots=True)
class CrossViewLossResult:
    """Left/right components and weighted total for one Phase-I step."""

    left: MaskedLossResult
    right: MaskedLossResult
    total: torch.Tensor | None
    left_weight: float
    right_weight: float

    @property
    def available(self) -> bool:
        return self.total is not None

    def as_dict(self) -> dict[str, torch.Tensor | None]:
        return {
            "left_photometric": self.left.loss,
            "right_cross_view": self.right.loss,
            "total": self.total,
        }


def cross_view_loss(
    render: RenderOutput | torch.Tensor,
    observation: torch.Tensor,
    mask: torch.Tensor,
    loss_config: RobustLossConfig | None = None,
) -> MaskedLossResult:
    """Compute a finite, channel-averaged masked L1/Charbonnier loss.

    Inputs use ``[B,C,H,W]`` images and a boolean ``[B,1,H,W]`` pixel mask.
    The denominator is the number of valid pixels times channel count.  An
    empty mask returns ``loss=None`` rather than a fabricated zero.
    """

    config = loss_config or RobustLossConfig()
    predicted = render.image if isinstance(render, RenderOutput) else render
    _validate_image_pair(predicted, observation)
    _validate_mask("mask", mask, torch.empty_like(predicted[:, :1]))
    finite = torch.isfinite(predicted).all(dim=1, keepdim=True) & torch.isfinite(observation).all(
        dim=1, keepdim=True
    )
    valid = mask & finite
    valid_count = int(valid.sum().item())
    total_count = int(mask.numel())
    nonfinite_count = int((mask & ~finite).sum().item())
    definition = (
        "mean_valid_pixel_channel_l1"
        if config.kind is RobustLossKind.L1
        else "mean_valid_pixel_channel_charbonnier"
    )
    if valid_count == 0:
        return MaskedLossResult(None, 0, 0, total_count, nonfinite_count, True, definition)
    residual = predicted - observation
    if config.kind is RobustLossKind.L1:
        rho = residual.abs()
    else:
        rho = torch.sqrt(residual.square() + config.charbonnier_epsilon**2)
    expanded = valid.expand_as(rho)
    element_count = int(expanded.sum().item())
    value = rho[expanded].sum() / element_count
    if not bool(torch.isfinite(value)):
        raise ValueError("masked cross-view loss is non-finite")
    return MaskedLossResult(
        value,
        valid_count,
        element_count,
        total_count,
        nonfinite_count,
        False,
        definition,
    )


def aggregate_view_losses(
    left: MaskedLossResult,
    right: MaskedLossResult,
    *,
    left_weight: float = 1.0,
    right_weight: float = 1.0,
) -> CrossViewLossResult:
    """Combine available left/right losses without hiding skipped views."""

    if left_weight < 0 or right_weight < 0:
        raise ValueError("view weights must be non-negative")
    terms: list[torch.Tensor] = []
    if left.loss is not None and left_weight:
        terms.append(left.loss * left_weight)
    if right.loss is not None and right_weight:
        terms.append(right.loss * right_weight)
    total = None if not terms else torch.stack(terms).sum()
    return CrossViewLossResult(left, right, total, left_weight, right_weight)


def _validate_image_pair(predicted: object, observation: object) -> None:
    if not isinstance(predicted, torch.Tensor) or not isinstance(observation, torch.Tensor):
        raise TypeError("render and observation must be tensors")
    if predicted.ndim != 4 or observation.shape != predicted.shape:
        raise ValueError("render and observation must share shape [B, C, H, W]")
    if not torch.is_floating_point(predicted) or not torch.is_floating_point(observation):
        raise TypeError("render and observation must be floating-point")
    if predicted.dtype != observation.dtype or predicted.device != observation.device:
        raise ValueError("render and observation must share dtype and device")


class ReferenceRenderer:
    """Small differentiable CPU reference renderer for Plan 08 mechanics.

    It projects Gaussian centers using the explicit camera-to-world contract
    and splats colors with a fixed screen-space kernel.  Covariance is carried
    by the request and validated at the Plan 07 boundary, but is intentionally
    not used to claim native renderer behavior.
    """

    backend_id = "development_reference_renderer.v1"
    production_parity_validated = False

    def __init__(self, *, radius_px: float = 1.0, background: float = 0.0) -> None:
        if radius_px <= 0 or not torch.isfinite(torch.tensor(radius_px)):
            raise ValueError("radius_px must be finite and strictly positive")
        if not torch.isfinite(torch.tensor(background)):
            raise ValueError("background must be finite")
        self.radius_px = float(radius_px)
        self.background = float(background)

    def render(self, request: RendererRequest, camera_request: CameraRequest) -> RenderOutput:
        """Render one explicit request from one explicit camera."""

        if not isinstance(request, RendererRequest):
            raise TypeError("request must be a RendererRequest")
        if not isinstance(camera_request, CameraRequest):
            raise TypeError("camera_request must be a CameraRequest")
        means = request.means3d
        camera = camera_request.camera
        if camera.batch_size != means.shape[0]:
            raise ValueError("camera batch dimension must match renderer request")
        if camera.device != means.device or camera.intrinsics.dtype != means.dtype:
            raise ValueError("camera must share dtype/device with renderer request")
        if not bool(torch.isfinite(camera.intrinsics).all()) or not bool(
            torch.isfinite(camera.world_from_camera).all()
        ):
            raise ValueError("camera tensors must be finite")

        height, width = camera_request.image_size
        rotation_cw = camera.world_from_camera[..., :3, :3]
        translation = camera.world_from_camera[..., :3, 3]
        points_camera = torch.matmul(
            (means - translation.unsqueeze(1)), rotation_cw.transpose(-1, -2)
        )
        z = points_camera[..., 2]
        safe_z = torch.where(z > 0, z, torch.ones_like(z))
        intrinsics = camera.intrinsics
        u = intrinsics[:, 0, 0].unsqueeze(1) * points_camera[..., 0] / safe_z
        u = u + intrinsics[:, 0, 2].unsqueeze(1)
        v = intrinsics[:, 1, 1].unsqueeze(1) * points_camera[..., 1] / safe_z
        v = v + intrinsics[:, 1, 2].unsqueeze(1)
        point_valid = request.valid_mask & (z > 0) & torch.isfinite(u) & torch.isfinite(v)

        yy, xx = torch.meshgrid(
            torch.arange(height, device=means.device, dtype=means.dtype),
            torch.arange(width, device=means.device, dtype=means.dtype),
            indexing="ij",
        )
        du = xx.view(1, 1, height, width) - u.unsqueeze(-1).unsqueeze(-1)
        dv = yy.view(1, 1, height, width) - v.unsqueeze(-1).unsqueeze(-1)
        kernel = torch.exp(-0.5 * (du.square() + dv.square()) / self.radius_px**2)
        weights = kernel * request.opacities.squeeze(-1).unsqueeze(-1).unsqueeze(-1)
        weights = weights * point_valid.unsqueeze(-1).unsqueeze(-1).to(means.dtype)
        denominator = weights.sum(dim=1, keepdim=True)
        colors = request.colors
        if colors is None:
            colors = torch.ones((*means.shape[:2], 1), dtype=means.dtype, device=means.device)
        numer = (weights.unsqueeze(2) * colors.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        background = torch.full_like(numer, self.background)
        image = torch.where(denominator > 0, numer / denominator, background)
        depth_numer = (weights * points_camera[..., 2].unsqueeze(-1).unsqueeze(-1)).sum(
            dim=1, keepdim=True
        )
        depth = torch.where(
            denominator > 0, depth_numer / denominator, torch.zeros_like(denominator)
        )
        visibility = denominator > 1.0e-8
        return RenderOutput(image=image, depth=depth, visibility=visibility)

    __call__ = render


__all__ = [
    "CameraRequest",
    "CameraView",
    "CrossViewCameraRequest",
    "CrossViewLossResult",
    "CrossViewMaskResult",
    "MaskedLossResult",
    "MaskEvidence",
    "ReferenceRenderer",
    "RobustLossConfig",
    "RobustLossKind",
    "aggregate_view_losses",
    "build_cross_view_mask",
    "cross_view_loss",
    "make_camera_request",
]
