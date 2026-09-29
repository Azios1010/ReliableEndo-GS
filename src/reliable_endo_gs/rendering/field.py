"""Independent metric GaussianField boundary; no model/native dictionaries."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from reliable_endo_gs.contracts.cameras import CameraBatch
from reliable_endo_gs.contracts.deployment import RectifiedStereoCamera
from reliable_endo_gs.contracts.gaussians import GaussianField
from reliable_endo_gs.contracts.rendering import RenderOutput

if TYPE_CHECKING:
    from reliable_endo_gs.rendering.cross_view import CameraRequest
    from reliable_endo_gs.rendering.protocol import RendererRequest


@dataclass(frozen=True, slots=True)
class FieldRenderRequest:
    """Metric left-camera field; normalized wxyz, standard-deviation scales.

    RGB [B,N,3], centres/scales [B,N,3], rotations [B,N,4], alpha [B,N,1].
    Optional uncertainty/covariance fields are rejected: this path has one
    unambiguous R/S support representation. Invalid slots are sanitized before
    dispatch, protecting backends which compute before applying masks.
    """

    field: GaussianField
    camera: RectifiedStereoCamera
    image_hw: tuple[int, int]
    view: str = "left"

    def __post_init__(self) -> None:
        if self.view not in ("left", "right"):
            raise ValueError("view must be left or right")
        if len(self.image_hw) != 2 or any(type(x) is not int or x < 1 for x in self.image_hw):
            raise ValueError("image_hw requires two positive integers")
        f = self.field
        if any(x is not None for x in (f.cov_surface, f.cov_center, f.cov_effective)):
            raise ValueError("field boundary uses R/S support only; covariance is ambiguous")
        if f.colors.shape[-1] != 3 or f.means3d.shape[0] != self.camera.K_L.shape[0]:
            raise ValueError("RGB field and matching camera batch required")
        if f.means3d.dtype != self.camera.K_L.dtype or f.means3d.device != self.camera.K_L.device:
            raise ValueError("field/camera dtype and device must match explicitly")
        for tensor in (f.means3d, f.colors, f.rotations, f.scales, f.opacities):
            if not bool(torch.isfinite(tensor[f.valid_mask]).all()):
                raise ValueError("valid field values must be finite")
        if not bool((f.scales[f.valid_mask] > 0).all()):
            raise ValueError("scales are positive metric standard deviations")
        colors = f.colors[f.valid_mask]
        if not bool(((colors >= 0) & (colors <= 1)).all()):
            raise ValueError("RGB must be in [0,1]")
        alpha = f.opacities[f.valid_mask]
        if not bool(((alpha >= 0) & (alpha <= 1)).all()):
            raise ValueError("alpha must be in [0,1]")
        norms = torch.linalg.vector_norm(f.rotations[f.valid_mask], dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
            raise ValueError("rotations must be normalized wxyz quaternions")


class FieldRenderer(Protocol):
    """Backend consumes only a first-party field request; no teacher or GT."""

    def render(self, request: FieldRenderRequest) -> RenderOutput:
        """Return image [B,3,H,W]; depth is optional, never assumed."""
        ...


class ExistingRenderBackend(Protocol):
    """Structural contract of existing first-party renderer adapters."""

    def render(self, request: "RendererRequest", camera_request: "CameraRequest") -> RenderOutput:
        """Render using the existing request types."""
        ...


class ExistingRendererBridge:
    """Compatibility bridge to existing first-party reference/native adapters.

    Legacy representation tags are confined here, not required of the model.
    No uncertainty computation occurs. A native adapter still needs separately
    validated native camera settings; this bridge does not claim native parity.
    """

    def __init__(self, renderer: ExistingRenderBackend) -> None:
        if not callable(getattr(renderer, "render", None)):
            raise TypeError("renderer must expose render(request, camera)")
        self.renderer = renderer

    @property
    def identity(self) -> str:
        """Expose the concrete backend identity to A04 provenance/checkpoints."""

        identity = getattr(self.renderer, "identity", None)
        if isinstance(identity, str) and identity.strip():
            return identity
        backend_id = getattr(self.renderer, "backend_id", None)
        if isinstance(backend_id, str) and backend_id.strip():
            return backend_id
        raise ValueError("renderer backend must expose a stable identity or backend_id")

    def render(self, request: FieldRenderRequest) -> RenderOutput:
        """Translate explicit rectified left/right cameras and sanitized fields."""
        from reliable_endo_gs.probabilistic_gs.marginalization import RepresentationVariant
        from reliable_endo_gs.rendering.cross_view import CameraRequest
        from reliable_endo_gs.rendering.protocol import CovarianceRequest, RendererRequest

        f = request.field
        mask = f.valid_mask.unsqueeze(-1)
        identity = torch.zeros_like(f.rotations)
        identity[..., 0] = 1
        translated = RendererRequest(
            means3d=torch.where(mask, f.means3d, torch.zeros_like(f.means3d)),
            colors=torch.where(mask, f.colors, torch.zeros_like(f.colors)),
            opacities=torch.where(mask, f.opacities, torch.zeros_like(f.opacities)),
            valid_mask=f.valid_mask,
            covariance_source=CovarianceRequest.NONE,
            covariance=None,
            variant=RepresentationVariant.BASELINE,
            frame="left_camera",
            length_unit="m",
            rotations=torch.where(mask, f.rotations, identity),
            scales=torch.where(mask, f.scales, torch.ones_like(f.scales)),
        )
        c = request.camera
        if request.view == "left":
            world_from_camera = torch.eye(4, dtype=c.K_L.dtype, device=c.K_L.device)
            world_from_camera = world_from_camera.repeat(c.K_L.shape[0], 1, 1)
            intrinsics = c.K_L
        else:
            world_from_camera = torch.linalg.inv(c.T_R_from_L)
            intrinsics = c.K_R
        camera_request = CameraRequest(
            CameraBatch(intrinsics, world_from_camera), request.view, request.image_hw
        )
        output = self.renderer.render(translated, camera_request)
        if not isinstance(output, RenderOutput):
            raise TypeError("backend must return RenderOutput")
        if output.image.shape != (f.means3d.shape[0], 3, *request.image_hw):
            raise ValueError("backend image shape mismatch")
        if output.image.device != f.means3d.device or not bool(torch.isfinite(output.image).all()):
            raise ValueError("backend image must be finite on the field device")
        return output
