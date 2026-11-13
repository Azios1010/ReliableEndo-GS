"""Production boundary for the pinned Graphdeco Gaussian rasterizer.

The adapter deliberately does not construct camera matrices or Gaussian
covariances.  Callers provide an explicit native-camera settings batch, and
Plan 07 owns the covariance fields carried by :class:`RendererRequest`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from math import isfinite
from typing import Any, Protocol

import torch

from reliable_endo_gs.contracts import RenderOutput
from reliable_endo_gs.contracts.deployment import RectifiedStereoCamera
from reliable_endo_gs.probabilistic_gs.marginalization import RepresentationVariant
from reliable_endo_gs.rendering.cross_view import CameraRequest, CameraView
from reliable_endo_gs.rendering.protocol import (
    CovarianceRequest,
    RendererRequest,
)
from reliable_endo_gs.utils.inference_mode import camera_settings_reuse_enabled
from reliable_endo_gs.utils.readback import raise_first_failure, read_scalars

GRAPHDECO_RASTERIZER_REVISION = "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d"
GRAPHDECO_COVARIANCE_ORDER = ("xx", "xy", "xz", "yy", "yz", "zz")
# Pinned auxiliary.h::in_frustum culls p_view.z <= 0.2 in native scene units.
GRAPHDECO_NEAR_THRESHOLD = 0.2
# The pinned rasterizer maps NDC to pixels as ((ndc + 1) * S - 1) / 2 and evaluates pixel index j at coordinate j
# (cuda_rasterizer/auxiliary.h:41-44, forward.cu:233,282,333), i.e. its frustum expects an edge-based principal point. The A00 contract
# uses integer pixel centres (geometry/conventions.py). The frustum is therefore built with the principal point shifted by half a pixel,
# once, here only; K, backprojection and Gaussian centres are untouched. Measured: without it a Gaussian renders 0.500 px up-left.
NATIVE_FRUSTUM_PRINCIPAL_POINT_OFFSET_PX = 0.5


def pack_graphdeco_covariance(covariance: torch.Tensor) -> torch.Tensor:
    """Pack full symmetric covariance into Graphdeco's six-value ABI.

    The scientific representation remains ``[..., 3, 3]``.  Graphdeco's
    native kernels consume one contiguous row per primitive in the pinned
    order ``xx, xy, xz, yy, yz, zz``.  This helper only indexes existing
    entries; it never symmetrizes, detaches, moves, or stabilizes them.
    """

    if not isinstance(covariance, torch.Tensor):
        raise TypeError("covariance must be a torch.Tensor")
    if covariance.ndim < 3 or tuple(covariance.shape[-2:]) != (3, 3):
        raise ValueError(
            "covariance must have shape [..., N, 3, 3] with an explicit primitive dimension; "
            f"got {tuple(covariance.shape)}"
        )
    if not torch.is_floating_point(covariance):
        raise TypeError("covariance must be floating-point")
    if not bool(torch.isfinite(covariance).all()):
        raise ValueError("covariance must contain only finite values")
    if not bool(torch.allclose(covariance, covariance.transpose(-1, -2))):
        raise ValueError("covariance must be symmetric")
    return torch.stack(
        (
            covariance[..., 0, 0],
            covariance[..., 0, 1],
            covariance[..., 0, 2],
            covariance[..., 1, 1],
            covariance[..., 1, 2],
            covariance[..., 2, 2],
        ),
        dim=-1,
    )


class NativeRendererUnavailableError(RuntimeError):
    """Raised when the optional compiled native rasterizer cannot be loaded."""


class NativeRendererConfigurationError(ValueError):
    """Raised when explicit native camera/render settings are incomplete."""


class NativeRendererInputError(ValueError):
    """Raised when a request cannot be represented by the native API."""


def native_camera_batch_from_rectified(
    camera: RectifiedStereoCamera,
    image_hw: tuple[int, int],
    view: CameraView | str = CameraView.LEFT,
    *,
    znear: float = 0.01,
    zfar: float = 100.0,
    native_units_per_metre: float = 1.0,
) -> NativeCameraBatch:
    """Construct Graphdeco settings from the first-party A00 camera contract.

    The projection convention is the explicit optical-axis, y-down A00
    convention.  ``viewmatrix`` is the transposed world-to-camera transform
    expected by the pinned Graphdeco ABI, and ``projmatrix`` is its composed
    view/projection matrix.  No baseline or third-party camera dictionaries are
    imported here. ``znear/zfar`` are in metres. Camera translations and
    clip distances are converted to the declared native scene units; focal
    lengths and the input camera remain unchanged.
    """

    if not isinstance(camera, RectifiedStereoCamera):
        raise TypeError("camera must be a RectifiedStereoCamera")
    view = CameraView(view)
    if len(image_hw) != 2 or any(type(item) is not int or item < 1 for item in image_hw):
        raise ValueError("image_hw must contain positive integer height and width")
    if not isfinite(znear) or not isfinite(zfar) or znear <= 0 or zfar <= znear:
        raise ValueError("znear/zfar must be positive with zfar > znear")
    if not isfinite(native_units_per_metre) or native_units_per_metre <= 0:
        raise ValueError("native_units_per_metre must be finite and positive")
    intrinsics = camera.K_L if view is CameraView.LEFT else camera.K_R
    if view is CameraView.LEFT:
        world_from_camera = torch.eye(4, dtype=intrinsics.dtype, device=intrinsics.device).repeat(
            intrinsics.shape[0], 1, 1
        )
    else:
        world_from_camera = torch.linalg.inv(camera.T_R_from_L)
    # Change native scene units, including the stereo camera translation.
    # A00 camera tensors remain in metres and are never modified in place.
    world_from_camera = world_from_camera.clone()
    world_from_camera[:, :3, 3] *= native_units_per_metre
    znear *= native_units_per_metre
    zfar *= native_units_per_metre
    world_to_camera = torch.linalg.inv(world_from_camera)
    viewmatrix = world_to_camera.transpose(-1, -2).contiguous()
    height, width = image_hw
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2] + NATIVE_FRUSTUM_PRINCIPAL_POINT_OFFSET_PX
    cy = intrinsics[:, 1, 2] + NATIVE_FRUSTUM_PRINCIPAL_POINT_OFFSET_PX
    raise_first_failure(
        [
            (
                (fx > 0).all() & (fy > 0).all(),
                lambda: NativeRendererConfigurationError(
                    "A00 focal lengths must be strictly positive"
                ),
            )
        ]
    )
    near_fx = znear / fx
    near_fy = znear / fy
    left = -(float(width) - cx) * near_fx
    right = cx * near_fx
    bottom = (cy - float(height)) * near_fy
    top = cy * near_fy
    projection = torch.zeros(
        (intrinsics.shape[0], 4, 4), dtype=intrinsics.dtype, device=intrinsics.device
    )
    projection[:, 0, 0] = 2.0 * znear / (right - left)
    projection[:, 1, 1] = 2.0 * znear / (top - bottom)
    projection[:, 0, 2] = (right + left) / (right - left)
    projection[:, 1, 2] = (top + bottom) / (top - bottom)
    projection[:, 3, 2] = 1.0
    projection[:, 2, 2] = zfar / (zfar - znear)
    projection[:, 2, 3] = -(zfar * znear) / (zfar - znear)
    projection = projection.transpose(-1, -2).contiguous()
    return NativeCameraBatch(
        viewmatrix=viewmatrix,
        projmatrix=torch.bmm(viewmatrix, projection).contiguous(),
        campos=world_from_camera[:, :3, 3].contiguous(),
        tanfovx=(float(width) / (2.0 * fx)).contiguous(),
        tanfovy=(float(height) / (2.0 * fy)).contiguous(),
        background=torch.zeros(
            (intrinsics.shape[0], 3), dtype=intrinsics.dtype, device=intrinsics.device
        ),
        native_units_per_metre=native_units_per_metre,
    )


@dataclass(frozen=True, slots=True)
class NativeRasterizationSettings:
    """Settings for one unbatched Graphdeco rasterizer invocation."""

    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    native_units_per_metre: float = 1.0


@dataclass(frozen=True, slots=True)
class NativeCameraBatch:
    """Explicit, already-resolved native camera settings for a request.

    Tensor shapes are ``viewmatrix/projmatrix [B,4,4]``, ``campos [B,3]``,
    ``tanfovx/tanfovy [B]``, and ``background [B,3]``. Translations use
    ``native_units_per_metre``; the ABI dispatcher converts metric Gaussian
    coordinates and support to those same units. No field is derived
    from :class:`~reliable_endo_gs.contracts.CameraBatch` by this adapter.
    """

    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    campos: torch.Tensor
    tanfovx: torch.Tensor
    tanfovy: torch.Tensor
    background: torch.Tensor
    scale_modifier: float = 1.0
    sh_degree: int = 0
    prefiltered: bool = False
    debug: bool = False
    native_units_per_metre: float = 1.0
    # host copies of tanfov per batch index, filled only inside reuse_camera_settings() (see utils.inference_mode)
    _host_tanfov: dict[int, tuple[float, float]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        tensors = {
            "viewmatrix": (self.viewmatrix, (4, 4)),
            "projmatrix": (self.projmatrix, (4, 4)),
            "campos": (self.campos, (3,)),
            "tanfovx": (self.tanfovx, ()),
            "tanfovy": (self.tanfovy, ()),
            "background": (self.background, (3,)),
        }
        batch_size: int | None = None
        finite_flags: list[tuple[str, torch.Tensor]] = []
        for name, (value, trailing) in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim != len(trailing) + 1:
                raise ValueError(f"{name} must have shape [B, {', '.join(map(str, trailing))}]")
            if tuple(value.shape[1:]) != trailing:
                raise ValueError(f"{name} has invalid trailing shape {tuple(value.shape[1:])}")
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must be floating-point")
            if batch_size is None:
                batch_size = int(value.shape[0])
            elif value.shape[0] != batch_size:
                raise ValueError("native camera settings batch dimensions must match")
            if value.device != self.viewmatrix.device:
                raise ValueError("native camera settings must share device")
            if value.dtype != self.viewmatrix.dtype:
                raise TypeError("native camera settings must share dtype")
            finite_flags.append((name, torch.isfinite(value).all()))
        raise_first_failure(
            [
                (flag, lambda n=name: ValueError(f"{n} must be finite"))
                for name, flag in finite_flags
            ]
        )
        if batch_size is None or batch_size < 1:
            raise ValueError("native camera settings batch must be non-empty")
        if self.sh_degree < 0:
            raise ValueError("sh_degree must be non-negative")
        if not torch.isfinite(torch.tensor(self.scale_modifier)) or self.scale_modifier <= 0:
            raise ValueError("scale_modifier must be finite and strictly positive")
        if not isfinite(self.native_units_per_metre) or self.native_units_per_metre <= 0:
            raise ValueError("native_units_per_metre must be finite and positive")

    @property
    def batch_size(self) -> int:
        """Return the number of camera settings."""

        return int(self.viewmatrix.shape[0])

    def item(self, index: int, *, image_size: tuple[int, int]) -> NativeRasterizationSettings:
        """Select one native settings object without changing its tensors."""

        if index < 0 or index >= self.batch_size:
            raise IndexError(f"camera settings index {index} is out of range")
        height, width = image_size
        cached = self._host_tanfov.get(index) if camera_settings_reuse_enabled() else None
        if cached is None:
            tanfovx, tanfovy = read_scalars([self.tanfovx[index], self.tanfovy[index]])
            if camera_settings_reuse_enabled():
                self._host_tanfov[index] = (tanfovx, tanfovy)
        else:
            tanfovx, tanfovy = cached
        if tanfovx <= 0 or tanfovy <= 0:
            raise NativeRendererConfigurationError("tanfovx and tanfovy must be positive")
        return NativeRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.background[index],
            scale_modifier=float(self.scale_modifier),
            viewmatrix=self.viewmatrix[index],
            projmatrix=self.projmatrix[index],
            sh_degree=self.sh_degree,
            campos=self.campos[index],
            prefiltered=self.prefiltered,
            debug=self.debug,
            native_units_per_metre=self.native_units_per_metre,
        )


CameraSettingsProvider = Callable[[CameraRequest], NativeCameraBatch]
CameraSettingsSource = Mapping[CameraView | str, NativeCameraBatch] | CameraSettingsProvider


class NativeRasterizerBackend(Protocol):
    """Injected backend boundary used by production and fake contract tests."""

    def render(
        self,
        settings: NativeRasterizationSettings,
        *,
        means3d: torch.Tensor,
        means2d: torch.Tensor,
        colors_precomp: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor | None,
        rotations: torch.Tensor | None,
        cov3d_precomp: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return native ``(color [3,H,W], radii [N])``."""


def _load_native_module() -> Any:
    try:
        module = import_module("diff_gaussian_rasterization")
    except (ImportError, OSError) as error:
        raise NativeRendererUnavailableError(
            "pinned diff_gaussian_rasterization is unavailable; "
            "install the compiled native backend before production rendering"
        ) from error
    if not hasattr(module, "GaussianRasterizationSettings") or not hasattr(
        module, "GaussianRasterizer"
    ):
        raise NativeRendererUnavailableError(
            "diff_gaussian_rasterization does not expose the pinned Graphdeco API"
        )
    return module


def native_renderer_available() -> bool:
    """Return whether the optional pinned native Python API can be imported."""

    try:
        _load_native_module()
    except NativeRendererUnavailableError:
        return False
    return True


class GraphdecoRasterizerBackend:
    """Thin call-shape adapter for Graphdeco revision ``59f5f77``."""

    backend_id = f"graphdeco_diff_gaussian_rasterization.v{GRAPHDECO_RASTERIZER_REVISION[:7]}"

    def __init__(self, module: Any | None = None) -> None:
        self._module = _load_native_module() if module is None else module
        if not hasattr(self._module, "GaussianRasterizationSettings") or not hasattr(
            self._module, "GaussianRasterizer"
        ):
            raise NativeRendererUnavailableError(
                "native module lacks GaussianRasterizationSettings/GaussianRasterizer"
            )

    def render(
        self,
        settings: NativeRasterizationSettings,
        *,
        means3d: torch.Tensor,
        means2d: torch.Tensor,
        colors_precomp: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor | None,
        rotations: torch.Tensor | None,
        cov3d_precomp: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        settings_type = self._module.GaussianRasterizationSettings
        rasterizer_type = self._module.GaussianRasterizer
        native_settings = settings_type(
            image_height=settings.image_height,
            image_width=settings.image_width,
            tanfovx=settings.tanfovx,
            tanfovy=settings.tanfovy,
            bg=settings.bg,
            scale_modifier=settings.scale_modifier,
            viewmatrix=settings.viewmatrix,
            projmatrix=settings.projmatrix,
            sh_degree=settings.sh_degree,
            campos=settings.campos,
            prefiltered=settings.prefiltered,
            debug=settings.debug,
        )
        rasterizer = rasterizer_type(raster_settings=native_settings)
        result = rasterizer(
            means3D=means3d,
            means2D=means2d,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3d_precomp,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("native rasterizer must return exactly (color, radii)")
        color, radii = result
        if not isinstance(color, torch.Tensor) or not isinstance(radii, torch.Tensor):
            raise RuntimeError("native rasterizer returned non-tensor output")
        return color, radii


class ProductionRenderer:
    """Render canonical Plan 07 requests through the optional native backend."""

    backend_id = GraphdecoRasterizerBackend.backend_id
    production_parity_validated = False

    def __init__(
        self,
        camera_settings: CameraSettingsSource | None = None,
        *,
        backend: NativeRasterizerBackend | None = None,
    ) -> None:
        self.camera_settings = camera_settings
        self._backend = backend

    @property
    def native_available(self) -> bool:
        """Return availability of the injected or default native backend."""

        return self._backend is not None or native_renderer_available()

    def render(self, request: RendererRequest, camera_request: CameraRequest) -> RenderOutput:
        """Render one batched request with one unbatched native call per item."""

        if not isinstance(request, RendererRequest):
            raise TypeError("request must be a RendererRequest")
        if not isinstance(camera_request, CameraRequest):
            raise TypeError("camera_request must be a CameraRequest")
        if camera_request.camera.batch_size != request.means3d.shape[0]:
            raise NativeRendererConfigurationError(
                "CameraRequest batch must match renderer request"
            )
        if (
            camera_request.camera.device != request.means3d.device
            or camera_request.camera.intrinsics.dtype != request.means3d.dtype
        ):
            raise NativeRendererConfigurationError(
                "CameraRequest tensors must share dtype/device with renderer request"
            )
        if request.colors is None:
            raise NativeRendererInputError(
                "production Graphdeco rendering requires 3-channel colors_precomp"
            )
        if request.colors.shape[-1] != 3:
            raise NativeRendererInputError("native colors_precomp must have exactly 3 channels")
        raise_first_failure(
            [
                (
                    torch.isfinite(camera_request.camera.intrinsics).all()
                    & torch.isfinite(camera_request.camera.world_from_camera).all(),
                    lambda: NativeRendererConfigurationError(
                        "CameraRequest tensors must be finite"
                    ),
                ),
                (
                    (torch.isfinite(request.colors) | ~request.valid_mask.unsqueeze(-1)).all(),
                    lambda: NativeRendererInputError("colors must be finite on valid primitives"),
                ),
            ]
        )
        self._validate_variant(request)
        camera_settings = self._resolve_camera_settings(camera_request)
        if camera_settings.batch_size != request.means3d.shape[0]:
            raise NativeRendererConfigurationError(
                "native camera settings batch must match renderer request"
            )
        if camera_settings.viewmatrix.device != request.means3d.device:
            raise NativeRendererConfigurationError(
                "native camera settings must share device with renderer request"
            )
        if camera_settings.viewmatrix.dtype != request.means3d.dtype:
            raise NativeRendererConfigurationError(
                "native camera settings must share dtype with renderer request"
            )
        if camera_settings.native_units_per_metre != 1.0 and request.length_unit != "m":
            raise NativeRendererInputError(
                "native_units_per_metre conversion requires a request in metres"
            )
        backend = self._backend if self._backend is not None else GraphdecoRasterizerBackend()

        images: list[torch.Tensor] = []
        visibility: list[torch.Tensor] = []
        batch_size, primitive_count, _ = request.means3d.shape
        for batch_index in range(batch_size):
            settings = camera_settings.item(batch_index, image_size=camera_request.image_size)
            valid = request.valid_mask[batch_index]
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                images.append(
                    settings.bg.view(3, 1, 1).expand(3, settings.image_height, settings.image_width)
                )
                visibility.append(
                    torch.zeros(primitive_count, dtype=torch.int32, device=request.means3d.device)
                )
                continue
            means3d = request.means3d[batch_index, valid_indices].contiguous()
            colors = request.colors[batch_index, valid_indices].contiguous()
            opacities = request.opacities[batch_index, valid_indices].contiguous()
            means2d = torch.zeros_like(means3d, requires_grad=True)
            scales, rotations, covariance = self._native_gaussian_arguments(
                request, batch_index, valid_indices
            )
            # Uniform unit conversion at the ABI boundary preserves screen
            # projection/covariance and autograd. Camera matrices use the same
            # scale; first-party GaussianField/RendererRequest stay in metres.
            units = settings.native_units_per_metre
            if units != 1.0:
                means3d = means3d * units
                scales = None if scales is None else scales * units
                covariance = None if covariance is None else covariance * (units * units)
            color, radii = backend.render(
                settings,
                means3d=means3d,
                means2d=means2d,
                colors_precomp=colors,
                opacities=opacities,
                scales=scales,
                rotations=rotations,
                cov3d_precomp=covariance,
            )
            self._validate_native_output(
                color,
                radii,
                settings=settings,
                expected_count=int(valid_indices.numel()),
                device=request.means3d.device,
                dtype=request.means3d.dtype,
            )
            full_radii = torch.zeros(primitive_count, dtype=radii.dtype, device=radii.device)
            full_radii[valid_indices] = radii.reshape(-1)
            images.append(color)
            visibility.append(full_radii)
        return RenderOutput(torch.stack(images, dim=0), visibility=torch.stack(visibility, dim=0))

    def _resolve_camera_settings(self, camera_request: CameraRequest) -> NativeCameraBatch:
        source = self.camera_settings
        view = CameraView(camera_request.view)
        if source is None:
            raise NativeRendererConfigurationError(
                "production rendering requires explicit NativeCameraBatch settings; "
                "camera matrices are never inferred from CameraBatch"
            )
        if callable(source):
            settings = source(camera_request)
        else:
            try:
                settings = source[view]
            except KeyError:
                try:
                    settings = source[view.value]
                except KeyError as error:
                    raise NativeRendererConfigurationError(
                        f"no native camera settings for view {view.value!r}"
                    ) from error
        if not isinstance(settings, NativeCameraBatch):
            raise NativeRendererConfigurationError(
                "camera settings provider must return NativeCameraBatch"
            )
        return settings

    @staticmethod
    def _validate_variant(request: RendererRequest) -> None:
        supported = {
            RepresentationVariant.BASELINE,
            RepresentationVariant.COVARIANCE_ONLY,
            RepresentationVariant.STEREO_COVARIANCE_CORRECTED,
        }
        if request.variant not in supported:
            raise NativeRendererInputError(
                f"renderer variant {request.variant.value!r} is deferred or unsupported"
            )
        expected_source = (
            CovarianceRequest.NONE
            if request.variant is RepresentationVariant.BASELINE
            else CovarianceRequest.EFFECTIVE
        )
        if request.covariance_source is not expected_source:
            raise NativeRendererInputError(
                f"variant {request.variant.value!r} requires covariance source "
                f"{expected_source.value!r}"
            )

    @staticmethod
    def _native_gaussian_arguments(
        request: RendererRequest,
        batch_index: int,
        valid_indices: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if request.covariance_source is CovarianceRequest.NONE:
            if request.scales is None or request.rotations is None:
                raise NativeRendererInputError(
                    "baseline production rendering requires canonical scales and rotations"
                )
            rotations = request.rotations[batch_index, valid_indices].contiguous()
            scales = request.scales[batch_index, valid_indices].contiguous()
            rotation_norm = torch.linalg.vector_norm(rotations, dim=-1)
            raise_first_failure(
                [
                    (
                        torch.isfinite(rotations).all()
                        & (rotation_norm > torch.finfo(rotations.dtype).eps).all(),
                        lambda: NativeRendererInputError(
                            "valid baseline rotations must be finite and non-zero"
                        ),
                    ),
                    (
                        torch.isfinite(scales).all() & (scales > 0).all(),
                        lambda: NativeRendererInputError(
                            "valid baseline scales must be finite and positive"
                        ),
                    ),
                ]
            )
            return scales, rotations, None
        if request.covariance is None:
            raise NativeRendererInputError("covariance variant requires canonical covariance")
        covariance = request.covariance[batch_index, valid_indices].contiguous()
        if not bool(torch.isfinite(covariance).all()):
            raise NativeRendererInputError("valid covariance must be finite")
        try:
            packed_covariance = pack_graphdeco_covariance(covariance)
        except (TypeError, ValueError) as error:
            raise NativeRendererInputError(
                f"canonical covariance cannot be packed for Graphdeco: {error}"
            ) from error
        return None, None, packed_covariance

    @staticmethod
    def _validate_native_output(
        color: torch.Tensor,
        radii: torch.Tensor,
        *,
        settings: NativeRasterizationSettings,
        expected_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if color.shape != (3, settings.image_height, settings.image_width):
            raise RuntimeError(
                "native rasterizer color must have shape "
                f"[3,{settings.image_height},{settings.image_width}], got {tuple(color.shape)}"
            )
        if color.dtype != dtype or color.device != device:
            raise RuntimeError("native rasterizer color must preserve request dtype/device")
        if not torch.is_floating_point(color):
            raise RuntimeError("native rasterizer color must be finite floating-point")
        raise_first_failure(
            [
                (
                    torch.isfinite(color).all(),
                    lambda: RuntimeError("native rasterizer color must be finite floating-point"),
                )
            ]
        )
        if radii.ndim != 1 or radii.shape[0] != expected_count:
            raise RuntimeError(
                f"native rasterizer radii must have shape [{expected_count}], got {tuple(radii.shape)}"
            )
        if radii.device != device:
            raise RuntimeError("native rasterizer radii must share device with request")


class ProductionFieldRenderer:
    """First-party ``GaussianField`` to Graphdeco renderer boundary.

    The adapter constructs native camera settings from the A00 contract for
    each request and delegates the ABI call to :class:`ProductionRenderer`.
    An injected backend is supported only for focused tests; with no backend,
    the optional compiled extension is loaded lazily and unavailable status is
    reported as ``NativeRendererUnavailableError``.
    A uniform native scene-unit conversion maps the pinned kernel's 0.2 near
    threshold to ``znear_m``. Metric geometry and historical rendering paths
    remain unchanged. The conversion policy is bound into renderer identity.
    """

    def __init__(
        self,
        *,
        backend: NativeRasterizerBackend | None = None,
        znear_m: float = 0.01,
        zfar_m: float = 100.0,
    ) -> None:
        if not isfinite(znear_m) or not isfinite(zfar_m) or znear_m <= 0 or zfar_m <= znear_m:
            raise ValueError("znear_m/zfar_m must be finite with 0 < znear_m < zfar_m")
        self.backend = backend
        self.znear_m = znear_m
        self.zfar_m = zfar_m
        self.native_units_per_metre = GRAPHDECO_NEAR_THRESHOLD / znear_m
        # (camera tensors are kept alive so their id/storage cannot be recycled while an entry exists)
        self._camera_cache: dict[tuple[object, ...], tuple[tuple[torch.Tensor, ...], NativeCameraBatch]] = {}
        self.identity = (
            f"graphdeco_{GRAPHDECO_RASTERIZER_REVISION[:7]}_a00_camera_v3_halfpixel"
            f"_near_{znear_m:.17g}m_far_{zfar_m:.17g}m"
            f"_units_{self.native_units_per_metre:.17g}"
        )

    @property
    def metric_near_plane_m(self) -> float:
        """Native centre-culling threshold, expressed in first-party metres."""

        return self.znear_m

    @property
    def native_available(self) -> bool:
        return self.backend is not None or native_renderer_available()

    def _native_camera(
        self, camera: RectifiedStereoCamera, image_hw: tuple[int, int], view: CameraView
    ) -> NativeCameraBatch:
        """Native settings for ``camera``; reused inside ``reuse_camera_settings()`` while its tensors are unchanged."""

        def build() -> NativeCameraBatch:
            return native_camera_batch_from_rectified(
                camera,
                image_hw,
                view,
                znear=self.znear_m,
                zfar=self.zfar_m,
                native_units_per_metre=self.native_units_per_metre,
            )

        if not camera_settings_reuse_enabled():
            return build()
        tensors = (camera.K_L, camera.K_R, camera.T_R_from_L)
        key = (view.value, tuple(image_hw)) + tuple(
            (id(t), t.data_ptr(), t._version, tuple(t.shape), t.dtype, str(t.device)) for t in tensors
        )
        hit = self._camera_cache.get(key)
        if hit is not None:
            return hit[1]
        value = build()
        if len(self._camera_cache) >= 16:
            self._camera_cache.pop(next(iter(self._camera_cache)))
        self._camera_cache[key] = (tensors, value)
        return value

    def render(self, request: object) -> RenderOutput:
        from reliable_endo_gs.rendering.field import ExistingRendererBridge, FieldRenderRequest

        if not isinstance(request, FieldRenderRequest):
            raise TypeError("production field renderer requires FieldRenderRequest")
        if self.backend is None and not native_renderer_available():
            raise NativeRendererUnavailableError(
                "A04 production renderer requires diff_gaussian_rasterization; "
                "install the pinned CUDA extension before native rendering"
            )
        view = CameraView(request.view)
        settings = self._native_camera(request.camera, request.image_hw, view)
        renderer = ProductionRenderer({view: settings}, backend=self.backend)
        return ExistingRendererBridge(renderer).render(request)


__all__ = [
    "GRAPHDECO_RASTERIZER_REVISION",
    "GRAPHDECO_COVARIANCE_ORDER",
    "CameraSettingsProvider",
    "CameraSettingsSource",
    "GraphdecoRasterizerBackend",
    "NativeCameraBatch",
    "NativeRasterizationSettings",
    "NativeRendererConfigurationError",
    "NativeRendererInputError",
    "NativeRendererUnavailableError",
    "NativeRasterizerBackend",
    "ProductionRenderer",
    "ProductionFieldRenderer",
    "native_camera_batch_from_rectified",
    "pack_graphdeco_covariance",
    "native_renderer_available",
]
