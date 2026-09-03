"""CPU contract tests for the optional production renderer boundary."""

from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import reliable_endo_gs.rendering.production as production
from reliable_endo_gs.contracts import CameraBatch
from reliable_endo_gs.probabilistic_gs import (
    RepresentationVariant,
    build_probabilistic_representation,
    surface_covariance,
)
from reliable_endo_gs.rendering import (
    CameraRequest,
    CameraView,
    CovarianceRequest,
    GraphdecoRasterizerBackend,
    NativeCameraBatch,
    NativeRendererConfigurationError,
    NativeRendererInputError,
    NativeRendererUnavailableError,
    ProductionRenderer,
    RendererRequest,
    native_renderer_available,
)
from reliable_endo_gs.training.phase1 import PhaseIConfig, PhaseIForward


class CaptureBackend:
    """Small CPU fake that records the native-boundary arguments."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def render(self, settings: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append({"settings": settings, **kwargs})
        count = kwargs["means3d"].shape[0]
        image = settings.bg.view(3, 1, 1).expand(3, settings.image_height, settings.image_width)
        radii = torch.arange(1, count + 1, dtype=torch.int32, device=kwargs["means3d"].device)
        return image.clone(), radii


def _camera_request(batch_size: int = 1) -> tuple[CameraRequest, NativeCameraBatch]:
    dtype = torch.float32
    camera = CameraBatch(
        intrinsics=torch.eye(3, dtype=dtype).repeat(batch_size, 1, 1),
        world_from_camera=torch.eye(4, dtype=dtype).repeat(batch_size, 1, 1),
    )
    request = CameraRequest(camera=camera, view=CameraView.LEFT, image_size=(2, 3))
    settings = NativeCameraBatch(
        viewmatrix=torch.eye(4, dtype=dtype).repeat(batch_size, 1, 1),
        projmatrix=torch.eye(4, dtype=dtype).repeat(batch_size, 1, 1),
        campos=torch.zeros(batch_size, 3, dtype=dtype),
        tanfovx=torch.ones(batch_size, dtype=dtype),
        tanfovy=torch.ones(batch_size, dtype=dtype),
        background=torch.zeros(batch_size, 3, dtype=dtype),
    )
    return request, settings


def _representation(
    variant: RepresentationVariant,
    *,
    center: torch.Tensor | None = None,
) -> Any:
    means = torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float32)
    rotations = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    scales = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
    opacities = torch.tensor([[[0.8]]], dtype=torch.float32)
    colors = torch.tensor([[[0.2, 0.3, 0.4]]], dtype=torch.float32)
    valid = torch.ones(1, 1, dtype=torch.bool)
    return build_probabilistic_representation(
        means,
        rotations,
        scales,
        opacities,
        center,
        valid,
        frame="camera",
        length_unit="m",
        variant=variant,
        colors=colors,
    )


def _renderer_request(
    variant: RepresentationVariant, center: torch.Tensor | None = None
) -> RendererRequest:
    representation = _representation(variant, center=center)
    source = (
        CovarianceRequest.NONE
        if variant is RepresentationVariant.BASELINE
        else CovarianceRequest.EFFECTIVE
    )
    return RendererRequest.from_representation(representation, covariance_source=source)


def test_baseline_and_zero_uncertainty_corrected_preserve_native_boundary() -> None:
    zero = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
    baseline = _renderer_request(RepresentationVariant.BASELINE)
    corrected = _renderer_request(RepresentationVariant.STEREO_COVARIANCE_CORRECTED, zero)
    camera, settings = _camera_request()
    backend = CaptureBackend()
    renderer = ProductionRenderer({CameraView.LEFT: settings}, backend=backend)

    renderer.render(baseline, camera)
    renderer.render(corrected, camera)
    first, second = backend.calls
    expected_surface = surface_covariance(baseline.rotations, baseline.scales)
    assert first["cov3d_precomp"] is None
    assert torch.equal(first["scales"], baseline.scales[0])
    assert torch.equal(first["rotations"], baseline.rotations[0])
    assert torch.allclose(second["cov3d_precomp"], expected_surface[0])
    assert torch.allclose(first["opacities"], second["opacities"])
    assert torch.equal(first["colors_precomp"], second["colors_precomp"])


def test_covariance_variant_preserves_full_off_diagonal_covariance() -> None:
    covariance = torch.tensor(
        [[[[2.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 3.0]]]],
        dtype=torch.float32,
    )
    request = RendererRequest(
        means3d=torch.tensor([[[0.0, 0.0, 2.0]]]),
        opacities=torch.tensor([[[0.5]]]),
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
        covariance_source=CovarianceRequest.EFFECTIVE,
        covariance=covariance,
        variant=RepresentationVariant.STEREO_COVARIANCE_CORRECTED,
        frame="camera",
        length_unit="m",
        colors=torch.tensor([[[0.1, 0.2, 0.3]]]),
    )
    camera, settings = _camera_request()
    backend = CaptureBackend()
    ProductionRenderer({"left": settings}, backend=backend).render(request, camera)
    sent = backend.calls[0]["cov3d_precomp"]
    assert torch.equal(sent, covariance[0])
    assert sent[0, 0, 1] != 0


def test_corrected_variant_passes_canonical_effective_opacity() -> None:
    center = torch.diag(torch.tensor([0.5, 0.25, 0.125], dtype=torch.float32)).reshape(1, 1, 3, 3)
    representation = _representation(
        RepresentationVariant.STEREO_COVARIANCE_CORRECTED, center=center
    )
    request = RendererRequest.from_representation(
        representation, covariance_source=CovarianceRequest.EFFECTIVE
    )
    camera, settings = _camera_request()
    backend = CaptureBackend()
    ProductionRenderer({CameraView.LEFT: settings}, backend=backend).render(request, camera)
    opacity = backend.calls[0]["opacities"]
    assert torch.allclose(opacity, representation.effective_opacity[0])
    assert bool((opacity >= 0).all() and (opacity <= 1).all())
    assert bool((opacity < representation.base_opacity[0]).all())


def test_invalid_padded_primitives_are_not_sent_and_visibility_is_scattered() -> None:
    means = torch.tensor([[[0.0, 0.0, 2.0], [float("nan"), 0.0, 2.0]]])
    request = RendererRequest(
        means3d=means,
        opacities=torch.tensor([[[0.5], [float("nan")]]]),
        valid_mask=torch.tensor([[True, False]]),
        covariance_source=CovarianceRequest.EFFECTIVE,
        covariance=torch.tensor(
            [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[float("nan")] * 3] * 3]]
        ),
        variant=RepresentationVariant.COVARIANCE_ONLY,
        frame="camera",
        length_unit="m",
        colors=torch.tensor([[[0.1, 0.2, 0.3], [float("nan")] * 3]]),
    )
    camera, settings = _camera_request()
    backend = CaptureBackend()
    output = ProductionRenderer({CameraView.LEFT: settings}, backend=backend).render(
        request, camera
    )
    assert len(backend.calls) == 1
    assert backend.calls[0]["means3d"].shape == (1, 3)
    assert output.visibility is not None
    assert output.visibility.tolist() == [[1, 0]]


def test_native_dependency_is_lazy_and_missing_backend_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_: str) -> Any:
        raise ModuleNotFoundError("missing native extension")

    monkeypatch.setattr(production, "import_module", missing)
    assert native_renderer_available() is False
    request = _renderer_request(RepresentationVariant.BASELINE)
    camera, settings = _camera_request()
    with pytest.raises(NativeRendererUnavailableError, match="diff_gaussian_rasterization"):
        ProductionRenderer({CameraView.LEFT: settings}).render(request, camera)


def test_missing_camera_settings_does_not_fall_back_to_reference() -> None:
    request = _renderer_request(RepresentationVariant.BASELINE)
    camera, _ = _camera_request()
    with pytest.raises(NativeRendererConfigurationError, match="explicit NativeCameraBatch"):
        ProductionRenderer(backend=CaptureBackend()).render(request, camera)


def test_camera_settings_must_match_request_dtype() -> None:
    request = _renderer_request(RepresentationVariant.BASELINE)
    camera, settings = _camera_request()
    double_settings = NativeCameraBatch(
        viewmatrix=settings.viewmatrix.double(),
        projmatrix=settings.projmatrix.double(),
        campos=settings.campos.double(),
        tanfovx=settings.tanfovx.double(),
        tanfovy=settings.tanfovy.double(),
        background=settings.background.double(),
    )
    with pytest.raises(NativeRendererConfigurationError, match="dtype"):
        ProductionRenderer({CameraView.LEFT: double_settings}, backend=CaptureBackend()).render(
            request, camera
        )


def test_deferred_variant_is_rejected() -> None:
    request = RendererRequest(
        means3d=torch.tensor([[[0.0, 0.0, 2.0]]]),
        opacities=torch.tensor([[[0.5]]]),
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
        covariance_source=CovarianceRequest.NONE,
        covariance=None,
        variant=RepresentationVariant.OPACITY_ONLY,
        frame="camera",
        length_unit="m",
        colors=torch.tensor([[[0.1, 0.2, 0.3]]]),
        rotations=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        scales=torch.ones(1, 1, 3),
    )
    camera, settings = _camera_request()
    with pytest.raises(NativeRendererInputError, match="deferred"):
        ProductionRenderer({CameraView.LEFT: settings}, backend=CaptureBackend()).render(
            request, camera
        )


def test_plan08_renderer_injection_accepts_production_renderer() -> None:
    camera, settings = _camera_request()
    del camera
    renderer = ProductionRenderer({CameraView.LEFT: settings}, backend=CaptureBackend())
    pipeline = PhaseIForward(
        lambda _: None,  # type: ignore[arg-type]
        lambda _: None,  # type: ignore[arg-type]
        renderer,
        config=PhaseIConfig(baseline_m=0.01),
    )
    assert pipeline.renderer is renderer


def test_graphdeco_backend_maps_exact_pinned_call_shape() -> None:
    Settings = namedtuple(
        "GaussianRasterizationSettings",
        "image_height image_width tanfovx tanfovy bg scale_modifier viewmatrix projmatrix sh_degree campos prefiltered debug",
    )

    class FakeRasterizer:
        last: FakeRasterizer | None = None

        def __init__(self, *, raster_settings: Any) -> None:
            self.raster_settings = raster_settings
            self.arguments: dict[str, Any] | None = None
            FakeRasterizer.last = self

        def __call__(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            self.arguments = kwargs
            return torch.zeros(3, 2, 3), torch.ones(kwargs["means3D"].shape[0], dtype=torch.int32)

    module = SimpleNamespace(
        GaussianRasterizationSettings=Settings,
        GaussianRasterizer=FakeRasterizer,
    )
    backend = GraphdecoRasterizerBackend(module)
    camera, settings_batch = _camera_request()
    settings = settings_batch.item(0, image_size=camera.image_size)
    means = torch.tensor([[0.0, 0.0, 2.0]])
    backend.render(
        settings,
        means3d=means,
        means2d=torch.zeros_like(means, requires_grad=True),
        colors_precomp=torch.ones(1, 3),
        opacities=torch.ones(1, 1),
        scales=None,
        rotations=None,
        cov3d_precomp=torch.eye(3).unsqueeze(0),
    )
    assert FakeRasterizer.last is not None
    assert FakeRasterizer.last.arguments is not None
    assert FakeRasterizer.last.arguments["shs"] is None
    assert FakeRasterizer.last.arguments["colors_precomp"].shape == (1, 3)
    assert FakeRasterizer.last.arguments["cov3D_precomp"].shape == (1, 3, 3)
    assert FakeRasterizer.last.arguments["scales"] is None
    assert FakeRasterizer.last.arguments["rotations"] is None
