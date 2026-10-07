"""Metric A00 geometry stays metric across native scene-unit conversion."""

from dataclasses import replace
from typing import Any

import pytest
import torch

from reliable_endo_gs.contracts import CameraBatch
from reliable_endo_gs.contracts.deployment import RectifiedStereoCamera
from reliable_endo_gs.contracts.gaussians import GaussianField
from reliable_endo_gs.probabilistic_gs.marginalization import RepresentationVariant
from reliable_endo_gs.rendering.cross_view import CameraRequest
from reliable_endo_gs.rendering.field import FieldRenderRequest
from reliable_endo_gs.rendering.production import (
    GRAPHDECO_NEAR_THRESHOLD,
    NativeRendererInputError,
    ProductionFieldRenderer,
    ProductionRenderer,
    native_camera_batch_from_rectified,
    pack_graphdeco_covariance,
)
from reliable_endo_gs.rendering.protocol import CovarianceRequest, RendererRequest


def _scene() -> FieldRenderRequest:
    intrinsics = torch.tensor([[[100.0, 0.0, 3.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]])
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 0, 3] = -0.004
    camera = RectifiedStereoCamera(intrinsics, intrinsics.clone(), transform, True)
    field = GaussianField(
        means3d=torch.tensor([[[0.001, 0.001, 0.06]]], requires_grad=True),
        scales=torch.tensor([[[0.001, 0.002, 0.0005]]], requires_grad=True),
        rotations=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        opacities=torch.tensor([[[0.8]]]),
        colors=torch.tensor([[[0.2, 0.3, 0.4]]]),
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    return FieldRenderRequest(field, camera, (5, 7))


class CaptureBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def render(self, settings: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append((settings, kwargs))
        support = kwargs["scales"] if kwargs["scales"] is not None else kwargs["cov3d_precomp"]
        signal = kwargs["means3d"].sum() + support.sum()
        image = signal.expand(3, settings.image_height, settings.image_width)
        return image, torch.ones(kwargs["means3d"].shape[0], dtype=torch.int32)


@pytest.mark.parametrize("view", ["left", "right"])
def test_native_scene_units_preserve_projection_support_and_metric_gradients(view: str) -> None:
    original = _scene()
    request = replace(original, view=view)
    backend = CaptureBackend()
    renderer = ProductionFieldRenderer(backend=backend)
    output = renderer.render(request)
    settings, sent = backend.calls[0]
    units = renderer.native_units_per_metre
    assert units == 20.0
    assert settings.native_units_per_metre == units
    torch.testing.assert_close(sent["means3d"], request.field.means3d[0] * units)
    torch.testing.assert_close(sent["scales"], request.field.scales[0] * units)
    metric_camera = native_camera_batch_from_rectified(request.camera, request.image_hw, view)
    metric_points = torch.cat((request.field.means3d[0], torch.ones(1, 1)), dim=-1)
    native_points = torch.cat((sent["means3d"], torch.ones(1, 1)), dim=-1)
    metric_clip = metric_points @ metric_camera.projmatrix[0]
    native_clip = native_points @ settings.projmatrix
    torch.testing.assert_close(native_clip / native_clip[:, 3:], metric_clip / metric_clip[:, 3:])
    metric_view = metric_points @ metric_camera.viewmatrix[0]
    native_view = native_points @ settings.viewmatrix
    torch.testing.assert_close(native_view[:, :3], metric_view[:, :3] * units)
    torch.testing.assert_close(
        sent["scales"] / native_view[:, 2:3], request.field.scales[0] / metric_view[:, 2:3]
    )
    torch.testing.assert_close(settings.campos, metric_camera.campos[0] * units)
    assert bool((native_view[:, 2] > GRAPHDECO_NEAR_THRESHOLD).all())
    torch.testing.assert_close(
        request.field.means3d.detach(), torch.tensor([[[0.001, 0.001, 0.06]]])
    )
    torch.testing.assert_close(
        request.field.scales.detach(), torch.tensor([[[0.001, 0.002, 0.0005]]])
    )
    assert request.camera.baseline.item() == pytest.approx(0.004)
    output.image.sum().backward()
    assert request.field.means3d.grad is not None
    assert request.field.scales.grad is not None
    torch.testing.assert_close(
        request.field.means3d.grad, torch.full_like(request.field.means3d, units * 3 * 5 * 7)
    )
    torch.testing.assert_close(
        request.field.scales.grad, torch.full_like(request.field.scales, units * 3 * 5 * 7)
    )


def test_native_near_plane_matches_declared_metric_plane_and_identity() -> None:
    request = _scene()
    renderer = ProductionFieldRenderer(backend=CaptureBackend(), znear_m=0.005)
    units = renderer.native_units_per_metre
    depths = torch.tensor([0.0025, 0.005, 0.006, 0.06])
    assert torch.equal(depths * units > GRAPHDECO_NEAR_THRESHOLD, depths > renderer.znear_m)
    assert renderer.identity != ProductionFieldRenderer(backend=CaptureBackend()).identity
    assert (
        native_camera_batch_from_rectified(request.camera, request.image_hw).native_units_per_metre
        == 1.0
    )


def test_native_covariance_conversion_uses_squared_units_and_keeps_gradients() -> None:
    scene = _scene()
    covariance = torch.tensor(
        [[[[1e-6, 2e-7, 0.0], [2e-7, 2e-6, 0.0], [0.0, 0.0, 3e-6]]]],
        requires_grad=True,
    )
    request = RendererRequest(
        means3d=scene.field.means3d,
        colors=scene.field.colors,
        opacities=scene.field.opacities,
        valid_mask=scene.field.valid_mask,
        covariance=covariance,
        covariance_source=CovarianceRequest.EFFECTIVE,
        variant=RepresentationVariant.COVARIANCE_ONLY,
        frame="left_camera",
        length_unit="m",
    )
    settings = native_camera_batch_from_rectified(
        scene.camera, scene.image_hw, native_units_per_metre=20.0
    )
    camera = CameraRequest(
        CameraBatch(scene.camera.K_L, torch.eye(4).unsqueeze(0)), "left", scene.image_hw
    )
    backend = CaptureBackend()
    renderer = ProductionRenderer({"left": settings}, backend=backend)
    output = renderer.render(request, camera)
    _, sent = backend.calls[0]
    torch.testing.assert_close(
        sent["cov3d_precomp"], pack_graphdeco_covariance(covariance[0]) * 400.0
    )
    assert sent["scales"] is None and sent["rotations"] is None
    output.image.sum().backward()
    assert covariance.grad is not None and bool(torch.isfinite(covariance.grad).all())
    assert covariance.grad[0, 0, 0, 1] == 400.0 * 3 * 5 * 7
    assert covariance.grad[0, 0, 1, 0] == 0.0
    with pytest.raises(NativeRendererInputError, match="request in metres"):
        renderer.render(replace(request, length_unit="mm"), camera)


@pytest.mark.parametrize(
    "near,far", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.5), (float("nan"), 1.0), (0.01, float("inf"))]
)
def test_native_clip_policy_rejects_invalid_limits(near: float, far: float) -> None:
    with pytest.raises(ValueError, match="znear_m"):
        ProductionFieldRenderer(znear_m=near, zfar_m=far)
