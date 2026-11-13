"""Regression test with the REAL native rasterizer: rendered Gaussian centroids must sit at the projected pixel coordinate.

Convention (geometry/conventions.py): integer pixel centres, pixel ``i`` is at ``u = i``. A single isotropic Gaussian whose centre projects
analytically to ``(u, v)`` must render with an intensity centroid at array column ``u`` / row ``v``. Historically the production camera builder
omitted the half-pixel term of the native frustum and every centroid was 0.500 px up-left (measured -0.5007/-0.5005 px, 336 renders).

Pre-recorded tolerance (written before the corrected renderer was run): mean absolute offset per axis <= 0.01 px and every single
measurement <= 0.05 px. The corrected probe measured mean -0.0005/-0.0004 px and a maximum of 0.024 px. The test does not use a mocked
backend: it is skipped only when CUDA or the compiled extension is genuinely unavailable.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from reliable_endo_gs.contracts.deployment import RectifiedStereoCamera
from reliable_endo_gs.contracts.gaussians import GaussianField
from reliable_endo_gs.rendering.field import FieldRenderRequest
from reliable_endo_gs.rendering.production import (
    NATIVE_FRUSTUM_PRINCIPAL_POINT_OFFSET_PX,
    ProductionFieldRenderer,
    native_renderer_available,
)

H, W = 512, 640
MARGIN = 24  # keeps the 3-sigma footprint (<= 3.5 px sigma) clear of the image border
TOLERANCE_MEAN_PX = 0.01
TOLERANCE_MAX_PX = 0.05
BASELINE_M = 0.0043641673

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not native_renderer_available(),
    reason="requires CUDA and the pinned native rasterizer extension",
)

CAMERAS = {
    # dataset_7 calibration after the production resize (K_L == K_R), recorded in the pixel-convention probe
    "real_dataset7": (592.0578, 317.7583, 317.7583, 255.88434),
    "synthetic_centred": (592.0583, 319.5, 319.5, 255.5),
    "synthetic_offcentre": (592.0583, 300.25, 305.5, 260.75),  # different principal points per eye
}
INTEGER_PIXELS = [(40, 40), (160, 128), (319, 255), (320, 256), (480, 400), (600, 470)]
FRACTIONAL_PIXELS = [(100.25, 200.5), (250.75, 90.125), (400.5, 300.25), (555.375, 420.875)]
DEPTHS_M = [0.03, 0.06, 0.12]
SIGMAS_PX = [2.0, 3.5]


def _camera(name: str) -> RectifiedStereoCamera:
    f, cx_left, cx_right, cy = CAMERAS[name]
    device = torch.device("cuda")

    def k(cx: float) -> torch.Tensor:
        return torch.tensor(
            [[f, 0, cx], [0, f, cy], [0, 0, 1.0]], dtype=torch.float32, device=device
        )[None]

    t = torch.eye(4, dtype=torch.float32, device=device)[None].clone()
    t[0, 0, 3] = -BASELINE_M
    return RectifiedStereoCamera(k(cx_left), k(cx_right), t, True)


def _project(camera: RectifiedStereoCamera, view: str, point: torch.Tensor) -> tuple[float, float]:
    k = (camera.K_L if view == "left" else camera.K_R)[0].double().cpu()
    p = point.double().cpu()
    if view == "right":
        t = camera.T_R_from_L[0].double().cpu()
        p = t[:3, :3] @ p + t[:3, 3]
    return float(k[0, 0] * p[0] / p[2] + k[0, 2]), float(k[1, 1] * p[1] / p[2] + k[1, 2])


def _render(renderer, camera, view, point, sigma_m):
    field = GaussianField(
        means3d=point.to(device="cuda", dtype=torch.float32).reshape(1, 1, 3),
        colors=torch.ones(1, 1, 3, device="cuda"),
        rotations=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device="cuda"),
        scales=torch.full((1, 1, 3), sigma_m, device="cuda"),
        opacities=torch.full((1, 1, 1), 0.5, device="cuda"),
        valid_mask=torch.ones(1, 1, dtype=torch.bool, device="cuda"),
    )
    with torch.no_grad():
        return (
            renderer.render(FieldRenderRequest(replace(field), camera, (H, W), view))
            .image[0, 0]
            .double()
            .cpu()
        )


def _centroid(image: torch.Tensor) -> tuple[float, float]:
    total = float(image.sum())
    xs = torch.arange(image.shape[1], dtype=torch.float64)
    ys = torch.arange(image.shape[0], dtype=torch.float64)
    return float((image.sum(0) * xs).sum() / total), float((image.sum(1) * ys).sum() / total)


def _offsets(
    camera_name: str,
    view: str,
    renderer: ProductionFieldRenderer,
    camera: RectifiedStereoCamera | None = None,
):
    camera = camera or _camera(camera_name)
    f = float(camera.K_L[0, 0, 0])
    cx, cy = float(camera.K_L[0, 0, 2]), float(camera.K_L[0, 1, 2])
    rows = []
    for z in DEPTHS_M:
        for u, v in [*INTEGER_PIXELS, *FRACTIONAL_PIXELS]:
            point = torch.tensor([(u - cx) * z / f, (v - cy) * z / f, z], dtype=torch.float64)
            pu, pv = _project(camera, view, point)
            if not (MARGIN <= pu <= W - 1 - MARGIN and MARGIN <= pv <= H - 1 - MARGIN):
                continue  # right-view disparity moved the Gaussian out of the image or into the border band
            for sigma_px in SIGMAS_PX:
                image = _render(renderer, camera, view, point, sigma_px * z / f)
                cu, cv = _centroid(image)
                rows.append((cu - pu, cv - pv))
    return rows


@pytest.mark.parametrize("camera_name", sorted(CAMERAS))
@pytest.mark.parametrize("view", ["left", "right"])
def test_native_centroid_matches_integer_pixel_centre_projection(
    camera_name: str, view: str
) -> None:
    rows = _offsets(camera_name, view, ProductionFieldRenderer())
    assert len(rows) >= 40, (
        "too few valid measurements; the fixture grid no longer covers the image"
    )
    du = torch.tensor([r[0] for r in rows])
    dv = torch.tensor([r[1] for r in rows])
    assert (
        abs(float(du.mean())) <= TOLERANCE_MEAN_PX and abs(float(dv.mean())) <= TOLERANCE_MEAN_PX
    ), (float(du.mean()), float(dv.mean()))
    assert (
        float(du.abs().max()) <= TOLERANCE_MAX_PX and float(dv.abs().max()) <= TOLERANCE_MAX_PX
    ), (float(du.abs().max()), float(dv.abs().max()))


def test_native_projection_test_detects_a_half_pixel_error() -> None:
    """Sensitivity control: removing the correction at the camera (principal point - 0.5) must be detected as -0.5 px."""
    camera = _camera("real_dataset7")
    shifted = RectifiedStereoCamera(
        camera.K_L
        - torch.tensor(
            [
                [0, 0, NATIVE_FRUSTUM_PRINCIPAL_POINT_OFFSET_PX],
                [0, 0, NATIVE_FRUSTUM_PRINCIPAL_POINT_OFFSET_PX],
                [0, 0, 0],
            ],
            device="cuda",
        ),
        camera.K_R
        - torch.tensor(
            [
                [0, 0, NATIVE_FRUSTUM_PRINCIPAL_POINT_OFFSET_PX],
                [0, 0, NATIVE_FRUSTUM_PRINCIPAL_POINT_OFFSET_PX],
                [0, 0, 0],
            ],
            device="cuda",
        ),
        camera.T_R_from_L,
        True,
    )
    renderer = ProductionFieldRenderer()
    f, cx, cy = 592.0578, 317.7583, 255.88434  # analytic prediction uses the UNSHIFTED calibration
    offsets = []
    for u, v in INTEGER_PIXELS[:4]:
        z = 0.06
        point = torch.tensor([(u - cx) * z / f, (v - cy) * z / f, z], dtype=torch.float64)
        pu, pv = _project(camera, "left", point)
        cu, cv = _centroid(_render(renderer, shifted, "left", point, 3.0 * z / f))
        offsets.append((cu - pu, cv - pv))
    mean_u = sum(o[0] for o in offsets) / len(offsets)
    mean_v = sum(o[1] for o in offsets) / len(offsets)
    assert abs(mean_u + 0.5) < 0.03 and abs(mean_v + 0.5) < 0.03, (mean_u, mean_v)


def test_renderer_identity_marks_the_corrected_camera_builder() -> None:
    assert "a00_camera_v3_halfpixel" in ProductionFieldRenderer().identity
    assert NATIVE_FRUSTUM_PRINCIPAL_POINT_OFFSET_PX == 0.5
