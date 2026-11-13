"""Opt-in reuse of native camera settings: exact values, invalidation, bounded cache, scope-only effect."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import reliable_endo_gs.rendering.production as production
from reliable_endo_gs.contracts.deployment import RectifiedStereoCamera
from reliable_endo_gs.rendering import CameraView, ProductionFieldRenderer
from reliable_endo_gs.utils.inference_mode import (
    camera_settings_reuse_enabled,
    reuse_camera_settings,
)


def _camera(shift: float = 3.0) -> RectifiedStereoCamera:
    left = torch.tensor(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    ).unsqueeze(0)
    right = left.clone()
    right[:, 0, 2] -= shift
    transform = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    transform[:, 0, 3] = -0.1
    return RectifiedStereoCamera(left, right, transform, True)


def _renderer() -> ProductionFieldRenderer:
    return ProductionFieldRenderer(backend=SimpleNamespace())  # the backend is never called here


def _same(a: production.NativeCameraBatch, b: production.NativeCameraBatch) -> bool:
    return all(
        torch.equal(getattr(a, n), getattr(b, n))
        for n in ("viewmatrix", "projmatrix", "campos", "tanfovx", "tanfovy", "background")
    )


def test_outside_the_scope_nothing_is_cached_and_the_scope_resets():
    renderer, camera = _renderer(), _camera()
    first = renderer._native_camera(camera, (6, 8), CameraView.RIGHT)
    second = renderer._native_camera(camera, (6, 8), CameraView.RIGHT)
    assert first is not second and _same(first, second) and not renderer._camera_cache
    with pytest.raises(RuntimeError):
        with reuse_camera_settings():
            assert camera_settings_reuse_enabled()
            raise RuntimeError("the scope must reset after an exception")
    assert not camera_settings_reuse_enabled()


def test_inside_the_scope_values_are_exact_and_reused():
    renderer, camera = _renderer(), _camera()
    reference = renderer._native_camera(camera, (6, 8), CameraView.RIGHT)
    with reuse_camera_settings():
        first = renderer._native_camera(camera, (6, 8), CameraView.RIGHT)
        second = renderer._native_camera(camera, (6, 8), CameraView.RIGHT)
    assert first is second and _same(first, reference)


def test_in_place_camera_change_and_other_view_or_size_invalidate():
    renderer, camera = _renderer(), _camera()
    with reuse_camera_settings():
        before = renderer._native_camera(camera, (6, 8), CameraView.RIGHT)
        camera.T_R_from_L[:, 0, 3] = -0.2  # in-place edit bumps the tensor version
        after = renderer._native_camera(camera, (6, 8), CameraView.RIGHT)
        assert after is not before and not torch.equal(after.viewmatrix, before.viewmatrix)
        assert renderer._native_camera(camera, (6, 8), CameraView.LEFT) is not after
        assert renderer._native_camera(camera, (6, 9), CameraView.RIGHT) is not after
    expected = _renderer()._native_camera(camera, (6, 8), CameraView.RIGHT)
    assert _same(after, expected)


def test_cache_is_bounded_and_keeps_keyed_tensors_alive():
    renderer = _renderer()
    with reuse_camera_settings():
        for index in range(40):
            camera = _camera(shift=1.0 + index)
            built = renderer._native_camera(camera, (6, 8), CameraView.LEFT)
            del camera  # the cache holds the tensors, so their ids/storage cannot be recycled
            assert _same(
                built,
                _renderer()._native_camera(_camera(shift=1.0 + index), (6, 8), CameraView.LEFT),
            )
        assert len(renderer._camera_cache) <= 16


def test_host_tanfov_is_read_once_per_cached_batch_only_inside_the_scope(monkeypatch):
    renderer, camera = _renderer(), _camera()
    calls: list[int] = []
    real = production.read_scalars

    def counting(values):
        calls.append(len(values))
        return real(values)

    monkeypatch.setattr(production, "read_scalars", counting)
    batch = renderer._native_camera(camera, (6, 8), CameraView.LEFT)
    batch.item(0, image_size=(6, 8))
    batch.item(0, image_size=(6, 8))
    outside = len(calls)
    with reuse_camera_settings():
        cached = renderer._native_camera(camera, (6, 8), CameraView.RIGHT)
        first = cached.item(0, image_size=(6, 8))
        second = cached.item(0, image_size=(6, 8))
    inside = len(calls) - outside
    assert outside >= 2  # no memo without the scope
    assert inside == 1 and (first.tanfovx, first.tanfovy) == (second.tanfovx, second.tanfovy)
