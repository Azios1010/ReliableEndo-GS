"""Deployment-only inference scope: skip outputs that no consumer of the rendered images reads.

Three scopes, all opt-in. The first two are bit-exact for the rendered outputs; the third is bit-exact on every valid primitive:

``skip_unread_surface_covariance`` makes ``compose_gaussian_attributes`` return ``cov_surface=None``. The renderer boundary
already drops every covariance field before rasterisation (``reconstruction.model._render``, ``actions.repair._render_outcome``),
so images, rotations, scales, opacities and colours are bit-identical; only the unused support covariance (and the
validation reads that come with it) is not computed. The canonical covariance formula keeps its single owner
(``build_surface_covariance``); nothing here re-implements it. Fields composed in this scope are NOT suitable for
covariance-consuming code (metrics, marginalisation), which never runs inside the deployed path.

``reuse_camera_settings`` lets ``ProductionFieldRenderer`` reuse the native camera settings (matrices, projection, host copies of
``tanfov``) while the camera tensors are unchanged. The cache lives on the renderer instance (no module-global state), is keyed
on tensor identity/storage/version with strong references to the keyed tensors, and is bypassed outside this scope.

``static_grid_layout`` keeps every cell of a fixed-stride sampling grid as a primitive slot (cells without a valid centre are holes with
``valid_mask=False``) instead of compacting the valid cells with ``nonzero``, and evaluates the surface frame only at the grid cells that
sampling can select. Slot ``k`` is grid cell ``k`` in row-major order, so a hole-free frame has exactly the dynamic layout; the work saved
(surface frames of the pixels that sampling discards, 3/4 at stride 2) and the removed host-synchronisation point are the two reasons for it.
It is a no-op when the sampling policy is not a fixed grid or when its cap would truncate the grid.

``compiled_kernels`` runs the pure tensor functions marked with ``compiled_when_enabled`` through ``torch.compile`` (Inductor fuses
their elementwise chains). It is a NUMERICAL APPROXIMATION: fused kernels contract and reorder floating-point operations, so results
agree to rounding, not bitwise. Compilation happens at the first call under the scope (never at import), so warm up before a CUDA
graph capture.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

_SKIP_SURFACE_COVARIANCE: ContextVar[bool] = ContextVar(
    "reliable_endo_gs_skip_surface_covariance", default=False
)


@contextmanager
def skip_unread_surface_covariance() -> Iterator[None]:
    """Within this scope, composed Gaussian fields carry no surface covariance."""
    token = _SKIP_SURFACE_COVARIANCE.set(True)
    try:
        yield
    finally:
        _SKIP_SURFACE_COVARIANCE.reset(token)


def surface_covariance_skipped() -> bool:
    return _SKIP_SURFACE_COVARIANCE.get()


_REUSE_CAMERA_SETTINGS: ContextVar[bool] = ContextVar(
    "reliable_endo_gs_reuse_camera_settings", default=False
)


@contextmanager
def reuse_camera_settings() -> Iterator[None]:
    """Within this scope, a renderer may reuse native camera settings built from unchanged camera tensors."""
    token = _REUSE_CAMERA_SETTINGS.set(True)
    try:
        yield
    finally:
        _REUSE_CAMERA_SETTINGS.reset(token)


def camera_settings_reuse_enabled() -> bool:
    return _REUSE_CAMERA_SETTINGS.get()


_STATIC_GRID_LAYOUT: ContextVar[bool] = ContextVar(
    "reliable_endo_gs_static_grid_layout", default=False
)


@contextmanager
def static_grid_layout() -> Iterator[None]:
    """Within this scope, fixed-grid sampling keeps one slot per grid cell (see the module docstring)."""
    token = _STATIC_GRID_LAYOUT.set(True)
    try:
        yield
    finally:
        _STATIC_GRID_LAYOUT.reset(token)


def static_grid_layout_enabled() -> bool:
    return _STATIC_GRID_LAYOUT.get()


_COMPILED_KERNELS: ContextVar[bool] = ContextVar("reliable_endo_gs_compiled_kernels", default=False)


@contextmanager
def compiled_kernels() -> Iterator[None]:
    """Within this scope, functions marked ``compiled_when_enabled`` run through ``torch.compile`` (numerical approximation)."""
    token = _COMPILED_KERNELS.set(True)
    try:
        yield
    finally:
        _COMPILED_KERNELS.reset(token)


def compiled_kernels_enabled() -> bool:
    return _COMPILED_KERNELS.get()


def compiled_when_enabled(function: _F) -> _F:
    """Mark a pure tensor function (no host reads, no context variables, no side effects) as compilable.

    Outside ``compiled_kernels()`` the original function runs unchanged; inside, a ``torch.compile``d copy (built on first use,
    static shapes) runs instead.
    """
    compiled: Callable[..., Any] | None = None

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal compiled
        if not _COMPILED_KERNELS.get():
            return function(*args, **kwargs)
        if compiled is None:
            import torch

            compiled = torch.compile(function, dynamic=False)
        return compiled(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


__all__ = [
    "camera_settings_reuse_enabled",
    "compiled_kernels",
    "compiled_kernels_enabled",
    "compiled_when_enabled",
    "reuse_camera_settings",
    "skip_unread_surface_covariance",
    "static_grid_layout",
    "static_grid_layout_enabled",
    "surface_covariance_skipped",
]
