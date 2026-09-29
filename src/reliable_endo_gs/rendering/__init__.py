"""Renderer-facing contracts with lazy historical/optional adapters.

The independent ``FieldRenderer`` boundary is import-safe and does not load
the optional probabilistic or native renderer packages. Historical renderer
exports remain available through lazy attribute resolution for existing
callers, while importing :mod:`reliable_endo_gs.rendering.field` stays isolated.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from reliable_endo_gs.rendering.field import FieldRenderer, FieldRenderRequest

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CameraRequest": ("reliable_endo_gs.rendering.cross_view", "CameraRequest"),
    "CameraView": ("reliable_endo_gs.rendering.cross_view", "CameraView"),
    "CovarianceRequest": ("reliable_endo_gs.rendering.protocol", "CovarianceRequest"),
    "CrossViewCameraRequest": ("reliable_endo_gs.rendering.cross_view", "CrossViewCameraRequest"),
    "CrossViewLossResult": ("reliable_endo_gs.rendering.cross_view", "CrossViewLossResult"),
    "CrossViewMaskResult": ("reliable_endo_gs.rendering.cross_view", "CrossViewMaskResult"),
    "GRAPHDECO_COVARIANCE_ORDER": ("reliable_endo_gs.rendering.production", "GRAPHDECO_COVARIANCE_ORDER"),
    "GRAPHDECO_RASTERIZER_REVISION": ("reliable_endo_gs.rendering.production", "GRAPHDECO_RASTERIZER_REVISION"),
    "CameraSettingsProvider": ("reliable_endo_gs.rendering.production", "CameraSettingsProvider"),
    "CameraSettingsSource": ("reliable_endo_gs.rendering.production", "CameraSettingsSource"),
    "GraphdecoRasterizerBackend": ("reliable_endo_gs.rendering.production", "GraphdecoRasterizerBackend"),
    "MaskedLossResult": ("reliable_endo_gs.rendering.cross_view", "MaskedLossResult"),
    "MaskEvidence": ("reliable_endo_gs.rendering.cross_view", "MaskEvidence"),
    "NativeCameraBatch": ("reliable_endo_gs.rendering.production", "NativeCameraBatch"),
    "NativeRasterizationSettings": ("reliable_endo_gs.rendering.production", "NativeRasterizationSettings"),
    "NativeRasterizerBackend": ("reliable_endo_gs.rendering.production", "NativeRasterizerBackend"),
    "NativeRendererConfigurationError": ("reliable_endo_gs.rendering.production", "NativeRendererConfigurationError"),
    "NativeRendererInputError": ("reliable_endo_gs.rendering.production", "NativeRendererInputError"),
    "NativeRendererUnavailableError": ("reliable_endo_gs.rendering.production", "NativeRendererUnavailableError"),
    "ProductionRenderer": ("reliable_endo_gs.rendering.production", "ProductionRenderer"),
    "ProductionFieldRenderer": ("reliable_endo_gs.rendering.production", "ProductionFieldRenderer"),
    "native_camera_batch_from_rectified": (
        "reliable_endo_gs.rendering.production",
        "native_camera_batch_from_rectified",
    ),
    "pack_graphdeco_covariance": ("reliable_endo_gs.rendering.production", "pack_graphdeco_covariance"),
    "ReferenceRenderer": ("reliable_endo_gs.rendering.cross_view", "ReferenceRenderer"),
    "RendererContractResult": ("reliable_endo_gs.rendering.reference", "RendererContractResult"),
    "RendererRequest": ("reliable_endo_gs.rendering.protocol", "RendererRequest"),
    "RobustLossConfig": ("reliable_endo_gs.rendering.cross_view", "RobustLossConfig"),
    "RobustLossKind": ("reliable_endo_gs.rendering.cross_view", "RobustLossKind"),
    "aggregate_view_losses": ("reliable_endo_gs.rendering.cross_view", "aggregate_view_losses"),
    "build_cross_view_mask": ("reliable_endo_gs.rendering.cross_view", "build_cross_view_mask"),
    "cross_view_loss": ("reliable_endo_gs.rendering.cross_view", "cross_view_loss"),
    "make_camera_request": ("reliable_endo_gs.rendering.cross_view", "make_camera_request"),
    "renderer_contract_metadata": ("reliable_endo_gs.rendering.reference", "renderer_contract_metadata"),
    "native_renderer_available": ("reliable_endo_gs.rendering.production", "native_renderer_available"),
    "validate_renderer_request": ("reliable_endo_gs.rendering.reference", "validate_renderer_request"),
}


def __getattr__(name: str) -> Any:
    """Resolve legacy renderer exports only when a caller explicitly requests them."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = ["FieldRenderRequest", "FieldRenderer", *_LAZY_EXPORTS]
