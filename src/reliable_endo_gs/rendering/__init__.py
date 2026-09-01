"""Renderer-facing contracts and bounded Phase-I cross-view utilities."""

from reliable_endo_gs.rendering.cross_view import (
    CameraRequest,
    CameraView,
    CrossViewCameraRequest,
    CrossViewLossResult,
    CrossViewMaskResult,
    MaskedLossResult,
    MaskEvidence,
    ReferenceRenderer,
    RobustLossConfig,
    RobustLossKind,
    aggregate_view_losses,
    build_cross_view_mask,
    cross_view_loss,
    make_camera_request,
)
from reliable_endo_gs.rendering.protocol import (
    CovarianceRequest,
    RendererRequest,
)
from reliable_endo_gs.rendering.reference import (
    RendererContractResult,
    renderer_contract_metadata,
    validate_renderer_request,
)

__all__ = [
    "CameraRequest",
    "CameraView",
    "CovarianceRequest",
    "CrossViewCameraRequest",
    "CrossViewLossResult",
    "CrossViewMaskResult",
    "MaskedLossResult",
    "MaskEvidence",
    "ReferenceRenderer",
    "RendererContractResult",
    "RendererRequest",
    "RobustLossConfig",
    "RobustLossKind",
    "aggregate_view_losses",
    "build_cross_view_mask",
    "cross_view_loss",
    "make_camera_request",
    "renderer_contract_metadata",
    "validate_renderer_request",
]
