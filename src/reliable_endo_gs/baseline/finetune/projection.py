"""Principal-point-aware projection for the native MVSplat/DepthSplat CUDA decoders.

Both upstream ``render_cuda`` functions build a symmetric frustum from the field of view, so cx and cy never reach the
rasterizer and every rendered image is shifted by (cx + 0.5 - W/2, cy + 0.5 - H/2) pixels relative to the camera model. On
the A00 grid (512x640) that is up to +4.6 px vertically (datasets 1, 2, 3) and -1.7 px horizontally (datasets 6, 7). The
encoder's plane-sweep cost volume uses the full K, so rendering and geometry would disagree.

With normalized intrinsics (u + 0.5) / W the projection matrix needs only the off-axis terms
``P[0, 2] = 2 * cx_n - 1`` and ``P[1, 2] = 2 * cy_n - 1``; the focal scale, the near/far rows and the rasterizer are unchanged.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch


def off_axis_terms(projection: torch.Tensor, normalized_intrinsics: torch.Tensor) -> torch.Tensor:
    """projection [B,4,4] symmetric, intrinsics [B,3,3] normalized -> projection with the principal-point terms."""
    if projection.shape[0] != normalized_intrinsics.shape[0]:
        raise ValueError("projection and intrinsics must share the batch size")
    result = projection.clone()
    result[:, 0, 2] = 2.0 * normalized_intrinsics[:, 0, 2] - 1.0
    result[:, 1, 2] = 2.0 * normalized_intrinsics[:, 1, 2] - 1.0
    return result


def apply_principal_point_projection() -> dict[str, Any]:
    """Patch the native decoder module in this process (launch from the upstream repository). Call once, before rendering."""
    splatting = importlib.import_module("src.model.decoder.cuda_splatting")
    decoder = importlib.import_module("src.model.decoder.decoder_splatting_cuda")
    if getattr(splatting, "_principal_point_patched", False):
        raise RuntimeError("principal-point projection already applied")
    original_projection = splatting.get_projection_matrix
    original_render = splatting.render_cuda
    pending: dict[str, torch.Tensor] = {}

    def projection(near: Any, far: Any, fov_x: Any, fov_y: Any) -> torch.Tensor:
        return off_axis_terms(original_projection(near, far, fov_x, fov_y), pending["intrinsics"])

    def render(extrinsics: Any, intrinsics: Any, *args: Any, **kwargs: Any) -> Any:
        pending["intrinsics"] = intrinsics
        try:
            return original_render(extrinsics, intrinsics, *args, **kwargs)
        finally:
            pending.pop("intrinsics", None)

    splatting.get_projection_matrix = projection
    splatting.render_cuda = render
    decoder.render_cuda = render
    splatting._principal_point_patched = True
    return {
        "patched": [
            "cuda_splatting.get_projection_matrix",
            "cuda_splatting.render_cuda",
            "decoder_splatting_cuda.render_cuda",
        ]
    }
