"""Forwarding shim for Stage-2 data adapter.

Provides compatibility with imports from ``reliable_endo_gs.data.stage2_adapter``.
All functionality is canonical under :mod:`reliable_endo_gs.data.stage2`.
"""

from __future__ import annotations

from reliable_endo_gs.data.stage2 import (
    DEFAULT_ZFAR,
    DEFAULT_ZNEAR,
    ScaredStage2Dataset,
    ScaredStage2Sample,
    compute_camera_center,
    compute_disp_const,
    compute_fov,
    compute_projection_matrix,
    compute_safe_depth,
    compute_world_view_transform,
    stage1_to_stage2_sample,
    stage2_batch_to_native_input,
    stage2_collate_fn,
    validate_stage2_sample,
)

__all__ = [
    "DEFAULT_ZFAR",
    "DEFAULT_ZNEAR",
    "ScaredStage2Dataset",
    "ScaredStage2Sample",
    "compute_camera_center",
    "compute_disp_const",
    "compute_fov",
    "compute_projection_matrix",
    "compute_safe_depth",
    "compute_world_view_transform",
    "stage1_to_stage2_sample",
    "stage2_batch_to_native_input",
    "stage2_collate_fn",
    "validate_stage2_sample",
]
