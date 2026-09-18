"""SCARED-C cached-frame adapter for the camera-local Stage-2 control.

SCARED-C frame poses are not yet a verified world-frame contract. This adapter
therefore uses an identity world-to-camera transform and supports same-view
Gaussian reconstruction only. It must not support novel-view claims.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from reliable_endo_gs.data.stage2 import ScaredStage2Sample, stage1_to_stage2_sample


def cached_stage1_to_scared_c_stage2_sample(sample: Mapping[str, object]) -> ScaredStage2Sample:
    """Convert one audited cached SCARED-C frame to the upstream Stage-2 schema."""

    required_tensors = ("left", "right", "disparity", "intr", "right_intr", "Q")
    missing = [name for name in required_tensors if not isinstance(sample.get(name), torch.Tensor)]
    if missing:
        raise ValueError(f"cached Stage-1 sample is missing tensor fields: {missing}")
    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("cached Stage-1 sample has no sample_id")
    parts = sample_id.split("/")
    if len(parts) != 4 or parts[0] != "scared_c":
        raise ValueError(f"invalid SCARED-C sample_id: {sample_id!r}")
    return stage1_to_stage2_sample(
        {
            "left": sample["left"],
            "right": sample["right"],
            "disparity": sample["disparity"],
            "intr": sample["intr"],
            "right_intr": sample["right_intr"],
            "Q": sample["Q"],
            "extr": torch.cat(
                (torch.eye(3, dtype=torch.float32), torch.zeros((3, 1), dtype=torch.float32)),
                dim=1,
            ),
            "sample_id": sample_id,
            "name": sample_id,
            "dataset_id": parts[1],
            "keyframe_id": parts[2],
            "frame_id": parts[3],
        }
    )
