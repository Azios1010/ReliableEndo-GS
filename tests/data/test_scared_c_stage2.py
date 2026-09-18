from __future__ import annotations

import pytest
import torch

from reliable_endo_gs.data.scared_c_stage2 import cached_stage1_to_scared_c_stage2_sample


def test_cached_stage1_sample_becomes_camera_local_stage2_sample() -> None:
    height, width = 4, 5
    result = cached_stage1_to_scared_c_stage2_sample(
        {
            "left": torch.zeros((3, height, width), dtype=torch.float32),
            "right": torch.zeros((3, height, width), dtype=torch.float32),
            "disparity": torch.full((1, height, width), 2.0, dtype=torch.float32),
            "intr": torch.tensor(
                [[10.0, 0.0, 2.0], [0.0, 11.0, 2.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            "right_intr": torch.tensor(
                [[10.0, 0.0, 2.0], [0.0, 11.0, 2.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            "Q": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 20.0], [0.0, 0.0, 2.0, 0.0]],
                dtype=torch.float32,
            ),
            "sample_id": "scared_c/dataset_1/keyframe_1/1",
        }
    )

    assert result["name"] == "scared_c/dataset_1/keyframe_1/1"
    assert result["lmain"]["disp_const"] == pytest.approx(10.0)
    assert torch.equal(result["lmain"]["extr"], torch.cat((torch.eye(3), torch.zeros((3, 1))), dim=1))
    assert torch.equal(result["lmain"]["mask"], torch.ones((1, height, width)))


def test_cached_stage1_sample_rejects_invalid_identity() -> None:
    with pytest.raises(ValueError, match="missing tensor fields"):
        cached_stage1_to_scared_c_stage2_sample({"sample_id": "wrong"})
