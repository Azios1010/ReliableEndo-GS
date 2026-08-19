"""StereoBatch structural contract tests."""

import pytest
import torch

from reliable_endo_gs.contracts import CameraBatch, StereoBatch


def _cameras(batch: int = 2, *, device: str = "cpu") -> CameraBatch:
    return CameraBatch(
        torch.eye(3, device=device).repeat(batch, 1, 1),
        torch.eye(4, device=device).repeat(batch, 1, 1),
    )


def _batch(**overrides: object) -> StereoBatch:
    values: dict[str, object] = {
        "left": torch.zeros(2, 3, 4, 5),
        "right": torch.zeros(2, 3, 4, 5),
        "left_camera": _cameras(),
        "right_camera": _cameras(),
        "sample_ids": ["a", "b"],
        "sequence_ids": ["s0", "s1"],
    }
    values.update(overrides)
    return StereoBatch(**values)  # type: ignore[arg-type]


def test_valid_stereo_batch_freezes_ids_and_masks(stereo_batch: StereoBatch) -> None:
    assert tuple(stereo_batch.sample_ids) == ("sample-0", "sample-1")
    assert stereo_batch.batch_size == 2
    assert stereo_batch.spatial_shape == (4, 5)
    assert stereo_batch.left.shape == stereo_batch.right.shape
    with pytest.raises(TypeError):
        stereo_batch.masks["other"] = torch.ones(2, 1, 4, 5, dtype=torch.bool)  # type: ignore[index]


def test_stereo_batch_rejects_image_and_batch_mismatch() -> None:
    with pytest.raises(ValueError, match="left must have rank 4"):
        _batch(left=torch.zeros(2, 4, 5))
    with pytest.raises(ValueError, match="right must have shape"):
        _batch(right=torch.zeros(2, 3, 5, 5))
    with pytest.raises(ValueError, match="left_camera batch dimension"):
        _batch(left_camera=_cameras(1))
    with pytest.raises(ValueError, match="sample_ids must contain 2 items"):
        _batch(sample_ids=["only-one"])


def test_stereo_batch_rejects_optional_map_and_mask_errors() -> None:
    with pytest.raises(ValueError, match="gt_depth must have shape"):
        _batch(gt_depth=torch.zeros(2, 1, 3, 5))
    with pytest.raises(ValueError, match=r"masks\['valid'\] must have shape"):
        _batch(masks={"valid": torch.ones(2, 4, 5, dtype=torch.bool)})
    with pytest.raises(TypeError, match="dtype torch.bool"):
        _batch(masks={"valid": torch.ones(2, 1, 4, 5)})


def test_stereo_batch_rejects_dtype_and_device_mismatch() -> None:
    with pytest.raises(TypeError, match="right must have the same dtype"):
        _batch(right=torch.zeros(2, 3, 4, 5, dtype=torch.float64))
    with pytest.raises(ValueError, match="right must be on the same device"):
        _batch(right=torch.zeros(2, 3, 4, 5, device="meta"))
    with pytest.raises(ValueError, match="left_camera must be on the same device"):
        _batch(left_camera=_cameras(device="meta"))
