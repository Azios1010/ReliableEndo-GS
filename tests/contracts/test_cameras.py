"""CameraBatch structural contract tests."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from reliable_endo_gs.contracts import CameraBatch


def test_valid_camera_batch_preserves_tensors_and_schema() -> None:
    intrinsics = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    transforms = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    cameras = CameraBatch(intrinsics=intrinsics, world_from_camera=transforms)

    assert cameras.intrinsics is intrinsics
    assert cameras.world_from_camera is transforms
    assert cameras.batch_size == 2
    assert cameras.device == torch.device("cpu")
    assert cameras.SCHEMA_VERSION == "1.0"
    with pytest.raises(FrozenInstanceError):
        cameras.intrinsics = transforms  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("intrinsics", [[1.0]], TypeError, "intrinsics must be a torch.Tensor"),
        ("intrinsics", torch.zeros(3, 3), ValueError, "intrinsics must have rank 3"),
        (
            "intrinsics",
            torch.zeros(2, 4, 4),
            ValueError,
            "intrinsics must end with shape",
        ),
        (
            "world_from_camera",
            torch.zeros(2, 3, 3),
            ValueError,
            "world_from_camera must end with shape",
        ),
        ("intrinsics", torch.ones(2, 3, 3, dtype=torch.int64), TypeError, "floating-point"),
    ],
)
def test_camera_rejects_invalid_structure(
    field: str, value: object, error: type[Exception], message: str
) -> None:
    values: dict[str, object] = {
        "intrinsics": torch.zeros(2, 3, 3),
        "world_from_camera": torch.zeros(2, 4, 4),
    }
    values[field] = value
    with pytest.raises(error, match=message):
        CameraBatch(**values)  # type: ignore[arg-type]


def test_camera_rejects_batch_dtype_and_device_mismatch() -> None:
    with pytest.raises(ValueError, match="batch dimension must match"):
        CameraBatch(torch.zeros(2, 3, 3), torch.zeros(1, 4, 4))
    with pytest.raises(TypeError, match="same dtype"):
        CameraBatch(
            torch.zeros(2, 3, 3, dtype=torch.float32),
            torch.zeros(2, 4, 4, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="same device"):
        CameraBatch(torch.zeros(2, 3, 3), torch.zeros(2, 4, 4, device="meta"))
