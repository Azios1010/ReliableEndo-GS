"""Camera tensor contracts without projection or geometry algorithms."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from reliable_endo_gs.contracts.common import (
    require_floating,
    require_rank,
    require_same_device,
    require_same_dtype,
    require_tensor,
    require_trailing_shape,
)


@dataclass(frozen=True, slots=True)
class CameraBatch:
    """A batch of calibrated cameras.

    ``intrinsics`` has shape ``[B, 3, 3]`` and ``world_from_camera`` has
    shape ``[B, 4, 4]``. The transform direction is explicitly camera to
    world; this contract never inverts it or assumes unverified axis,
    handedness, pixel-center, or dataset conventions.
    """

    SCHEMA_NAME: ClassVar[str] = "camera_batch"
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    intrinsics: torch.Tensor
    world_from_camera: torch.Tensor

    def __post_init__(self) -> None:
        intrinsics = require_tensor("intrinsics", self.intrinsics)
        world_from_camera = require_tensor("world_from_camera", self.world_from_camera)
        require_rank("intrinsics", intrinsics, 3)
        require_rank("world_from_camera", world_from_camera, 3)
        require_trailing_shape("intrinsics", intrinsics, (3, 3))
        require_trailing_shape("world_from_camera", world_from_camera, (4, 4))
        if intrinsics.shape[0] != world_from_camera.shape[0]:
            raise ValueError(
                "world_from_camera batch dimension must match intrinsics; "
                f"expected {intrinsics.shape[0]}, got {world_from_camera.shape[0]}"
            )
        require_floating("intrinsics", intrinsics)
        require_floating("world_from_camera", world_from_camera)
        require_same_device("world_from_camera", world_from_camera, "intrinsics", intrinsics)
        require_same_dtype("world_from_camera", world_from_camera, "intrinsics", intrinsics)

    @property
    def batch_size(self) -> int:
        """Return ``B`` without inspecting tensor values."""

        return int(self.intrinsics.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the shared camera tensor device."""

        return self.intrinsics.device
