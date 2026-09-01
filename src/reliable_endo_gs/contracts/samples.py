"""Input sample contracts for baseline-neutral stereo batches."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import torch

from reliable_endo_gs.contracts.cameras import CameraBatch
from reliable_endo_gs.contracts.common import (
    freeze_mapping,
    require_bool,
    require_floating,
    require_rank,
    require_same_device,
    require_same_dtype,
    require_shape,
    require_tensor,
)


def _freeze_ids(name: str, values: Sequence[str], batch_size: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    frozen = tuple(values)
    if len(frozen) != batch_size:
        raise ValueError(f"{name} must contain {batch_size} items; got {len(frozen)}")
    for value in frozen:
        if not isinstance(value, str):
            raise TypeError(f"{name} entries must be strings; got {type(value).__name__}")
    return frozen


@dataclass(frozen=True, slots=True)
class StereoBatch:
    """A baseline-neutral stereo input batch.

    Images use ``[B, 3, H, W]``. Optional disparity, depth, and named masks
    use ``[B, 1, H, W]``. No normalization, rectification, units, or camera
    convention is inferred by this container.
    """

    SCHEMA_NAME: ClassVar[str] = "stereo_batch"
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    left: torch.Tensor
    right: torch.Tensor
    left_camera: CameraBatch
    right_camera: CameraBatch
    sample_ids: Sequence[str]
    sequence_ids: Sequence[str]
    gt_disparity: torch.Tensor | None = None
    gt_depth: torch.Tensor | None = None
    gt_depth_xyz: torch.Tensor | None = None
    gt_right_depth_xyz: torch.Tensor | None = None
    masks: Mapping[str, torch.Tensor] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        left = require_tensor("left", self.left)
        right = require_tensor("right", self.right)
        require_rank("left", left, 4)
        require_rank("right", right, 4)
        if left.shape[1] != 3:
            raise ValueError(f"left must have shape [B, 3, H, W]; got {tuple(left.shape)}")
        require_shape("right", right, tuple(left.shape))
        require_floating("left", left)
        require_floating("right", right)
        require_same_device("right", right, "left", left)
        require_same_dtype("right", right, "left", left)

        if not isinstance(self.left_camera, CameraBatch):
            raise TypeError("left_camera must be a CameraBatch")
        if not isinstance(self.right_camera, CameraBatch):
            raise TypeError("right_camera must be a CameraBatch")
        batch_size, _, height, width = left.shape
        if self.left_camera.batch_size != batch_size:
            raise ValueError(
                "left_camera batch dimension must match left; "
                f"expected {batch_size}, got {self.left_camera.batch_size}"
            )
        if self.right_camera.batch_size != batch_size:
            raise ValueError(
                "right_camera batch dimension must match right; "
                f"expected {batch_size}, got {self.right_camera.batch_size}"
            )
        if self.left_camera.device != left.device:
            raise ValueError(
                f"left_camera must be on the same device as left ({left.device}); "
                f"got {self.left_camera.device}"
            )
        if self.right_camera.device != right.device:
            raise ValueError(
                f"right_camera must be on the same device as right ({right.device}); "
                f"got {self.right_camera.device}"
            )

        object.__setattr__(
            self, "sample_ids", _freeze_ids("sample_ids", self.sample_ids, batch_size)
        )
        object.__setattr__(
            self, "sequence_ids", _freeze_ids("sequence_ids", self.sequence_ids, batch_size)
        )

        expected_map_shape = (batch_size, 1, height, width)
        expected_xyz_shape = (batch_size, 3, height, width)
        for name, value in (("gt_disparity", self.gt_disparity), ("gt_depth", self.gt_depth)):
            if value is None:
                continue
            tensor = require_tensor(name, value)
            require_shape(name, tensor, expected_map_shape)
            require_floating(name, tensor)
            require_same_device(name, tensor, "left", left)
            require_same_dtype(name, tensor, "left", left)

        if self.gt_depth_xyz is not None:
            tensor = require_tensor("gt_depth_xyz", self.gt_depth_xyz)
            require_shape("gt_depth_xyz", tensor, expected_xyz_shape)
            require_floating("gt_depth_xyz", tensor)
            require_same_device("gt_depth_xyz", tensor, "left", left)
            require_same_dtype("gt_depth_xyz", tensor, "left", left)

        if self.gt_right_depth_xyz is not None:
            tensor = require_tensor("gt_right_depth_xyz", self.gt_right_depth_xyz)
            require_shape("gt_right_depth_xyz", tensor, expected_xyz_shape)
            require_floating("gt_right_depth_xyz", tensor)
            require_same_device("gt_right_depth_xyz", tensor, "right", right)
            require_same_dtype("gt_right_depth_xyz", tensor, "right", right)

        frozen_masks = freeze_mapping(self.masks, name="masks")
        for mask_name, mask_value in frozen_masks.items():
            tensor = require_tensor(f"masks[{mask_name!r}]", mask_value)
            require_shape(f"masks[{mask_name!r}]", tensor, expected_map_shape)
            require_bool(f"masks[{mask_name!r}]", tensor)
            require_same_device(f"masks[{mask_name!r}]", tensor, "left", left)
        object.__setattr__(self, "masks", frozen_masks)

        frozen_metadata = freeze_mapping(self.metadata, name="metadata")
        object.__setattr__(self, "metadata", frozen_metadata)

    @property
    def batch_size(self) -> int:
        """Return the image batch size."""

        return int(self.left.shape[0])

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return ``(H, W)`` for both stereo images."""

        return int(self.left.shape[2]), int(self.left.shape[3])

    @property
    def device(self) -> torch.device:
        """Return the shared device for coupled sample tensors."""

        return self.left.device
