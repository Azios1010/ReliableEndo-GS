"""Normalized stereo prediction contracts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import torch

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


@dataclass(frozen=True, slots=True)
class StereoPrediction:
    """A baseline-independent disparity prediction.

    Final disparity, validity, and optional calibrated ``sigma_d`` use
    ``[B, 1, H, W]``. Iterations may be one tensor ``[B, K, H, W]`` or an
    explicit sequence of tensors matching final disparity. No upstream hidden
    state is part of this contract.
    """

    SCHEMA_NAME: ClassVar[str] = "stereo_prediction"
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    disparity: torch.Tensor
    valid_mask: torch.Tensor
    disparity_iterations: torch.Tensor | Sequence[torch.Tensor] | None = None
    sigma_d: torch.Tensor | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        disparity = require_tensor("disparity", self.disparity)
        valid_mask = require_tensor("valid_mask", self.valid_mask)
        require_rank("disparity", disparity, 4)
        if disparity.shape[1] != 1:
            raise ValueError(
                f"disparity must have shape [B, 1, H, W]; got {tuple(disparity.shape)}"
            )
        require_shape("valid_mask", valid_mask, tuple(disparity.shape))
        require_floating("disparity", disparity)
        require_bool("valid_mask", valid_mask)
        require_same_device("valid_mask", valid_mask, "disparity", disparity)

        if self.sigma_d is not None:
            sigma_d = require_tensor("sigma_d", self.sigma_d)
            require_shape("sigma_d", sigma_d, tuple(disparity.shape))
            require_floating("sigma_d", sigma_d)
            require_same_device("sigma_d", sigma_d, "disparity", disparity)
            require_same_dtype("sigma_d", sigma_d, "disparity", disparity)

        iterations = self.disparity_iterations
        if isinstance(iterations, torch.Tensor):
            require_rank("disparity_iterations", iterations, 4)
            expected = (
                disparity.shape[0],
                iterations.shape[1],
                disparity.shape[2],
                disparity.shape[3],
            )
            require_shape("disparity_iterations", iterations, expected)
            require_floating("disparity_iterations", iterations)
            require_same_device("disparity_iterations", iterations, "disparity", disparity)
            require_same_dtype("disparity_iterations", iterations, "disparity", disparity)
        elif iterations is not None:
            if isinstance(iterations, (str, bytes)) or not isinstance(iterations, Sequence):
                raise TypeError(
                    "disparity_iterations must be a torch.Tensor, a sequence of tensors, or None"
                )
            frozen_iterations = tuple(iterations)
            for index, value in enumerate(frozen_iterations):
                name = f"disparity_iterations[{index}]"
                tensor = require_tensor(name, value)
                require_shape(name, tensor, tuple(disparity.shape))
                require_floating(name, tensor)
                require_same_device(name, tensor, "disparity", disparity)
                require_same_dtype(name, tensor, "disparity", disparity)
            object.__setattr__(self, "disparity_iterations", frozen_iterations)

        object.__setattr__(
            self, "diagnostics", freeze_mapping(self.diagnostics, name="diagnostics")
        )

    @property
    def batch_size(self) -> int:
        """Return ``B``."""

        return int(self.disparity.shape[0])

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return ``(H, W)``."""

        return int(self.disparity.shape[2]), int(self.disparity.shape[3])

    @property
    def device(self) -> torch.device:
        """Return the prediction device."""

        return self.disparity.device
