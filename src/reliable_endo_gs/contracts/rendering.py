"""Renderer-agnostic output contracts."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from reliable_endo_gs.contracts.common import (
    require_floating,
    require_min_rank,
    require_rank,
    require_same_device,
    require_same_dtype,
    require_shape,
    require_tensor,
)


@dataclass(frozen=True, slots=True)
class RenderOutput:
    """Minimal output for one rendered view.

    ``image`` uses ``[B, C, H, W]``. Optional depth uses ``[B, 1, H, W]``.
    Optional visibility is renderer-agnostic and may be pixel-level or
    Gaussian-level, but it must have a leading batch dimension.
    """

    SCHEMA_NAME: ClassVar[str] = "render_output"
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    image: torch.Tensor
    depth: torch.Tensor | None = None
    visibility: torch.Tensor | None = None

    def __post_init__(self) -> None:
        image = require_tensor("image", self.image)
        require_rank("image", image, 4)
        require_floating("image", image)
        batch_size, channels, height, width = image.shape
        if channels < 1:
            raise ValueError(
                f"image must have shape [B, C, H, W] with C >= 1; got {tuple(image.shape)}"
            )

        if self.depth is not None:
            depth = require_tensor("depth", self.depth)
            require_shape("depth", depth, (batch_size, 1, height, width))
            require_floating("depth", depth)
            require_same_device("depth", depth, "image", image)
            require_same_dtype("depth", depth, "image", image)

        if self.visibility is not None:
            visibility = require_tensor("visibility", self.visibility)
            require_min_rank("visibility", visibility, 2)
            if visibility.shape[0] != batch_size:
                raise ValueError(
                    "visibility batch dimension must match image; "
                    f"expected {batch_size}, got {visibility.shape[0]}"
                )
            require_same_device("visibility", visibility, "image", image)

    @property
    def batch_size(self) -> int:
        """Return ``B``."""

        return int(self.image.shape[0])

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return ``(H, W)``."""

        return int(self.image.shape[2]), int(self.image.shape[3])

    @property
    def device(self) -> torch.device:
        """Return the render tensor device."""

        return self.image.device
