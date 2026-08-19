"""Gaussian field tensor contracts without covariance construction."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from reliable_endo_gs.contracts.common import (
    require_bool,
    require_floating,
    require_rank,
    require_same_device,
    require_same_dtype,
    require_shape,
    require_tensor,
)


@dataclass(frozen=True, slots=True)
class GaussianField:
    """A batch of Gaussian primitives with structurally validated tensors.

    ``cov_surface`` is spatial support, ``cov_center`` is uncertainty in
    center location, and ``cov_effective`` is a future renderer-facing
    derived covariance. They remain separate optional inputs; this contract
    never computes or substitutes one for another.
    """

    SCHEMA_NAME: ClassVar[str] = "gaussian_field"
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    means3d: torch.Tensor
    colors: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    valid_mask: torch.Tensor
    cov_surface: torch.Tensor | None = None
    cov_center: torch.Tensor | None = None
    cov_effective: torch.Tensor | None = None

    def __post_init__(self) -> None:
        means3d = require_tensor("means3d", self.means3d)
        require_rank("means3d", means3d, 3)
        if means3d.shape[2] != 3:
            raise ValueError(f"means3d must have shape [B, N, 3]; got {tuple(means3d.shape)}")
        require_floating("means3d", means3d)
        batch_size, gaussian_count, _ = means3d.shape

        colors = require_tensor("colors", self.colors)
        require_rank("colors", colors, 3)
        if colors.shape[:2] != (batch_size, gaussian_count) or colors.shape[2] < 1:
            raise ValueError(
                f"colors must have shape [B, N, C] with C >= 1; got {tuple(colors.shape)}"
            )
        require_floating("colors", colors)
        require_same_device("colors", colors, "means3d", means3d)
        require_same_dtype("colors", colors, "means3d", means3d)

        for name, value, expected in (
            ("rotations", self.rotations, (batch_size, gaussian_count, 4)),
            ("scales", self.scales, (batch_size, gaussian_count, 3)),
            ("opacities", self.opacities, (batch_size, gaussian_count, 1)),
        ):
            tensor = require_tensor(name, value)
            require_rank(name, tensor, 3)
            require_shape(name, tensor, expected)
            require_floating(name, tensor)
            require_same_device(name, tensor, "means3d", means3d)
            require_same_dtype(name, tensor, "means3d", means3d)

        valid_mask = require_tensor("valid_mask", self.valid_mask)
        require_shape("valid_mask", valid_mask, (batch_size, gaussian_count))
        require_bool("valid_mask", valid_mask)
        require_same_device("valid_mask", valid_mask, "means3d", means3d)

        covariance_shape = (batch_size, gaussian_count, 3, 3)
        for name, covariance in (
            ("cov_surface", self.cov_surface),
            ("cov_center", self.cov_center),
            ("cov_effective", self.cov_effective),
        ):
            if covariance is None:
                continue
            tensor = require_tensor(name, covariance)
            require_shape(name, tensor, covariance_shape)
            require_floating(name, tensor)
            require_same_device(name, tensor, "means3d", means3d)
            require_same_dtype(name, tensor, "means3d", means3d)

    @property
    def batch_size(self) -> int:
        """Return ``B``."""

        return int(self.means3d.shape[0])

    @property
    def gaussian_count(self) -> int:
        """Return padded primitive count ``N``."""

        return int(self.means3d.shape[1])

    @property
    def device(self) -> torch.device:
        """Return the shared Gaussian tensor device."""

        return self.means3d.device
