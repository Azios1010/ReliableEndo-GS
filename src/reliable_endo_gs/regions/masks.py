"""Immutable sparse write regions and deployment target-ROI contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from reliable_endo_gs.contracts.regions import TargetROI as TargetROI
from reliable_endo_gs.regions.grid import Region


@dataclass(frozen=True, slots=True)
class MaskRegion(Region):
    """Sparse pixels in core bounding box; halo is READ ONLY.

    Sorted flattened indices are immutable and include no RGB or GT data.
    A component may cross any historical grid boundary. Bounding boxes do not
    authorize writes into holes, background, or other components.
    """

    image_hw: tuple[int, int]
    flat_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        h, w = self.image_hw
        if (
            any(type(v) is not int or v <= 0 for v in self.image_hw)
            or not self.flat_indices
            or any(type(i) is not int or not 0 <= i < h * w for i in self.flat_indices)
            or tuple(sorted(set(self.flat_indices))) != self.flat_indices
        ):
            raise ValueError("mask region requires unique sorted on-grid pixel indices")
        ys = tuple(i // w for i in self.flat_indices)
        xs = tuple(i % w for i in self.flat_indices)
        if self.core != (min(ys), min(xs), max(ys) + 1, max(xs) + 1):
            raise ValueError("mask core must be the exact pixel bounding box")
        y0, x0, y1, x1 = self.halo
        cy0, cx0, cy1, cx1 = self.core
        if not (0 <= y0 <= cy0 < cy1 <= y1 <= h and 0 <= x0 <= cx0 < cx1 <= x1 <= w):
            raise ValueError("mask read halo must contain core and lie on image grid")

    @property
    def pixel_count(self) -> int:
        """Actual writable count, including holes and irregular boundaries."""
        return len(self.flat_indices)

    def mask(self, image: torch.Tensor) -> torch.Tensor:
        """Return sparse bool [1,1,H,W] writes on the image device."""
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[-2:] != self.image_hw:
            raise ValueError("mask region image grid mismatch")
        result = torch.zeros_like(image[:, :1], dtype=torch.bool)
        indices = torch.tensor(self.flat_indices, device=image.device, dtype=torch.long)
        result.reshape(-1).index_fill_(0, indices, True)
        return result

    @classmethod
    def from_indices(
        cls, index: int, image_hw: tuple[int, int], indices: tuple[int, ...], halo: int
    ) -> MaskRegion:
        """Construct tight core and clipped context halo without changing pixels."""
        if type(halo) is not int or halo < 0 or not indices:
            raise ValueError("nonempty indices and nonnegative integer halo required")
        h, w = image_hw
        ys, xs = [i // w for i in indices], [i % w for i in indices]
        core = (min(ys), min(xs), max(ys) + 1, max(xs) + 1)
        y0, x0, y1, x1 = core
        return cls(
            index,
            core,
            (max(0, y0 - halo), max(0, x0 - halo), min(h, y1 + halo), min(w, x1 + halo)),
            image_hw,
            tuple(sorted(indices)),
        )
