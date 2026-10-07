"""Fixed image cores and read halos, without learned or GT-dependent regions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Region:
    """Half-open core/halo bounds (y0, x0, y1, x1) on one output grid."""

    index: int
    core: tuple[int, int, int, int]
    halo: tuple[int, int, int, int]

    @property
    def pixel_count(self) -> int:
        """Number of writable pixels, independent of the read halo."""
        y0, x0, y1, x1 = self.core
        return (y1 - y0) * (x1 - x0)

    def mask(self, image: torch.Tensor) -> torch.Tensor:
        """Return a bool [1,1,H,W] core write mask for batch-size one."""
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("regions require an image [1,C,H,W]")
        y0, x0, y1, x1 = self.core
        if not 0 <= y0 < y1 <= image.shape[-2] or not 0 <= x0 < x1 <= image.shape[-1]:
            raise ValueError("region core lies outside the image grid")
        result = torch.zeros_like(image[:, :1], dtype=torch.bool)
        result[..., y0:y1, x0:x1] = True
        return result


def grid_regions(
    image_hw: tuple[int, int], tile_hw: tuple[int, int], halo: int
) -> tuple[Region, ...]:
    """Partition every pixel into one core, including non-divisible edges."""
    if any(type(x) is not int or x <= 0 for x in (*image_hw, *tile_hw)):
        raise ValueError("image and tile sizes must be positive integers")
    if type(halo) is not int or halo < 0:
        raise ValueError("halo must be a nonnegative integer")
    height, width = image_hw
    regions: list[Region] = []
    for y in range(0, height, tile_hw[0]):
        for x in range(0, width, tile_hw[1]):
            y1, x1 = min(y + tile_hw[0], height), min(x + tile_hw[1], width)
            regions.append(
                Region(
                    len(regions),
                    (y, x, y1, x1),
                    (
                        max(0, y - halo),
                        max(0, x - halo),
                        min(height, y1 + halo),
                        min(width, x1 + halo),
                    ),
                )
            )
    return tuple(regions)
