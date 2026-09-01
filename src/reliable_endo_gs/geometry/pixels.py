"""Pixel-grid ownership for the declared integer-pixel-centre convention."""

import torch

from reliable_endo_gs.geometry.conventions import GeometryConvention


def pixel_grid(
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    convention: GeometryConvention,
) -> torch.Tensor:
    """Return ``[B, H, W, 2]`` pixel centres ordered as ``(u, v)``.

    Under ``integer_pixel_centers``, the upper-left pixel centre is ``(0, 0)``
    and the right/down axes increase with ``u``/``v`` respectively.
    """

    if batch_size < 1:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    if height < 1 or width < 1:
        raise ValueError(f"height and width must be positive; got ({height}, {width})")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"dtype must be floating-point; got {dtype}")

    # GeometryConvention validates the sole supported pixel convention.
    _ = convention
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((u, v), dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
