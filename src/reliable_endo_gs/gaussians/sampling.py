"""Deterministic source-pixel sampling for the geometry prior.

Sampling owns discrete source correspondence and count caps.  It never uses a
learned score or a random generator, so a policy is reproducible from the
validity map and image shape alone.  Values gathered through a
:class:`SamplingResult` retain autograd connectivity; masks, coordinates and
indices are intentionally hard/discrete.
"""

from dataclasses import dataclass
from typing import Literal

import torch

SamplingMode = Literal["dense", "fixed", "sparse"]


@dataclass(frozen=True, slots=True)
class SamplingPolicy:
    """Configuration for deterministic dense, fixed-grid or sparse sampling.

    ``cap`` is a per-image maximum.  ``max_samples`` is an explicit alias for
    callers that prefer the count terminology; supplying both requires equal
    values.  Dense mode enumerates valid pixels in row-major order, fixed mode
    keeps a regular ``stride`` grid, and sparse mode keeps every ``stride``th
    valid pixel in row-major order.  Invalid pixels are never emitted.
    """

    mode: SamplingMode = "dense"
    stride: int = 1
    # A finite default keeps the A04 reference path from materialising an
    # uncapped full-resolution Gaussian field on a 16 GiB deployment target.
    cap: int | None = 100_000
    max_samples: int | None = None
    offset_xy: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.mode not in ("dense", "fixed", "sparse"):
            raise ValueError(f"unsupported sampling mode: {self.mode!r}")
        if not isinstance(self.stride, int) or isinstance(self.stride, bool) or self.stride < 1:
            raise ValueError("stride must be a positive integer")
        for name, value in (("cap", self.cap), ("max_samples", self.max_samples)):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 1
            ):
                raise ValueError(f"{name} must be None or a positive integer")
        if self.cap is not None and self.max_samples is not None and self.cap != self.max_samples:
            raise ValueError("cap and max_samples must agree when both are supplied")
        if (
            not isinstance(self.offset_xy, tuple)
            or len(self.offset_xy) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) for value in self.offset_xy
            )
        ):
            raise ValueError("offset_xy must be a pair of integers")

    @property
    def effective_cap(self) -> int | None:
        """Return the configured per-image cap."""

        return self.cap if self.cap is not None else self.max_samples


@dataclass(frozen=True, slots=True)
class SamplingResult:
    """Padded deterministic source correspondence.

    ``source_index`` is row-major ``v*W + u`` with ``-1`` in padded slots;
    ``source_xy`` stores integer pixel centres as floating-point ``(u,v)``.
    ``valid_mask`` identifies emitted slots, while ``coverage`` is the same
    hard selection projected back to ``[B,1,H,W]``.  ``eligible_mask`` keeps
    the pre-cap candidates for diagnostics.
    """

    source_index: torch.Tensor
    source_xy: torch.Tensor
    valid_mask: torch.Tensor
    coverage: torch.Tensor
    eligible_mask: torch.Tensor
    counts: torch.Tensor
    policy: SamplingPolicy

    @property
    def indices(self) -> torch.Tensor:
        """Alias for ``source_index``."""

        return self.source_index

    @property
    def coordinates(self) -> torch.Tensor:
        """Alias for ``source_xy``."""

        return self.source_xy

    @property
    def source_coordinates(self) -> torch.Tensor:
        """Explicit alias used by the Gaussian-prior contract."""

        return self.source_xy

    @property
    def selected_count(self) -> torch.Tensor:
        """Alias for per-image selected counts."""

        return self.counts


_DEFAULT_SAMPLING_POLICY = SamplingPolicy()


def _normalise_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must have dtype torch.bool")
    if valid_mask.ndim == 4 and valid_mask.shape[1] == 1:
        valid_mask = valid_mask[:, 0]
    if valid_mask.ndim != 3:
        raise ValueError(f"valid_mask must be [B,H,W] or [B,1,H,W]; got {tuple(valid_mask.shape)}")
    if min(valid_mask.shape) < 1:
        raise ValueError("valid_mask must have positive batch and spatial dimensions")
    return valid_mask


def _candidate_mask(valid: torch.Tensor, policy: SamplingPolicy) -> torch.Tensor:
    batch_size, height, width = valid.shape
    if policy.mode == "dense":
        return valid
    yy = torch.arange(height, device=valid.device)[:, None]
    xx = torch.arange(width, device=valid.device)[None, :]
    regular_grid = ((yy - policy.offset_xy[1]) % policy.stride == 0) & (
        (xx - policy.offset_xy[0]) % policy.stride == 0
    )
    if policy.mode == "fixed":
        return valid & regular_grid.unsqueeze(0)

    # Sparse mode is a validity-list policy rather than a geometric grid:
    # every stride-th valid row-major index is retained.  The rank operation
    # is integer/bool and intentionally creates no gradient path.
    valid_rank = valid.to(torch.int64).cumsum(dim=-1) - 1
    # The row-major rank must continue across rows.
    row_counts = valid.to(torch.int64).sum(dim=-1, keepdim=True)
    prior_rows = torch.cat(
        (
            torch.zeros((batch_size, 1, 1), dtype=torch.int64, device=valid.device),
            row_counts.cumsum(dim=1)[:, :-1],
        ),
        dim=1,
    )
    rank = valid_rank + prior_rows
    return valid & (rank % policy.stride == 0)


def sample_grid(
    valid_mask: torch.Tensor,
    *,
    policy: SamplingPolicy | None = None,
) -> SamplingResult:
    """Select deterministic source pixels from a hard validity map.

    The output has a common padded ``N`` equal to the largest selected count
    in the batch, never larger than the configured cap.  Empty images return
    ``N=0`` with correctly typed tensors.
    """

    resolved_policy = policy or _DEFAULT_SAMPLING_POLICY
    valid = _normalise_mask(valid_mask)
    batch_size, height, width = valid.shape
    eligible = _candidate_mask(valid, resolved_policy)
    flat_eligible = eligible.reshape(batch_size, height * width)
    cap = resolved_policy.effective_cap
    flat_indices: list[torch.Tensor] = []
    counts: list[int] = []
    for batch_index in range(batch_size):
        indices = torch.nonzero(flat_eligible[batch_index], as_tuple=False).flatten()
        if cap is not None:
            indices = indices[:cap]
        flat_indices.append(indices)
        counts.append(int(indices.numel()))
    max_count = max(counts, default=0)
    source_index = torch.full((batch_size, max_count), -1, dtype=torch.long, device=valid.device)
    valid_slots = torch.zeros((batch_size, max_count), dtype=torch.bool, device=valid.device)
    for batch_index, indices in enumerate(flat_indices):
        if indices.numel() == 0:
            continue
        source_index[batch_index, : indices.numel()] = indices
        valid_slots[batch_index, : indices.numel()] = True
    if max_count:
        u = source_index.remainder(width)
        v = torch.div(source_index.clamp_min(0), width, rounding_mode="floor")
        source_xy = torch.stack((u, v), dim=-1).to(dtype=torch.get_default_dtype())
        source_xy = torch.where(valid_slots.unsqueeze(-1), source_xy, torch.zeros_like(source_xy))
    else:
        source_xy = torch.empty(
            (batch_size, 0, 2), dtype=torch.get_default_dtype(), device=valid.device
        )
    coverage = torch.zeros_like(valid)
    for batch_index, indices in enumerate(flat_indices):
        if indices.numel():
            coverage[batch_index].flatten()[indices] = True
    counts_tensor = torch.tensor(counts, dtype=torch.long, device=valid.device)
    return SamplingResult(
        source_index=source_index,
        source_xy=source_xy,
        valid_mask=valid_slots,
        coverage=coverage.unsqueeze(1),
        eligible_mask=eligible.unsqueeze(1),
        counts=counts_tensor,
        policy=resolved_policy,
    )


def gather_source(
    source: torch.Tensor,
    sampling: SamplingResult,
    *,
    channel_first: bool = True,
) -> torch.Tensor:
    """Gather source values using hard correspondence while retaining gradients.

    ``source`` is ``[B,C,H,W]`` by default, or ``[B,H,W,C]`` when
    ``channel_first=False``.  Invalid/padded slots are zero-filled and do not
    alter the source tensor.
    """

    if not isinstance(source, torch.Tensor):
        raise TypeError("source must be a torch.Tensor")
    if source.ndim != 4:
        raise ValueError("source must have rank four")
    batch_size = sampling.source_index.shape[0]
    if source.shape[0] != batch_size:
        raise ValueError("source batch dimension must match sampling")
    if channel_first:
        height, width = source.shape[2:4]
        source_flat = source.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, source.shape[1]
        )
    else:
        height, width = source.shape[1:3]
        source_flat = source.reshape(batch_size, height * width, source.shape[3])
    if sampling.coverage.shape[-2:] != (height, width):
        raise ValueError("source spatial dimensions must match sampling")
    if sampling.source_index.numel() == 0:
        return source_flat.new_empty((batch_size, 0, source_flat.shape[-1]))
    safe_index = sampling.source_index.clamp_min(0)
    gathered = torch.gather(
        source_flat,
        dim=1,
        index=safe_index.unsqueeze(-1).expand(-1, -1, source_flat.shape[-1]),
    )
    return torch.where(sampling.valid_mask.unsqueeze(-1), gathered, torch.zeros_like(gathered))


def deterministic_sample(
    valid_mask: torch.Tensor,
    *,
    policy: SamplingPolicy | None = None,
) -> SamplingResult:
    """Alias for :func:`sample_grid`."""

    return sample_grid(valid_mask, policy=policy)


def sample_source(
    source: torch.Tensor,
    sampling: SamplingResult,
    *,
    channel_first: bool = True,
) -> torch.Tensor:
    """Alias for :func:`gather_source`."""

    return gather_source(source, sampling, channel_first=channel_first)


SamplingConfig = SamplingPolicy
SamplingCorrespondence = SamplingResult
sample_pixels = sample_grid


__all__ = [
    "SamplingMode",
    "SamplingPolicy",
    "SamplingConfig",
    "SamplingResult",
    "SamplingCorrespondence",
    "deterministic_sample",
    "gather_source",
    "sample_grid",
    "sample_pixels",
    "sample_source",
]
