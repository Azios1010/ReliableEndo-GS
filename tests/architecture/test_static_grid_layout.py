"""Static grid layout (utils.inference_mode.static_grid_layout): fixed-stride sampling keeps one slot per grid cell.

Contract: the grid evaluation of the surface frame equals the dense evaluation sliced at the grid centres, and a prior built in
the scope equals the dynamic prior slot for slot on every valid slot (holes are padding). Outside the scope nothing changes.
"""

import pytest
import torch

from reliable_endo_gs.contracts.deployment import RectifiedStereoCamera
from reliable_endo_gs.gaussians import GeometryPriorPolicy, SamplingPolicy, build_geometry_prior
from reliable_endo_gs.gaussians.sampling import fixed_grid_cells
from reliable_endo_gs.geometry.surface import (
    SurfaceFramePolicy,
    estimate_surface_frame,
    estimate_surface_frame_on_grid,
)
from reliable_endo_gs.utils.inference_mode import static_grid_layout, static_grid_layout_enabled


def _points(height: int, width: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """[1,3,H,W] camera points: a bumpy surface with depth jumps, so every branch of the frame estimate is exercised."""
    generator = torch.Generator().manual_seed(seed)
    v, u = torch.meshgrid(
        torch.arange(height, dtype=dtype), torch.arange(width, dtype=dtype), indexing="ij"
    )
    z = 0.08 + 0.002 * torch.rand((height, width), generator=generator, dtype=dtype)
    z = z + 0.01 * torch.sin(u / 3.0) * torch.cos(v / 4.0)
    z = torch.where(
        torch.rand((height, width), generator=generator, dtype=dtype) < 0.05, z + 0.05, z
    )
    points = torch.stack(((u - width / 2) * z / 100.0, (v - height / 2) * z / 100.0, z), dim=0)
    return points.unsqueeze(0)


def _valid(height: int, width: int, seed: int, hole_rate: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 1)
    return torch.rand((1, 1, height, width), generator=generator) >= hole_rate


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("stride", "offset", "size"),
    [
        (2, (0, 0), (17, 23)),
        (2, (1, 0), (16, 20)),
        (3, (2, 1), (19, 22)),
        (1, (0, 0), (9, 9)),
        (2, (5, -3), (12, 13)),
    ],
)
def test_grid_surface_frame_equals_dense_frame_at_the_grid_centres(dtype, stride, offset, size):
    height, width = size
    points = _points(height, width, dtype, seed=3)
    valid = _valid(height, width, seed=3, hole_rate=0.1)
    policy = SurfaceFramePolicy()
    dense = estimate_surface_frame(points, valid, policy=policy, layout="channel_first")
    grid = estimate_surface_frame_on_grid(
        points, valid, stride=stride, offset_xy=offset, policy=policy, layout="channel_first"
    )
    y0, x0 = offset[1] % stride, offset[0] % stride
    for name in (
        "normals",
        "tangent_1",
        "tangent_2",
        "frame",
        "scales",
        "valid_mask",
        "normal_valid_mask",
        "fallback_mask",
        "discontinuity_mask",
        "degeneracy_mask",
        "tangent_fallback_mask",
        "neighbor_count",
    ):
        expected = getattr(dense, name)[:, y0::stride, x0::stride]
        assert torch.equal(getattr(grid, name), expected), name


def _camera(dtype: torch.dtype) -> RectifiedStereoCamera:
    left = torch.tensor(
        [[100.0, 0.0, 10.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]], dtype=dtype
    ).unsqueeze(0)
    right = left.clone()
    transform = torch.eye(4, dtype=dtype).unsqueeze(0)
    transform[:, 0, 3] = -0.005
    return RectifiedStereoCamera(left, right, transform, True)


def _prior_pair(hole_rate: float, *, sampling: SamplingPolicy, height: int = 16, width: int = 22):
    dtype = torch.float32
    generator = torch.Generator().manual_seed(11)
    disparity = 20.0 + 2.0 * torch.rand((1, 1, height, width), generator=generator, dtype=dtype)
    support = _valid(height, width, seed=5, hole_rate=hole_rate)
    rgb = torch.rand((1, 3, height, width), generator=generator, dtype=dtype)
    policy = GeometryPriorPolicy(sampling=sampling)
    camera = _camera(dtype)
    dynamic = build_geometry_prior(disparity, camera, support, rgb=rgb, policy=policy)
    with static_grid_layout():
        static = build_geometry_prior(disparity, camera, support, rgb=rgb, policy=policy)
    return dynamic, static


_PER_SLOT = (
    "mu",
    "R0",
    "rotations",
    "S0",
    "alpha0",
    "rgb",
    "source_index",
    "source_xy",
    "cov_surface",
)


def test_scope_is_off_by_default_and_restored():
    assert not static_grid_layout_enabled()
    with static_grid_layout():
        assert static_grid_layout_enabled()
    assert not static_grid_layout_enabled()


def test_hole_free_frame_has_exactly_the_dynamic_layout():
    dynamic, static = _prior_pair(0.0, sampling=SamplingPolicy(mode="fixed", stride=2, cap=100_000))
    for name in _PER_SLOT:
        assert torch.equal(getattr(static, name), getattr(dynamic, name)), name
    for name in ("valid_mask", "fallback_mask", "discontinuity_mask", "tangent_fallback_mask"):
        assert torch.equal(getattr(static, name), getattr(dynamic, name)), name
    assert static.diagnostics == dynamic.diagnostics
    assert torch.equal(static.sampling.coverage, dynamic.sampling.coverage)
    assert torch.equal(static.sampling.eligible_mask, dynamic.sampling.eligible_mask)
    assert torch.equal(static.sampling.counts, dynamic.sampling.counts)


def test_holes_are_padding_and_every_valid_slot_equals_the_dynamic_slot():
    dynamic, static = _prior_pair(
        0.3, sampling=SamplingPolicy(mode="fixed", stride=2, cap=100_000, offset_xy=(1, 0))
    )
    keep = static.valid_mask[0]
    assert int(keep.sum()) == int(dynamic.valid_mask.sum()) < static.valid_mask.shape[1]
    for name in _PER_SLOT:
        compact = getattr(static, name)[0][keep]
        assert torch.equal(compact, getattr(dynamic, name)[0][: compact.shape[0]]), name
    # hole slots carry the same finite placeholders that padded slots always had
    holes = ~keep
    assert bool((static.alpha0[0][holes] == 0).all())
    assert bool((static.source_index[0][holes] == -1).all())
    assert bool(torch.isfinite(static.mu).all())
    assert static.diagnostics == dynamic.diagnostics


@pytest.mark.parametrize(
    "sampling",
    [
        SamplingPolicy(mode="dense", stride=1, cap=None),
        SamplingPolicy(mode="sparse", stride=2, cap=None),
        SamplingPolicy(
            mode="fixed", stride=2, cap=10
        ),  # the cap would truncate the grid: data-dependent, stays dynamic
    ],
)
def test_other_policies_keep_the_dynamic_layout(sampling):
    assert fixed_grid_cells(16, 22, sampling) is None
    dynamic, static = _prior_pair(0.2, sampling=sampling)
    for name in _PER_SLOT:
        assert torch.equal(getattr(static, name), getattr(dynamic, name)), name


def test_fixed_grid_cells_follow_the_dynamic_candidate_rule():
    cells = fixed_grid_cells(
        7, 9, SamplingPolicy(mode="fixed", stride=3, cap=None, offset_xy=(-2, 4))
    )
    assert cells == (1, 1, 2, 3)  # rows 1,4 ; columns 1,4,7
