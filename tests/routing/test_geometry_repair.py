"""Deployment candidates: sign, footprint, support, abstention and no mutation."""

from dataclasses import replace

import pytest
import torch

from reliable_endo_gs.actions.geometry_repair import GeometryRepair, GeometryRepairConfig
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera


def _batch(left, right):
    k = torch.tensor([[[100.0, 0, 32], [0, 100.0, 12], [0, 0, 1.0]]])
    kr = k.clone()
    kr[:, 0, 2] -= 2  # raw disparity includes a nonzero principal-point offset
    t = torch.eye(4)[None]
    t[:, 0, 3] = -0.005
    return DeploymentStereoBatch(
        left,
        right,
        RectifiedStereoCamera(k, kr, t, True),
        torch.ones_like(left[:, :1], dtype=torch.bool),
    )


def test_correspondence_recovers_known_shift_with_right_direction_and_core():
    torch.manual_seed(19)
    left = torch.rand(1, 3, 24, 64)
    right = torch.zeros_like(left)
    right[..., :-14] = left[..., 14:]
    batch = _batch(left, right)
    disparity = torch.full_like(left[:, :1], 12.0, requires_grad=True)
    before = disparity.detach().clone()
    core = torch.zeros_like(batch.support)
    core[..., 5:19, 20:50] = True
    action = GeometryRepair(GeometryRepairConfig())
    result = action.apply(batch, disparity, core)
    assert int(result.accepted.sum()) > 100
    assert (result.disparity[result.accepted] == 14).all()
    assert torch.equal(result.disparity[~core], disparity[~core])
    assert torch.equal(disparity, before) and not result.disparity.requires_grad
    assert (result.disparity - disparity).abs().max() <= 2.5


def test_no_support_and_ambiguous_texture_abstain():
    left = torch.full((1, 3, 24, 64), 0.5)
    batch = _batch(left, left.clone())
    disparity = torch.full_like(left[:, :1], 12)
    action = GeometryRepair(GeometryRepairConfig())
    for sample in (batch, replace(batch, support=torch.zeros_like(batch.support))):
        result = action.apply(sample, disparity, torch.ones_like(batch.support))
        assert not result.accepted.any() and torch.equal(result.disparity, disparity)


def test_support_outlier_repair_bounded_and_color_edge_abstains():
    left = torch.full((1, 3, 24, 64), 0.5)
    batch = _batch(left, left.clone())
    disparity = torch.full_like(left[:, :1], 10.0)
    disparity[..., 12, 32] = 14
    core = torch.zeros_like(batch.support)
    core[..., 12, 32] = True
    action = GeometryRepair(GeometryRepairConfig(mode="support_median"))
    result = action.apply(batch, disparity, core)
    assert int(result.accepted.sum()) == 1
    assert result.disparity[0, 0, 12, 32] == 11  # 25% of effective disparity 12
    assert torch.equal(result.disparity[~core], disparity[~core])
    dark = left.clone()
    dark[..., 12, 32] = 0
    other = action.apply(replace(batch, left=dark), disparity, core)
    assert not other.accepted.any() and torch.equal(other.disparity, disparity)


def test_candidate_input_and_config_guards():
    image = torch.full((1, 3, 24, 64), 0.5)
    batch = _batch(image, image)
    action = GeometryRepair(GeometryRepairConfig())
    with pytest.raises(ValueError):
        action.apply(batch, torch.full_like(batch.support.float(), float("nan")), batch.support)
    with pytest.raises(ValueError):
        GeometryRepairConfig(window=4)
    with pytest.raises(ValueError):
        GeometryRepairConfig(effective_fraction=1)
    with pytest.raises(ValueError):
        GeometryRepairConfig(mode="GT")
