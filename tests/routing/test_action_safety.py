"""Focused contracts for action validity and inference no-op behavior."""

from dataclasses import replace

import torch

from reliable_endo_gs.actions.repair import preserve_zero_gaussian_residual
from reliable_endo_gs.actions.stereo_matching import MatchingStereoRepair
from reliable_endo_gs.contracts.gaussians import GaussianField
from reliable_endo_gs.gaussians import AttributeResiduals


def test_matching_head_exposes_missing_evidence_as_invalid() -> None:
    head = MatchingStereoRepair(width=2, radius_px=2, step_px=1, temperature=0.1, hidden_width=2)
    left = torch.full((1, 3, 4, 8), 0.5)
    right = left.clone()
    disparity = torch.full((1, 1, 4, 8), 100.0)
    residual = torch.zeros_like(left)
    support = torch.ones((1, 1, 4, 8), dtype=torch.bool)
    with torch.no_grad():
        output, valid = head.forward_pair(
            left,
            right,
            disparity,
            residual,
            (0, 0, 4, 8),
            support,
            torch.zeros((1, 1, 1, 1)),
            return_validity=True,
        )
    assert output.shape == (1, 1, 4, 8)
    assert valid.shape == output.shape and not bool(valid.any())


def test_zero_gaussian_noop_keeps_identity_and_gradients() -> None:
    base_scales = torch.ones(1, 2, 3, requires_grad=True)
    base = GaussianField(
        means3d=torch.zeros(1, 2, 3),
        colors=torch.zeros(1, 2, 3),
        rotations=torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2]),
        scales=base_scales,
        opacities=torch.ones(1, 2, 1),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    drift = torch.full_like(base.scales, 1e-6, requires_grad=True)
    changed = replace(base, scales=base.scales + drift)
    residuals = AttributeResiduals(
        torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.zeros(1, 2, 1)
    )
    result = preserve_zero_gaussian_residual(base, changed, residuals)
    assert torch.equal(result.scales, base.scales)
    drift_grad, base_grad = torch.autograd.grad(result.scales.sum(), (drift, base.scales))
    torch.testing.assert_close(drift_grad, torch.ones_like(drift))
    torch.testing.assert_close(base_grad, torch.ones_like(base.scales))
