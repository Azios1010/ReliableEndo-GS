import math

import pytest
import torch

from reliable_endo_gs.probabilistic_gs import (
    build_surface_covariance,
    quaternion_to_rotation_matrix,
    surface_covariance,
)


def test_identity_rotation_preserves_anisotropic_scale_squared() -> None:
    rotations = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    scales = torch.tensor([[[2.0, 3.0, 4.0]]], dtype=torch.float64)

    covariance = surface_covariance(rotations, scales)

    expected = torch.diag(torch.tensor([4.0, 9.0, 16.0], dtype=torch.float64)).reshape(1, 1, 3, 3)
    assert torch.equal(covariance, expected)


def test_wxyz_quaternion_rotates_support_covariance() -> None:
    angle = math.pi / 2.0
    rotations = torch.tensor(
        [[[math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]]], dtype=torch.float64
    )
    scales = torch.tensor([[[2.0, 3.0, 4.0]]], dtype=torch.float64)

    covariance = surface_covariance(rotations, scales)

    expected = torch.diag(torch.tensor([9.0, 4.0, 16.0], dtype=torch.float64)).reshape(1, 1, 3, 3)
    assert torch.allclose(covariance, expected, atol=1e-12, rtol=1e-12)


def test_surface_covariance_is_symmetric_positive_definite_and_rotation_invariant() -> None:
    rotations = torch.tensor([[[0.9238795, 0.0, 0.3826834, 0.0]]], dtype=torch.float64)
    scales = torch.tensor([[[0.5, 2.0, 4.0]]], dtype=torch.float64)

    covariance = surface_covariance(rotations, scales)
    eigenvalues = torch.linalg.eigvalsh(covariance)

    assert torch.allclose(covariance, covariance.transpose(-1, -2), atol=1e-12)
    assert bool((eigenvalues > 0).all())
    assert torch.allclose(
        covariance.diagonal(dim1=-2, dim2=-1).sum(-1),
        torch.tensor([[20.25]], dtype=torch.float64),
        atol=1e-5,
    )


def test_invalid_surface_parameters_are_masked_without_jitter() -> None:
    rotations = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    scales = torch.tensor([[[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]]])

    result = build_surface_covariance(rotations, scales)

    assert result.valid_mask.tolist() == [[True, False]]
    assert bool(torch.isnan(result.cov_surface[0, 1]).all())
    assert result.diagnostics.invalid_rotation_count == 1
    assert result.diagnostics.nonpositive_scale_count == 1


def test_quaternion_conversion_rejects_nonfinite_and_zero_norm() -> None:
    with pytest.raises(ValueError, match="finite"):
        quaternion_to_rotation_matrix(torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="norm"):
        quaternion_to_rotation_matrix(torch.zeros(1, 4))


def test_surface_covariance_keeps_dtype_and_supports_autograd() -> None:
    rotations = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    scales = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64, requires_grad=True)

    loss = surface_covariance(rotations, scales).sum()
    loss.backward()

    assert rotations.grad is not None and bool(torch.isfinite(rotations.grad).all())
    assert scales.grad is not None and bool(torch.isfinite(scales.grad).all())
