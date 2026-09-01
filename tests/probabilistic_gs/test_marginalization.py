import math

import pytest
import torch

from reliable_endo_gs.probabilistic_gs import (
    RepresentationVariant,
    build_probabilistic_representation,
    marginalize,
    rank1_kappa,
)


def _marginal_inputs(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, ...]:
    surface = torch.diag(torch.tensor([4.0, 4.0, 4.0], dtype=dtype)).reshape(1, 1, 3, 3)
    center = torch.zeros((1, 1, 3, 3), dtype=dtype)
    opacity = torch.tensor([[[0.8]]], dtype=dtype)
    valid = torch.ones((1, 1), dtype=torch.bool)
    return surface, center, opacity, valid


def test_zero_center_covariance_recovers_surface_and_opacity() -> None:
    surface, center, opacity, valid = _marginal_inputs()
    result = marginalize(
        surface,
        center,
        opacity,
        valid,
        surface_frame="camera",
        center_frame="camera",
        surface_unit="m",
        center_unit="m",
    )

    assert torch.equal(result.cov_effective, surface)
    assert torch.equal(result.effective_opacity, opacity)
    assert torch.equal(result.kappa, torch.ones_like(result.kappa))
    assert result.valid_mask.tolist() == [[True]]


def test_effective_covariance_adds_center_covariance() -> None:
    surface, center, opacity, valid = _marginal_inputs()
    center[..., 2, 2] = 4.0
    result = marginalize(
        surface,
        center,
        opacity,
        valid,
        surface_frame="camera",
        center_frame="camera",
        surface_unit="m",
        center_unit="m",
    )

    expected = torch.diag(torch.tensor([4.0, 4.0, 8.0], dtype=surface.dtype)).reshape(1, 1, 3, 3)
    assert torch.equal(result.cov_effective, expected)
    assert torch.allclose(result.kappa, torch.full_like(result.kappa, 1 / math.sqrt(2)))
    assert torch.allclose(result.effective_opacity, opacity / math.sqrt(2))


@pytest.mark.parametrize("anisotropic", [False, True])
def test_rank1_determinant_lemma_matches_full_logdet(anisotropic: bool) -> None:
    surface = torch.diag(torch.tensor([1.5, 4.0, 9.0], dtype=torch.float64)).reshape(1, 1, 3, 3)
    if anisotropic:
        angle = math.pi / 5.0
        q = torch.tensor([math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0], dtype=torch.float64)
        from reliable_endo_gs.probabilistic_gs import surface_covariance

        surface = surface_covariance(
            q.reshape(1, 1, 4), torch.tensor([[[1.5, 2.0, 3.0]]], dtype=torch.float64)
        )
    vector = torch.tensor([[[0.2, -0.4, 0.7]]], dtype=torch.float64)
    center = vector.unsqueeze(-1) @ vector.unsqueeze(-2)
    opacity = torch.ones((1, 1, 1), dtype=torch.float64)
    valid = torch.ones((1, 1), dtype=torch.bool)
    full = marginalize(
        surface,
        center,
        opacity,
        valid,
        surface_frame="world",
        center_frame="world",
        surface_unit="m",
        center_unit="m",
    )
    optimized = marginalize(
        surface,
        center,
        opacity,
        valid,
        surface_frame="world",
        center_frame="world",
        surface_unit="m",
        center_unit="m",
        rank1_vector=vector,
        use_rank1_optimization=True,
    )

    assert torch.allclose(optimized.kappa, full.kappa, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        optimized.effective_opacity, full.effective_opacity, atol=1e-12, rtol=1e-12
    )


def test_rank1_kappa_matches_known_quadratic_form() -> None:
    surface = torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3)
    vector = torch.tensor([[[1.0, 2.0, 2.0]]], dtype=torch.float64)

    assert torch.allclose(
        rank1_kappa(surface, vector), torch.full((1, 1), 1 / math.sqrt(10), dtype=torch.float64)
    )


def test_kappa_decreases_with_center_uncertainty() -> None:
    surface, _, opacity, valid = _marginal_inputs()
    kappas = []
    for magnitude in (0.0, 1.0, 2.0, 4.0):
        center = torch.zeros_like(surface)
        center[..., 0, 0] = magnitude**2
        result = marginalize(
            surface,
            center,
            opacity,
            valid,
            surface_frame="camera",
            center_frame="camera",
            surface_unit="m",
            center_unit="m",
        )
        kappas.append(result.kappa.item())
    assert kappas == sorted(kappas, reverse=True)
    assert all(0.0 < value <= 1.0 for value in kappas)


def test_frame_and_unit_mismatch_are_rejected() -> None:
    surface, center, opacity, valid = _marginal_inputs()
    with pytest.raises(ValueError, match="frames"):
        marginalize(
            surface,
            center,
            opacity,
            valid,
            surface_frame="world",
            center_frame="camera",
            surface_unit="m",
            center_unit="m",
        )
    with pytest.raises(ValueError, match="units"):
        marginalize(
            surface,
            center,
            opacity,
            valid,
            surface_frame="world",
            center_frame="world",
            surface_unit="m",
            center_unit="mm",
        )


def test_invalid_center_is_masked_and_not_replaced_with_certainty() -> None:
    surface, center, opacity, valid = _marginal_inputs()
    center[0, 0, 0, 0] = float("nan")
    result = marginalize(
        surface,
        center,
        opacity,
        valid,
        surface_frame="camera",
        center_frame="camera",
        surface_unit="m",
        center_unit="m",
    )

    assert not bool(result.valid_mask.item())
    assert bool(torch.isnan(result.cov_effective).all())
    assert bool(torch.isnan(result.effective_opacity).all())


def test_representation_keeps_surface_center_and_effective_fields_distinct() -> None:
    means = torch.zeros((1, 1, 3), dtype=torch.float64)
    rotations = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    scales = torch.tensor([[[2.0, 2.0, 2.0]]], dtype=torch.float64)
    opacity = torch.tensor([[[0.75]]], dtype=torch.float64)
    center = torch.zeros((1, 1, 3, 3), dtype=torch.float64)
    center[..., 2, 2] = 1.0
    representation = build_probabilistic_representation(
        means,
        rotations,
        scales,
        opacity,
        center,
        torch.ones((1, 1), dtype=torch.bool),
        frame="camera",
        length_unit="m",
    )

    assert representation.variant is RepresentationVariant.STEREO_COVARIANCE_CORRECTED
    assert representation.cov_surface is not representation.cov_center
    assert torch.allclose(representation.cov_effective, representation.cov_surface + center)
    assert representation.provenance.opacity_correction
    assert representation.provenance.frame == "camera"


def test_representation_rejects_undefined_variants_and_invalid_valid_opacity() -> None:
    means = torch.zeros((1, 1, 3), dtype=torch.float32)
    rotations = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    scales = torch.ones((1, 1, 3))
    center = torch.zeros((1, 1, 3, 3))
    valid = torch.ones((1, 1), dtype=torch.bool)
    with pytest.raises(NotImplementedError, match="deferred"):
        build_probabilistic_representation(
            means,
            rotations,
            scales,
            torch.ones((1, 1, 1)),
            center,
            valid,
            frame="camera",
            length_unit="m",
            variant=RepresentationVariant.OPACITY_ONLY,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build_probabilistic_representation(
            means,
            rotations,
            scales,
            torch.tensor([[[1.5]]]),
            center,
            valid,
            frame="camera",
            length_unit="m",
            variant=RepresentationVariant.BASELINE,
        )


def test_marginalization_is_differentiable_for_valid_inputs() -> None:
    surface = torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3).requires_grad_()
    center = torch.tensor(
        [[[[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    opacity = torch.tensor([[[0.6]]], dtype=torch.float64, requires_grad=True)
    result = marginalize(
        surface,
        center,
        opacity,
        torch.ones((1, 1), dtype=torch.bool),
        surface_frame="world",
        center_frame="world",
        surface_unit="m",
        center_unit="m",
    )
    loss = result.cov_effective.sum() + result.effective_opacity.sum()
    loss.backward()

    assert surface.grad is not None and bool(torch.isfinite(surface.grad).all())
    assert center.grad is not None and bool(torch.isfinite(center.grad).all())
    assert opacity.grad is not None and bool(torch.isfinite(opacity.grad).all())
