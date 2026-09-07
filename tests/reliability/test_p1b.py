"""Analytical synthetic tests for the preregistered P1b diagnostics."""

from __future__ import annotations

import pytest
import torch

from reliable_endo_gs.reliability.p1b import (
    P1B_DEVELOPMENT_SEQUENCES,
    P1B_FINAL_SEQUENCES,
    build_p1b_signals,
    compute_geometry_features,
    compute_photometric_features,
    compute_trajectory_features,
    shadow_diagnostics,
    validate_p1b_sequences,
)


def _mask_like(values: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(values[:, :1], dtype=torch.bool)


def _iterations(values: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 3, 1, 1)


def test_dynamics_monotonic_convergence_has_direct_efficient_path() -> None:
    features = compute_trajectory_features(
        _iterations((10.0, 8.0, 7.0)), torch.ones(1, 1, 1, 1, dtype=torch.bool)
    )
    assert features.signals["u1b_path"].score.item() == pytest.approx(3.0)
    assert features.signals["u1b_decay"].score.item() == pytest.approx(0.5, rel=1e-5)
    assert features.signals["u1b_acceleration"].score.item() == pytest.approx(1.0)
    assert features.signals["u1b_directional_agreement"].score.item() == pytest.approx(2.0 / 3.0)
    assert features.signals["u2b_path_efficiency"].score.item() == pytest.approx(1.0)
    assert features.signals["u2b_oscillation"].score.item() == pytest.approx(0.0)


def test_dynamics_stalled_trajectory_is_zero_movement_not_automatically_safe() -> None:
    features = compute_trajectory_features(
        _iterations((5.0, 5.0, 5.0)), torch.ones(1, 1, 1, 1, dtype=torch.bool)
    )
    assert features.signals["u1b_path"].score.item() == pytest.approx(0.0)
    assert features.signals["u1b_acceleration"].score.item() == pytest.approx(0.0)
    assert features.signals["u1b_decay"].score.item() == pytest.approx(0.0)
    assert features.signals["u2b_path_efficiency"].score.item() == pytest.approx(0.0)


def test_dynamics_oscillation_and_increasing_updates_are_retained() -> None:
    oscillating = compute_trajectory_features(
        _iterations((10.0, 8.0, 10.0)), torch.ones(1, 1, 1, 1, dtype=torch.bool)
    )
    increasing = compute_trajectory_features(
        _iterations((10.0, 8.0, 5.0)), torch.ones(1, 1, 1, 1, dtype=torch.bool)
    )
    decreasing = compute_trajectory_features(
        _iterations((10.0, 9.0, 8.5)), torch.ones(1, 1, 1, 1, dtype=torch.bool)
    )
    assert oscillating.signals["u2b_oscillation"].score.item() == pytest.approx(1.0)
    assert oscillating.signals["u2b_path_efficiency"].score.item() == pytest.approx(0.0)
    assert increasing.signals["u1b_decay"].score.item() == pytest.approx(1.5, rel=1e-5)
    assert decreasing.signals["u1b_decay"].score.item() == pytest.approx(0.5, rel=1e-5)


def _constant_stereo(
    *,
    height: int = 2,
    width: int = 6,
    disparity: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left = torch.full((1, 1, height, width), disparity)
    right = torch.full_like(left, disparity)
    return left, right, _mask_like(left), _mask_like(right)


def test_geometry_uses_positive_swapped_magnitude_and_strict_border_support() -> None:
    left, right, left_valid, right_valid = _constant_stereo()
    features = compute_geometry_features(left, right, left_valid, right_valid)
    valid = features.signals["u3b_valid_absolute"].valid_mask
    assert torch.allclose(
        features.signals["u3b_valid_absolute"].score[valid], torch.zeros_like(left[valid])
    )
    assert valid[:, :, :, 0].sum().item() == 0
    assert valid[:, :, :, 1:].all()
    assert features.signals["u3_corrected_raw"].score[valid].max().item() == pytest.approx(0.0)


def test_geometry_relative_scaling_and_invalid_support_are_explicit() -> None:
    left = torch.full((1, 1, 1, 5), 2.0)
    right = torch.full_like(left, 1.0)
    left_valid = _mask_like(left)
    right_valid = _mask_like(right)
    right_valid[..., 0] = False
    features = compute_geometry_features(left, right, left_valid, right_valid)
    relative = features.signals["u3b_relative"]
    assert relative.score[relative.valid_mask].mean().item() == pytest.approx(1.0 / 3.0, rel=1e-5)
    assert relative.valid_mask[..., 1].sum().item() == 0


def test_geometry_varying_disparity_preserves_subpixel_correspondence() -> None:
    width = 8
    x = torch.arange(width, dtype=torch.float32).reshape(1, 1, 1, width)
    right = 1.0 + 0.1 * x
    left = (1.0 + 0.1 * x) / 1.1
    valid = _mask_like(left)
    features = compute_geometry_features(left, right, valid, valid)
    residual = features.signals["u3b_valid_absolute"]
    assert residual.valid_mask[..., 0].sum().item() == 0
    assert residual.score[residual.valid_mask].max().item() < 1e-5


def test_geometry_visibility_removes_many_to_one_collision_without_gt() -> None:
    left = torch.ones(1, 1, 1, 6)
    left[..., 1] = 1.0
    left[..., 2] = 2.0
    right = torch.ones_like(left)
    valid = _mask_like(left)
    features = compute_geometry_features(left, right, valid, valid)
    assert features.signals["u3b_valid_absolute"].valid_mask[..., 1].item()
    assert features.signals["u3b_valid_absolute"].valid_mask[..., 2].item()
    assert not features.visibility_mask[..., 1].item()
    assert not features.visibility_mask[..., 2].item()
    assert features.occlusion_or_unsupported_mask[..., 1].item()


def _paired_images(width: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.linspace(-0.8, 0.8, width).reshape(1, 1, 1, width)
    left = base.repeat(1, 3, 3, 1)
    right = torch.zeros_like(left)
    right[..., :-1] = left[..., 1:]
    right[..., -1] = left[..., -1]
    return left, right


def test_photometric_perfect_warp_is_zero_on_bidirectional_support() -> None:
    left_image, right_image = _paired_images()
    left, right, left_valid, right_valid = _constant_stereo(height=3, width=8)
    geometry = compute_geometry_features(left, right, left_valid, right_valid)
    features = compute_photometric_features(
        left_image,
        right_image,
        left,
        right,
        geometry,
        left_valid,
        right_valid,
    )
    valid = features.signals["u4b_bidirectional_mean"].valid_mask
    assert torch.allclose(
        features.signals["u4b_bidirectional_mean"].score[valid],
        torch.zeros_like(features.signals["u4b_bidirectional_mean"].score[valid]),
        atol=1e-6,
    )
    assert features.bidirectional_valid_mask.any()


def test_photometric_brightness_offset_is_visible_and_local_normalization_is_robust() -> None:
    left_image, right_image = _paired_images(width=16)
    right_image = (right_image + 0.2).clamp(-1.0, 1.0)
    left, right, left_valid, right_valid = _constant_stereo(height=3, width=16)
    geometry = compute_geometry_features(left, right, left_valid, right_valid)
    features = compute_photometric_features(
        left_image,
        right_image,
        left,
        right,
        geometry,
        left_valid,
        right_valid,
    )
    raw = features.signals["u4_historical"]
    normalized = features.signals["u4b_normalized_photo"]
    assert raw.score[raw.valid_mask].mean().item() > 0.0
    assert (
        normalized.score[normalized.valid_mask].mean().item()
        < raw.score[raw.valid_mask].mean().item()
    )


def test_photometric_contrast_scaling_and_local_illumination_remain_separate() -> None:
    left_image, right_image = _paired_images()
    right_image = (right_image * 0.8).clamp(-1.0, 1.0)
    right_image[..., 4:] += 0.1
    left, right, left_valid, right_valid = _constant_stereo(height=3, width=8)
    geometry = compute_geometry_features(left, right, left_valid, right_valid)
    features = compute_photometric_features(
        left_image,
        right_image,
        left,
        right,
        geometry,
        left_valid,
        right_valid,
    )
    raw = features.signals["u4_historical"]
    normalized = features.signals["u4b_normalized_photo"]
    assert raw.score[raw.valid_mask].mean().item() > 0.0
    assert normalized.score[normalized.valid_mask].mean().item() >= 0.0


def test_photometric_gradient_ssim_edge_shift_and_specular_exclusion() -> None:
    left_image, right_image = _paired_images()
    right_image[..., 2] = 1.0
    left, right, left_valid, right_valid = _constant_stereo(height=3, width=8)
    geometry = compute_geometry_features(left, right, left_valid, right_valid)
    features = compute_photometric_features(
        left_image,
        right_image,
        left,
        right,
        geometry,
        left_valid,
        right_valid,
    )
    assert (
        features.signals["u4b_gradient"].score[features.signals["u4b_gradient"].valid_mask].max()
        > 0
    )
    assert (
        features.signals["u4b_ssim_like"].score[features.signals["u4b_ssim_like"].valid_mask].max()
        > 0
    )
    assert features.signals["u4b_specularity_aware"].valid_mask[..., 3].sum().item() == 0


def test_photometric_visibility_aware_variant_reports_reduced_occlusion_support() -> None:
    left_image, right_image = _paired_images()
    left, right, left_valid, right_valid = _constant_stereo(height=3, width=8)
    left[..., 1] = 1.0
    left[..., 2] = 2.0
    geometry = compute_geometry_features(left, right, left_valid, right_valid)
    features = compute_photometric_features(
        left_image,
        right_image,
        left,
        right,
        geometry,
        left_valid,
        right_valid,
    )
    visible = features.signals["u4b_visibility_aware"].valid_mask.sum().item()
    strict = (
        features.signals["u4b_visibility_aware"].valid_mask | geometry.occlusion_or_unsupported_mask
    )
    assert visible <= strict.sum().item()


def test_build_diagnostics_do_not_mutate_baseline_tensors_or_create_sigma() -> None:
    iterations = torch.tensor([[[[10.0, 10.0]], [[8.0, 8.0]], [[7.0, 7.0]]]])
    left_image, right_image = _paired_images(width=2)
    left_image = left_image[:, :, :1]
    right_image = right_image[:, :, :1]
    baseline_disparity = iterations[:, -1:].clone()
    gaussian_parameters = {
        "rotation": torch.randn(1, 3, 1, 2),
        "scale": torch.randn(1, 3, 1, 2),
        "opacity": torch.randn(1, 1, 1, 2),
    }
    before = {key: value.clone() for key, value in gaussian_parameters.items()}
    build_p1b_signals(
        iterations,
        left_image,
        right_image,
        right_disparity_iterations=iterations.clone(),
    )
    assert torch.equal(baseline_disparity, iterations[:, -1:])
    assert all(torch.equal(gaussian_parameters[key], before[key]) for key in gaussian_parameters)


def test_shadow_diagnostics_keep_d3_as_baseline_and_measure_later_iterations() -> None:
    iterations = torch.tensor([[[[10.0]], [[8.0]], [[7.0]], [[6.5]], [[6.25]], [[6.125]]]])
    diagnostics = shadow_diagnostics(iterations)
    assert diagnostics["remaining_trajectory_length"].item() == pytest.approx(0.875)
    assert diagnostics["late_update_magnitude"].item() == pytest.approx(0.125)
    assert diagnostics["distance_baseline_to_last"].item() == pytest.approx(0.875)


def test_p1b_sequence_guard_rejects_final_sequences() -> None:
    assert validate_p1b_sequences(P1B_DEVELOPMENT_SEQUENCES) == P1B_DEVELOPMENT_SEQUENCES
    for forbidden in P1B_FINAL_SEQUENCES:
        with pytest.raises(ValueError, match="final"):
            validate_p1b_sequences(P1B_DEVELOPMENT_SEQUENCES[:-1] + (forbidden,))
