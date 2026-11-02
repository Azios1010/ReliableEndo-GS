"""Pairwise distance error on synthetic point clouds (CPU)."""

import torch

from reliable_endo_gs.evaluation.pair_distance import pair_distance_metrics, seed_from_sample_id


def _plane(height: int = 80, width: int = 100) -> torch.Tensor:
    """A tilted plane in metres with 0.5 mm pixel spacing: [1, 3, H, W]."""
    yy, xx = torch.meshgrid(
        torch.arange(height).float(), torch.arange(width).float(), indexing="ij"
    )
    return torch.stack((xx * 0.0005, yy * 0.0005, 0.05 + xx * 0.0001), dim=0).unsqueeze(0)


def test_identical_prediction_has_zero_error_and_full_coverage() -> None:
    points = _plane()
    valid = torch.ones(1, 1, 80, 100, dtype=torch.bool)
    result = pair_distance_metrics(points, points.clone(), valid, valid, seed=1, max_offset_px=60)
    assert result["pair_dist_mae_mm_5_10"].value == 0.0
    assert result["pair_dist_coverage_5_10"].value == 1.0
    assert result["pair_dist_coverage_10_20"].total_count > 100


def test_a_scaled_prediction_gives_the_expected_relative_error() -> None:
    points = _plane()
    valid = torch.ones(1, 1, 80, 100, dtype=torch.bool)
    result = pair_distance_metrics(points * 1.1, points, valid, valid, seed=1, max_offset_px=60)
    assert abs(result["pair_dist_rel_5_10"].value - 0.1) < 1e-4
    assert result["pair_dist_mae_mm_10_20"].value > result["pair_dist_mae_mm_5_10"].value


def test_pairs_are_the_same_for_two_predictions_and_invalid_points_lower_the_coverage() -> None:
    points = _plane()
    valid = torch.ones(1, 1, 80, 100, dtype=torch.bool)
    half = valid.clone()
    half[..., :, 50:] = False
    a = pair_distance_metrics(points, points, valid, valid, seed=7, max_offset_px=60)
    b = pair_distance_metrics(points, points, valid, half, seed=7, max_offset_px=60)
    assert a["pair_dist_coverage_5_10"].total_count == b["pair_dist_coverage_5_10"].total_count
    assert b["pair_dist_coverage_5_10"].value < 1.0


def test_seed_is_stable_and_a_frame_without_valid_pixels_is_unavailable() -> None:
    assert seed_from_sample_id("dataset_8/keyframe_0/frame_data000001") == seed_from_sample_id(
        "dataset_8/keyframe_0/frame_data000001"
    )
    points = _plane()
    none = torch.zeros(1, 1, 80, 100, dtype=torch.bool)
    result = pair_distance_metrics(points, points, none, none, seed=1)
    assert (
        result["pair_dist_mae_mm_5_10"].value is None
        and result["pair_dist_mae_mm_5_10"].total_count == 0
    )
