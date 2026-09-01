"""Synthetic tensor tests for development-only metric contracts."""

import math

import pytest
import torch

from reliable_endo_gs.evaluation import (
    MetricInputError,
    aggregate_by_sequence,
    aggregate_metric_results,
    bad_pixel_rate,
    depth_mae,
    depth_rmse,
    endpoint_error,
    psnr,
    ssim,
)


def test_stereo_metrics_honor_masks_and_empty_denominators() -> None:
    predicted = torch.tensor([[[[1.0, 8.0]]]])
    target = torch.tensor([[[[0.0, 0.0]]]])
    mask = torch.tensor([[[[True, False]]]])
    assert endpoint_error(predicted, target, mask).value == 1.0
    assert bad_pixel_rate(predicted, target, mask).value == 0.0
    empty = endpoint_error(predicted, target, torch.zeros_like(mask))
    assert empty.value is None and empty.valid_count == 0


def test_metrics_exclude_non_finite_values() -> None:
    predicted = torch.tensor([[[[1.0, float("nan"), float("inf")]]]])
    target = torch.tensor([[[[0.0, 2.0, 3.0]]]])
    result = endpoint_error(predicted, target)
    assert result.value == 1.0
    assert result.valid_count == 1
    assert result.total_count == 3


def test_evaluation_does_not_mutate_contract_tensors() -> None:
    predicted = torch.tensor([[[[1.0, 2.0]]]])
    target = torch.tensor([[[[0.0, 1.0]]]])
    mask = torch.tensor([[[[True, False]]]])
    predicted_before = predicted.clone()
    target_before = target.clone()
    mask_before = mask.clone()
    endpoint_error(predicted, target, mask)
    assert torch.equal(predicted, predicted_before)
    assert torch.equal(target, target_before)
    assert torch.equal(mask, mask_before)


def test_depth_and_rendering_metrics_are_contract_level() -> None:
    predicted = torch.tensor([[[[1.0, 3.0]]]])
    target = torch.tensor([[[[1.0, 1.0]]]])
    assert depth_mae(predicted, target).value == 1.0
    assert depth_rmse(predicted, target).value == pytest.approx(2**0.5)
    image = torch.zeros(1, 3, 2, 2)
    assert math.isinf(psnr(image, image).value or 0.0)
    assert ssim(image, image).value == pytest.approx(1.0)


def test_metrics_reject_shape_guessing() -> None:
    with pytest.raises(MetricInputError):
        endpoint_error(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2))


def test_aggregation_is_valid_count_weighted() -> None:
    values = [
        endpoint_error(torch.tensor([[[[1.0]]]]), torch.zeros(1, 1, 1, 1)),
        endpoint_error(torch.tensor([[[[3.0, 3.0, 3.0]]]]), torch.zeros(1, 1, 1, 3)),
    ]
    summary = aggregate_metric_results(values)
    assert summary.value == pytest.approx(2.5)
    assert summary.valid_count == 4


def test_sequence_aggregation_preserves_group_boundaries() -> None:
    first = endpoint_error(torch.tensor([[[[1.0]]]]), torch.zeros(1, 1, 1, 1))
    second = endpoint_error(torch.tensor([[[[3.0, 3.0, 3.0]]]]), torch.zeros(1, 1, 1, 3))
    grouped = aggregate_by_sequence(["sequence_a", "sequence_b"], [{"epe": first}, {"epe": second}])
    assert grouped["sequence_a"]["epe"].value == 1.0
    assert grouped["sequence_a"]["epe"].valid_count == 1
    assert grouped["sequence_b"]["epe"].value == 3.0
    assert grouped["sequence_b"]["epe"].valid_count == 3
