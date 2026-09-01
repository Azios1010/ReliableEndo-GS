"""Regression tests for masked uncertainty ranking and calibration metrics."""

import pytest
import torch

from reliable_endo_gs.evaluation.uncertainty import (
    evaluate_uncertainty,
    evaluate_uncertainty_by_bins,
)


def test_metrics_separate_raw_ranking_from_sigma_calibration() -> None:
    error = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
    score = error.clone()
    valid = torch.ones_like(error, dtype=torch.bool)
    sigma = torch.ones_like(error)

    metrics = evaluate_uncertainty(error, score, valid, high_error_threshold=2.0, sigma_d=sigma)

    assert metrics.valid_count == 4
    assert metrics.pearson == pytest.approx(1.0)
    assert metrics.spearman == pytest.approx(1.0)
    assert metrics.auroc == pytest.approx(1.0)
    assert metrics.ause == pytest.approx(0.0)
    assert metrics.laplace_nll == pytest.approx(
        (0.0 + 1.0 + 2.0 + 3.0) / 4 * 2**0.5 - 0.5 * torch.log(torch.tensor(2.0)).item()
    )
    assert metrics.coverage_one_sigma == pytest.approx(0.5)


def test_metrics_empty_and_single_class_cases_are_explicit() -> None:
    values = torch.ones(1, 1, 1, 2)
    empty = evaluate_uncertainty(
        values, values, torch.zeros_like(values, dtype=torch.bool), high_error_threshold=1.0
    )
    one_class = evaluate_uncertainty(
        values, values, torch.ones_like(values, dtype=torch.bool), high_error_threshold=2.0
    )

    assert empty.valid_count == 0
    assert empty.auroc is None
    assert one_class.auroc is None


def test_fixed_bins_retain_empty_bins_without_data_dependent_edge_fitting() -> None:
    error = torch.tensor([[[[0.0, 1.0, 2.0]]]])
    score = error.clone()
    valid = torch.ones_like(error, dtype=torch.bool)
    depth = torch.tensor([[[[1.0, 3.0, 9.0]]]])

    binned = evaluate_uncertainty_by_bins(
        error, score, valid, depth, (2.0, 5.0), high_error_threshold=1.0
    )

    assert [item.valid_count for item in binned.metrics] == [1, 1, 1]
    assert binned.bin_edges == (2.0, 5.0)


def test_metrics_use_strict_high_error_threshold_boundary() -> None:
    error = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    score = torch.tensor([[[[0.0, 3.0, -1.0]]]])
    valid = torch.ones_like(error, dtype=torch.bool)

    metrics = evaluate_uncertainty(error, score, valid, high_error_threshold=2.0)

    assert metrics.auroc == pytest.approx(0.0)


def test_metrics_handle_singleton_constants_ties_and_nonfinite_values() -> None:
    singleton = torch.tensor([[[[2.0]]]])
    valid = torch.ones_like(singleton, dtype=torch.bool)
    one = evaluate_uncertainty(singleton, singleton, valid, high_error_threshold=1.0)
    assert one.valid_count == 1
    assert one.ause is None
    assert one.pearson is None
    assert one.spearman is None

    error = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
    valid = torch.ones_like(error, dtype=torch.bool)
    perfect = evaluate_uncertainty(error, error, valid, high_error_threshold=2.0)
    reversed_metrics = evaluate_uncertainty(error, error.flip(-1), valid, high_error_threshold=2.0)
    tied = evaluate_uncertainty(error, torch.ones_like(error), valid, high_error_threshold=2.0)
    assert perfect.spearman == pytest.approx(1.0)
    assert perfect.ause == pytest.approx(0.0)
    assert perfect.auroc == pytest.approx(1.0)
    assert reversed_metrics.spearman == pytest.approx(-1.0)
    assert reversed_metrics.ause == pytest.approx(1.0)
    assert reversed_metrics.auroc == pytest.approx(0.0)
    assert tied.spearman is None
    assert tied.auroc == pytest.approx(0.5)

    nonfinite_error = torch.tensor([[[[0.0, float("nan"), float("inf")]]]])
    nonfinite_score = torch.tensor([[[[0.0, 1.0, float("inf")]]]])
    finite = evaluate_uncertainty(
        nonfinite_error, nonfinite_score, valid[..., :3], high_error_threshold=1.0
    )
    assert finite.valid_count == 1


def test_metrics_reject_negative_absolute_error_and_sigma_dtype_mismatch() -> None:
    values = torch.ones(1, 1, 1, 2)
    valid = torch.ones_like(values, dtype=torch.bool)
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_uncertainty(
            torch.tensor([[[[-1.0, 0.0]]]]), values, valid, high_error_threshold=1.0
        )
    with pytest.raises(TypeError, match="same dtype"):
        evaluate_uncertainty(
            values, values, valid, high_error_threshold=1.0, sigma_d=values.double()
        )


def test_metrics_validate_sigma_shape_even_when_mask_is_empty() -> None:
    values = torch.ones(1, 1, 1, 2)
    empty = torch.zeros_like(values, dtype=torch.bool)
    with pytest.raises(ValueError, match="sigma_d"):
        evaluate_uncertainty(
            values, values, empty, high_error_threshold=1.0, sigma_d=torch.ones(1, 1, 1, 1)
        )
