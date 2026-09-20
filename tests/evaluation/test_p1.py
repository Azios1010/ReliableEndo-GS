"""Synthetic P1 metric and region aggregation tests."""

import torch

from reliable_endo_gs.evaluation.p1 import (
    _average_ranks,
    aggregate_fixed_grid,
    cluster_bootstrap_mean,
    evaluate_p1_proxy,
    oracle_disparity_error,
)


def test_average_ranks_handles_ties_without_changing_rank_semantics() -> None:
    values = torch.tensor([2.0, 1.0, 2.0, 1.0])

    assert torch.equal(_average_ranks(values), torch.tensor([2.5, 0.5, 2.5, 0.5]))


def test_oracle_error_is_gt_only_target() -> None:
    prediction = torch.tensor([[[[1.0, 3.0]]]])
    target = torch.tensor([[[[2.0, 1.0]]]])
    valid = torch.tensor([[[[True, False]]]])

    error, target_valid = oracle_disparity_error(prediction, target, valid)

    assert torch.equal(error, torch.tensor([[[[1.0, 0.0]]]]))
    assert torch.equal(target_valid, valid)


def test_p1_metrics_rank_and_risk_coverage_are_deterministic() -> None:
    error = torch.tensor([[[[0.1, 0.2, 0.8, 1.0]]]])
    score = torch.tensor([[[[0.0, 0.1, 0.8, 1.0]]]])
    valid = torch.ones_like(error, dtype=torch.bool)

    result = evaluate_p1_proxy(
        "synthetic",
        error,
        score,
        valid,
        high_error_threshold=0.5,
    )

    assert result.spearman == 1.0
    assert result.auroc == 1.0
    assert result.auprc == 1.0
    assert result.risk_coverage[-1]["coverage"] == 1.0


def test_fixed_grid_and_cluster_bootstrap_preserve_grouping() -> None:
    values = torch.arange(16.0).reshape(1, 1, 4, 4)
    valid = torch.ones_like(values, dtype=torch.bool)

    regions = aggregate_fixed_grid(values, valid, grid=(2, 2))
    bootstrap = cluster_bootstrap_mean({"a": 1.0, "b": 3.0}, repeats=20, seed=1314)

    assert len(regions) == 4
    assert regions[0]["mean"] == 2.5
    assert bootstrap["estimate"] == 2.0
    assert bootstrap["clusters"] == 2.0
