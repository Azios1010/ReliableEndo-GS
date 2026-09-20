"""P1 proxy metrics and deterministic aggregation utilities.

This module keeps inference proxies separate from ground-truth evaluation
targets.  It does not fit thresholds, calibrate scores, or access the final
untouched evaluation split.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class P1MetricSummary:
    """Pixel-level score metrics with explicit validity accounting."""

    proxy: str
    valid_count: int
    total_count: int
    spearman: float | None
    auroc: float | None
    auprc: float | None
    risk_coverage: tuple[dict[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "proxy": self.proxy,
            "valid_count": self.valid_count,
            "total_count": self.total_count,
            "spearman": self.spearman,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "risk_coverage": list(self.risk_coverage),
        }


def oracle_disparity_error(
    disparity: torch.Tensor,
    ground_truth: torch.Tensor,
    ground_truth_valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``|d_pred-d_gt|`` and an evaluation-only valid target mask."""

    if disparity.shape != ground_truth.shape or disparity.shape != ground_truth_valid_mask.shape:
        raise ValueError("disparity, ground truth, and mask must have identical shapes")
    if disparity.ndim != 4 or disparity.shape[1] != 1:
        raise ValueError("disparity tensors must have shape [B, 1, H, W]")
    if ground_truth_valid_mask.dtype is not torch.bool:
        raise TypeError("ground_truth_valid_mask must use torch.bool")
    valid = ground_truth_valid_mask & torch.isfinite(disparity) & torch.isfinite(ground_truth)
    error = (disparity - ground_truth).abs()
    return torch.where(valid, error, torch.zeros_like(error)), valid


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("ranking input must be one-dimensional")
    sorted_values, order = torch.sort(values, stable=True)
    size = values.numel()
    if size == 0:
        return torch.empty_like(values, dtype=torch.float64)
    group_start = torch.ones(size, dtype=torch.bool, device=values.device)
    group_start[1:] = sorted_values[1:] != sorted_values[:-1]
    group_ids = group_start.cumsum(dim=0) - 1
    starts = torch.nonzero(group_start, as_tuple=False).flatten()
    ends = torch.cat(
        (
            starts[1:],
            torch.tensor([size], dtype=starts.dtype, device=values.device),
        )
    )
    group_ranks = (starts + ends - 1).to(torch.float64).div(2.0)
    ranks_sorted = group_ranks[group_ids]
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def _correlation(first: torch.Tensor, second: torch.Tensor) -> float | None:
    if first.numel() < 2:
        return None
    first = first.double() - first.double().mean()
    second = second.double() - second.double().mean()
    denominator = torch.sqrt(first.square().sum() * second.square().sum())
    if float(denominator.item()) == 0.0:
        return None
    return float((first * second).sum().div(denominator).item())


def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    positive = int(labels.sum().item())
    negative = int(labels.numel()) - positive
    if positive == 0 or negative == 0:
        return None
    ranks = _average_ranks(scores)
    positive_rank_sum = ranks[labels].sum()
    return float(
        ((positive_rank_sum - positive * (positive - 1) / 2.0) / (positive * negative)).item()
    )


def _auprc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    positive = int(labels.sum().item())
    if positive == 0:
        return None
    order = torch.argsort(scores, descending=True, stable=True)
    ordered_labels = labels[order].to(torch.float64)
    true_positive = torch.cumsum(ordered_labels, dim=0)
    positions = torch.arange(1, labels.numel() + 1, device=labels.device, dtype=torch.float64)
    precision = true_positive / positions
    recall = true_positive / positive
    previous_recall = torch.cat((torch.zeros(1, device=labels.device, dtype=torch.float64), recall[:-1]))
    return float(((recall - previous_recall) * precision).sum().item())


def risk_coverage_curve(
    absolute_error: torch.Tensor,
    score: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    coverage_levels: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
) -> tuple[dict[str, float], ...]:
    """Return retained risk after keeping the least-uncertain pixels first."""

    if absolute_error.shape != score.shape or score.shape != valid_mask.shape:
        raise ValueError("absolute_error, score, and valid_mask must share shape")
    finite = valid_mask & torch.isfinite(absolute_error) & torch.isfinite(score)
    error = absolute_error[finite].reshape(-1)
    values = score[finite].reshape(-1)
    if error.numel() == 0:
        return tuple()
    order = torch.argsort(values, descending=False, stable=True)
    results: list[dict[str, float]] = []
    for level in coverage_levels:
        if not math.isfinite(float(level)) or not 0.0 < float(level) <= 1.0:
            raise ValueError("coverage levels must be finite and in (0, 1]")
        count = max(1, min(error.numel(), math.ceil(error.numel() * float(level))))
        retained = error[order[:count]]
        results.append(
            {
                "coverage": count / error.numel(),
                "risk": float(retained.mean().item()),
                "retained_count": float(count),
            }
        )
    return tuple(results)


def evaluate_p1_proxy(
    proxy: str,
    absolute_error: torch.Tensor,
    score: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    high_error_threshold: float | None = None,
    coverage_levels: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
) -> P1MetricSummary:
    """Evaluate one proxy without fitting any threshold on the evaluated data."""

    if absolute_error.shape != score.shape or score.shape != valid_mask.shape:
        raise ValueError("absolute_error, score, and valid_mask must share shape")
    if absolute_error.ndim != 4 or absolute_error.shape[1] != 1:
        raise ValueError("metric maps must have shape [B, 1, H, W]")
    if absolute_error.dtype != score.dtype:
        raise TypeError("absolute_error and score must share dtype")
    if valid_mask.dtype is not torch.bool:
        raise TypeError("valid_mask must use torch.bool")
    if high_error_threshold is not None and (
        not math.isfinite(float(high_error_threshold)) or high_error_threshold < 0.0
    ):
        raise ValueError("high_error_threshold must be finite and non-negative")
    finite = valid_mask & torch.isfinite(absolute_error) & torch.isfinite(score)
    error = absolute_error[finite].reshape(-1)
    values = score[finite].reshape(-1)
    labels = (
        error > high_error_threshold
        if high_error_threshold is not None
        else torch.zeros_like(error, dtype=torch.bool)
    )
    return P1MetricSummary(
        proxy=proxy,
        valid_count=int(error.numel()),
        total_count=int(valid_mask.numel()),
        spearman=_correlation(_average_ranks(values), _average_ranks(error)) if error.numel() else None,
        auroc=_auroc(values, labels) if high_error_threshold is not None else None,
        auprc=_auprc(values, labels) if high_error_threshold is not None else None,
        risk_coverage=risk_coverage_curve(
            absolute_error,
            score,
            valid_mask,
            coverage_levels=coverage_levels,
        ),
    )


def aggregate_fixed_grid(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    grid: tuple[int, int] = (4, 4),
) -> tuple[dict[str, Any], ...]:
    """Aggregate a map over a deterministic fixed grid without learning regions."""

    if values.shape != valid_mask.shape or values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("values and valid_mask must have shape [B, 1, H, W]")
    rows, columns = grid
    if rows < 1 or columns < 1:
        raise ValueError("grid dimensions must be positive")
    height, width = values.shape[-2:]
    output: list[dict[str, Any]] = []
    for row in range(rows):
        y0, y1 = height * row // rows, height * (row + 1) // rows
        for column in range(columns):
            x0, x1 = width * column // columns, width * (column + 1) // columns
            region_values = values[..., y0:y1, x0:x1]
            region_mask = valid_mask[..., y0:y1, x0:x1] & torch.isfinite(region_values)
            selected = region_values[region_mask]
            record: dict[str, Any] = {
                "region_id": f"r{row}c{column}",
                "row": row,
                "column": column,
                "total_count": int(region_mask.numel()),
                "valid_count": int(selected.numel()),
                "valid_fraction": float(region_mask.float().mean().item()),
            }
            if selected.numel():
                record.update(
                    {
                        "mean": float(selected.mean().item()),
                        "median": float(selected.median().item()),
                        "p90": float(torch.quantile(selected, 0.9).item()),
                        "max": float(selected.max().item()),
                    }
                )
            else:
                record.update({"mean": None, "median": None, "p90": None, "max": None})
            output.append(record)
    return tuple(output)


def sequence_macro(values_by_sequence: Mapping[str, Mapping[str, float | None]]) -> dict[str, float | None]:
    """Compute an unweighted macro mean over sequence summaries."""

    if not values_by_sequence:
        raise ValueError("at least one sequence summary is required")
    keys = tuple(next(iter(values_by_sequence.values())).keys())
    output: dict[str, float | None] = {}
    for key in keys:
        values = [summary[key] for summary in values_by_sequence.values() if summary[key] is not None]
        output[key] = sum(values) / len(values) if values else None
    return output


def cluster_bootstrap_mean(
    values_by_cluster: Mapping[str, float],
    *,
    repeats: int = 1000,
    seed: int = 1314,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Provide deterministic cluster-level bootstrap support for later runs."""

    if not values_by_cluster or repeats < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("clusters, repeats, and confidence must be valid")
    values = tuple(float(value) for value in values_by_cluster.values())
    rng = random.Random(seed)
    samples = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(repeats)]
    samples.sort()
    lower_index = max(0, min(repeats - 1, math.floor((1.0 - confidence) * repeats / 2.0)))
    upper_index = max(0, min(repeats - 1, math.ceil((1.0 + confidence) * repeats / 2.0) - 1))
    return {
        "estimate": sum(values) / len(values),
        "lower": samples[lower_index],
        "upper": samples[upper_index],
        "clusters": float(len(values)),
        "repeats": float(repeats),
        "seed": float(seed),
    }
