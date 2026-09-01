"""Read-only uncertainty metrics with explicit masks and frozen thresholds."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from reliable_endo_gs.contracts.common import (
    require_bool,
    require_floating,
    require_same_device,
    require_same_dtype,
)
from reliable_endo_gs.uncertainty.laplace import laplace_nll


@dataclass(frozen=True, slots=True)
class UncertaintyMetrics:
    """Metrics from one masked error/score evaluation, including valid count."""

    valid_count: int
    pearson: float | None
    spearman: float | None
    ause: float | None
    auroc: float | None
    laplace_nll: float | None
    coverage_one_sigma: float | None
    coverage_error: float | None


@dataclass(frozen=True, slots=True)
class BinnedUncertaintyMetrics:
    """Metrics stratified by fixed scalar-bin edges, retaining empty bins."""

    bin_edges: tuple[float, ...]
    metrics: tuple[UncertaintyMetrics, ...]


def _rank(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic average ranks; ties receive their shared mid-rank.

    ``stable=True`` uses flattened input order as the deterministic tie-break.
    The ranking is one-based up to an irrelevant constant shift, so Spearman
    and AUROC are invariant to that shift.
    """

    sorted_values, order = values.sort(stable=True)
    ranks_sorted = torch.empty_like(sorted_values, dtype=torch.float64)
    start = 0
    for index in range(1, sorted_values.numel() + 1):
        if index == sorted_values.numel() or sorted_values[index] != sorted_values[start]:
            ranks_sorted[start:index] = (start + index - 1) / 2.0
            start = index
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def _correlation(first: torch.Tensor, second: torch.Tensor) -> float | None:
    if first.numel() < 2:
        return None
    a = first.double() - first.double().mean()
    b = second.double() - second.double().mean()
    denominator = torch.sqrt((a.square().sum()) * (b.square().sum()))
    if denominator == 0:
        return None
    return float((a * b).sum().div(denominator).item())


def _ause(error: torch.Tensor, score: torch.Tensor) -> float | None:
    """Normalized area between score and oracle sparsification curves.

    At each fraction removed, the curve is the mean error among remaining
    samples. The area is integrated over fractions in ``[0, 1]`` after
    division by the global mean error, making zero-error cases exactly zero.
    Descending score/error ties use stable flattened-input order, which makes
    the otherwise underdetermined sparsification path reproducible.
    """

    count = error.numel()
    if bool((error < 0).any()):
        raise ValueError("absolute_error must be non-negative")
    if count < 2:
        return None
    remove_counts = torch.arange(0, count, device=error.device)
    score_order = score.argsort(descending=True, stable=True)
    oracle_order = error.argsort(descending=True, stable=True)
    score_remaining = torch.cumsum(error[score_order].flip(0), dim=0).flip(0)
    oracle_remaining = torch.cumsum(error[oracle_order].flip(0), dim=0).flip(0)
    baseline = error.mean()
    if baseline == 0:
        return 0.0
    score_curve = (
        torch.cat(
            (score_remaining / torch.arange(count, 0, -1, device=error.device), error.new_zeros(1))
        )
        / baseline
    )
    oracle_curve = (
        torch.cat(
            (oracle_remaining / torch.arange(count, 0, -1, device=error.device), error.new_zeros(1))
        )
        / baseline
    )
    fractions = torch.cat((remove_counts, error.new_tensor([count]))).to(error.dtype) / count
    return float(torch.trapezoid(score_curve - oracle_curve, fractions).item())


def _auroc(score: torch.Tensor, labels: torch.Tensor) -> float | None:
    positive = int(labels.sum().item())
    negative = labels.numel() - positive
    if positive == 0 or negative == 0:
        return None
    ranks = _rank(score)
    positive_rank_sum = ranks[labels].sum()
    return float(
        ((positive_rank_sum - positive * (positive - 1) / 2.0) / (positive * negative)).item()
    )


def evaluate_uncertainty(
    absolute_error: torch.Tensor,
    score: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    high_error_threshold: float,
    sigma_d: torch.Tensor | None = None,
) -> UncertaintyMetrics:
    """Evaluate raw ranking and optional probabilistic scale separately.

    ``high_error_threshold`` must be frozen from validation data before test
    evaluation.  ``sigma_d`` is optional; NLL and coverage are omitted for a
    raw score because arbitrary rank scores do not define a likelihood.
    """

    if absolute_error.shape != score.shape or valid_mask.shape != score.shape:
        raise ValueError("absolute_error, score, and valid_mask must share shape")
    if absolute_error.ndim != 4 or absolute_error.shape[1] != 1:
        raise ValueError("error and score maps must have shape [B, 1, H, W]")
    require_floating("absolute_error", absolute_error)
    require_floating("score", score)
    require_same_dtype("score", score, "absolute_error", absolute_error)
    require_same_device("score", score, "absolute_error", absolute_error)
    require_bool("valid_mask", valid_mask)
    require_same_device("valid_mask", valid_mask, "score", score)
    if bool((absolute_error < 0).any()):
        raise ValueError("absolute_error must be non-negative")
    if not math.isfinite(high_error_threshold) or high_error_threshold < 0:
        raise ValueError("high_error_threshold must be finite and non-negative")
    if sigma_d is not None:
        if sigma_d.shape != score.shape:
            raise ValueError("sigma_d must match score shape")
        require_floating("sigma_d", sigma_d)
        require_same_dtype("sigma_d", sigma_d, "score", score)
        require_same_device("sigma_d", sigma_d, "score", score)
    finite = torch.isfinite(absolute_error) & torch.isfinite(score)
    mask = valid_mask & finite
    error = absolute_error[mask]
    raw_score = score[mask]
    if error.numel() == 0:
        return UncertaintyMetrics(0, None, None, None, None, None, None, None)
    pearson = _correlation(raw_score, error)
    spearman = _correlation(_rank(raw_score), _rank(error))
    nll: float | None = None
    coverage: float | None = None
    coverage_error: float | None = None
    if sigma_d is not None:
        valid_sigma = sigma_d[mask]
        if not bool(torch.isfinite(valid_sigma).all()) or not bool((valid_sigma > 0).all()):
            raise ValueError("sigma_d must be finite and strictly positive for metric-valid pixels")
        nll = float(
            laplace_nll(
                torch.zeros_like(absolute_error),
                absolute_error,
                sigma_d / math.sqrt(2.0),
                mask,
            ).item()
        )
        coverage = float((error <= valid_sigma).float().mean().item())
        # For Laplace b = sigma/sqrt(2), P(|X| <= sigma) = 1-exp(-sqrt(2)).
        coverage_error = abs(coverage - (1.0 - math.exp(-math.sqrt(2.0))))
    return UncertaintyMetrics(
        valid_count=int(error.numel()),
        pearson=pearson,
        spearman=spearman,
        ause=_ause(error, raw_score),
        auroc=_auroc(raw_score, error > high_error_threshold),
        laplace_nll=nll,
        coverage_one_sigma=coverage,
        coverage_error=coverage_error,
    )


def evaluate_uncertainty_by_bins(
    absolute_error: torch.Tensor,
    score: torch.Tensor,
    valid_mask: torch.Tensor,
    bin_values: torch.Tensor,
    bin_edges: tuple[float, ...],
    *,
    high_error_threshold: float,
    sigma_d: torch.Tensor | None = None,
) -> BinnedUncertaintyMetrics:
    """Evaluate fixed depth or metadata bins without re-fitting any threshold.

    ``bin_edges`` are increasing upper boundaries. They must originate from
    the validation protocol; this routine only applies them and never learns
    bin definitions from the evaluated data.
    """

    if bin_values.shape != score.shape:
        raise ValueError("bin_values must match score shape")
    require_floating("bin_values", bin_values)
    require_same_dtype("bin_values", bin_values, "score", score)
    require_same_device("bin_values", bin_values, "score", score)
    if not all(math.isfinite(edge) for edge in bin_edges):
        raise ValueError("bin_edges must be finite")
    if any(next_edge <= edge for edge, next_edge in zip(bin_edges, bin_edges[1:], strict=False)):
        raise ValueError("bin_edges must be strictly increasing")
    finite_bin_values = torch.isfinite(bin_values)
    boundaries = (-math.inf, *bin_edges, math.inf)
    results: list[UncertaintyMetrics] = []
    for lower, upper in zip(boundaries, boundaries[1:], strict=False):
        membership = finite_bin_values & (bin_values >= lower) & (bin_values < upper)
        results.append(
            evaluate_uncertainty(
                absolute_error,
                score,
                valid_mask & membership,
                high_error_threshold=high_error_threshold,
                sigma_d=sigma_d,
            )
        )
    return BinnedUncertaintyMetrics(bin_edges=bin_edges, metrics=tuple(results))
