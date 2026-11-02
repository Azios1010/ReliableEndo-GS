"""Paired statistics and decision rules of the held-out comparison (rules fixed in E1_E3_E4_PROTOCOL.txt before the data).

Frame bootstrap resamples frames; keyframe-cluster bootstrap resamples whole keyframes (4 per dataset, so its intervals are weak
and reported as a sensitivity). Higher-is-better metrics: PSNR, SSIM, 1px accuracy; lower-is-better: LPIPS, EPE, depth errors.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

HIGHER_IS_BETTER = ("psnr", "ssim", "accuracy")


def higher_is_better(metric: str) -> bool:
    return any(token in metric for token in HIGHER_IS_BETTER)


def frame_bootstrap(
    delta: np.ndarray, resamples: int = 5000, seed: int = 20261101
) -> tuple[float, float, float]:
    """(mean, lo, hi) of the 95 percent percentile interval of the mean over frames."""
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.empty(resamples)
    for start in range(0, resamples, 500):
        stop = min(start + 500, resamples)
        means[start:stop] = delta[rng.integers(0, n, (stop - start, n))].mean(axis=1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cluster_bootstrap(
    delta: np.ndarray, clusters: Sequence[str], resamples: int = 5000, seed: int = 20261101
) -> tuple[float, float, float]:
    """Resample whole clusters (keyframes) with replacement; mean over all frames of the drawn clusters."""
    groups: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(delta, clusters, strict=True):
        groups[cluster].append(float(value))
    arrays = [np.array(v) for v in groups.values()]
    sums, counts = np.array([a.sum() for a in arrays]), np.array([len(a) for a in arrays])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(arrays), (resamples, len(arrays)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_summary(
    a: Mapping[str, float],
    b: Mapping[str, float],
    metric: str,
    resamples: int = 5000,
    seed: int = 20261101,
) -> dict[str, Any] | None:
    """A minus B over the shared frames, with both intervals, the win rate of A, and whether A is better with the frame interval excluding zero."""
    ids = sorted(set(a) & set(b))
    if len(ids) < 2:
        return None
    delta = np.array([a[i] - b[i] for i in ids])
    clusters = ["/".join(i.split("/")[:2]) for i in ids]
    mean, lo, hi = frame_bootstrap(delta, resamples, seed)
    _, clo, chi = cluster_bootstrap(delta, clusters, resamples, seed)
    better_sign = 1.0 if higher_is_better(metric) else -1.0
    wins = float(np.mean(delta * better_sign > 0))
    excludes = (lo > 0) if better_sign > 0 else (hi < 0)
    return {
        "frames": len(ids),
        "mean_a_minus_b": mean,
        "frame_ci95": [lo, hi],
        "cluster_ci95": [clo, chi],
        "win_rate_a": wins,
        "a_better_frame_ci": bool(excludes),
        "a_better_cluster_ci": bool((clo > 0) if better_sign > 0 else (chi < 0)),
    }


def kd_vs_gt_verdict(
    epe_gt: Mapping[str, float],
    epe_kd: Mapping[str, float],
    depth_gt: Mapping[str, float],
    depth_kd: Mapping[str, float],
) -> dict[str, Any]:
    """Rule (b) of the protocol on the held-out datasets 8 and 9 (r = mean EPE_GT / mean EPE_KD per dataset; keyframe counts)."""
    result: dict[str, Any] = {"per_dataset": {}}
    keyframe_kd_lower = keyframe_gt_lower = keyframes = 0
    r_values, depth_agrees = [], []
    for dataset in ("dataset_8", "dataset_9"):
        ids = sorted(i for i in set(epe_gt) & set(epe_kd) if i.startswith(dataset + "/"))
        if not ids:
            return {"verdict": "not evaluable", "per_dataset": result["per_dataset"]}
        r = float(np.mean([epe_gt[i] for i in ids]) / np.mean([epe_kd[i] for i in ids]))
        dids = sorted(i for i in set(depth_gt) & set(depth_kd) if i.startswith(dataset + "/"))
        r_depth = float(np.mean([depth_gt[i] for i in dids]) / np.mean([depth_kd[i] for i in dids]))
        by_kf: dict[str, list[str]] = defaultdict(list)
        for i in ids:
            by_kf["/".join(i.split("/")[:2])].append(i)
        for frames in by_kf.values():
            keyframes += 1
            gt_mean, kd_mean = (
                np.mean([epe_gt[i] for i in frames]),
                np.mean([epe_kd[i] for i in frames]),
            )
            keyframe_kd_lower += int(kd_mean < gt_mean)
            keyframe_gt_lower += int(gt_mean < kd_mean)
        result["per_dataset"][dataset] = {"r_epe": r, "r_depth": r_depth, "frames": len(ids)}
        r_values.append(r)
        depth_agrees.append((r > 1) == (r_depth > 1))
    result.update(
        keyframes=keyframes,
        keyframes_kd_lower=keyframe_kd_lower,
        keyframes_gt_lower=keyframe_gt_lower,
    )
    kd_better = all(r >= 1.25 for r in r_values) and keyframe_kd_lower >= 6 and all(depth_agrees)
    gt_better = all(r <= 0.80 for r in r_values) and keyframe_gt_lower >= 6 and all(depth_agrees)
    no_difference = all(0.85 <= r <= 1.18 for r in r_values) and not kd_better and not gt_better
    result["verdict"] = (
        "KD better"
        if kd_better
        else "GT better"
        if gt_better
        else "no meaningful difference"
        if no_difference
        else "mixed"
    )
    return result
