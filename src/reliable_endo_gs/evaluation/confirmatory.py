"""Pre-registered confirmatory statistics for the repair module (see docs/success_bar_confirmatory_2026-10-31.txt).

Pure functions on per-frame arrays: a seeded paired bootstrap and the success-bar verdict. The thresholds below ARE the registered bar; changing one after a
result exists is a protocol violation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

MIN_RIGHT_PSNR_GAIN_DB = 0.5
MIN_POSITIVE_FRAME_SHARE = 0.80
SSIM_LOWER_BOUND = -0.005
LPIPS_UPPER_BOUND = 0.005
MAX_LAST80_RATIO = 1.05
RESAMPLES = 5000
LEVEL = 0.95
REQUIRED_DATASETS = ("dataset_8", "dataset_9")


@dataclass(frozen=True)
class FrameRows:
    """One entry per frame of one dataset. Gains are refined minus STOP on the right view; ``last80_mse_*`` are the right-view MSE of the last 80 columns."""

    keyframe: Sequence[str]
    right_psnr_gain: np.ndarray
    right_ssim_gain: np.ndarray
    right_lpips_change: np.ndarray
    last80_mse_refined: np.ndarray
    last80_mse_stop: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.keyframe)
        arrays = (
            self.right_psnr_gain,
            self.right_ssim_gain,
            self.right_lpips_change,
            self.last80_mse_refined,
            self.last80_mse_stop,
        )
        if n == 0 or any(np.asarray(a).shape != (n,) for a in arrays):
            raise ValueError("FrameRows needs one non-empty value per frame in every array")


def bootstrap_interval(
    values: np.ndarray, *, resamples: int = RESAMPLES, seed: int = 0, level: float = LEVEL
) -> tuple[float, float, float]:
    """(mean, lower, upper) of the frame-level paired bootstrap of the mean; ``seed`` makes it reproducible."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(resamples, v.size))].mean(axis=1)
    tail = (1 - level) / 2
    return float(v.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def cluster_bootstrap_interval(
    values: np.ndarray,
    clusters: Sequence[str],
    *,
    resamples: int = RESAMPLES,
    seed: int = 0,
    level: float = LEVEL,
) -> tuple[float, float, float]:
    """Secondary: resample whole keyframes (few clusters, so the interval is weak; reported next to the frame-level one)."""
    v = np.asarray(values, dtype=np.float64)
    names = sorted(set(clusters))
    groups = [v[np.array([c == name for c in clusters])] for name in names]
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for k in range(resamples):
        picked = rng.integers(0, len(groups), size=len(groups))
        means[k] = np.concatenate([groups[i] for i in picked]).mean()
    tail = (1 - level) / 2
    return float(v.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def dataset_verdict(rows: FrameRows, *, seed: int = 0) -> dict[str, object]:
    """Every registered condition for one dataset with its value; ``passed`` is the conjunction."""
    gain = np.asarray(rows.right_psnr_gain, dtype=np.float64)
    mean, low, high = bootstrap_interval(gain, seed=seed)
    keyframes = sorted(set(rows.keyframe))
    per_keyframe = {
        k: float(gain[np.array([c == k for c in rows.keyframe])].mean()) for k in keyframes
    }
    _, ssim_low, _ = bootstrap_interval(np.asarray(rows.right_ssim_gain), seed=seed + 1)
    lpips_mean, _, lpips_high = bootstrap_interval(
        np.asarray(rows.right_lpips_change), seed=seed + 2
    )
    ssim_mean = float(np.mean(rows.right_ssim_gain))
    ratio = float(np.mean(rows.last80_mse_refined) / np.mean(rows.last80_mse_stop))
    finite = all(
        bool(np.isfinite(np.asarray(a, dtype=np.float64)).all())
        for a in (
            gain,
            rows.right_ssim_gain,
            rows.right_lpips_change,
            rows.last80_mse_refined,
            rows.last80_mse_stop,
        )
    )
    checks = {
        "P1_mean_gain_at_least_0.5dB": mean >= MIN_RIGHT_PSNR_GAIN_DB,
        "P2_bootstrap_lower_bound_positive": low > 0,
        "P3_positive_on_80pct_of_frames": float((gain > 0).mean()) >= MIN_POSITIVE_FRAME_SHARE,
        "P4_every_keyframe_mean_positive": all(v > 0 for v in per_keyframe.values()),
        "N1_ssim_not_worse": ssim_mean >= 0 and ssim_low >= SSIM_LOWER_BOUND,
        "N2_lpips_not_worse": lpips_mean <= 0 and lpips_high <= LPIPS_UPPER_BOUND,
        "N3_last80_columns_within_1.05x": ratio <= MAX_LAST80_RATIO,
        "N4_finite": finite,
    }
    return {
        "frames": int(gain.size),
        "right_psnr_gain": {
            "mean": mean,
            "ci95": [low, high],
            "positive_share": float((gain > 0).mean()),
        },
        "per_keyframe_mean_gain": per_keyframe,
        "right_ssim_gain_mean": ssim_mean,
        "right_lpips_change_mean": lpips_mean,
        "last80_mse_ratio": ratio,
        "checks": checks,
        "passed": all(checks.values()),
    }


def confirmatory_verdict(datasets: Mapping[str, FrameRows], *, seed: int = 0) -> dict[str, object]:
    """ "supported" only if every condition holds on BOTH final_test datasets; otherwise "not supported" with the failing conditions named."""
    missing = [d for d in REQUIRED_DATASETS if d not in datasets]
    if missing:
        raise ValueError(f"missing datasets: {missing}")
    per = {d: dataset_verdict(datasets[d], seed=seed) for d in REQUIRED_DATASETS}
    failing = {d: [name for name, ok in v["checks"].items() if not ok] for d, v in per.items()}  # type: ignore[attr-defined]
    return {
        "verdict": "supported" if all(v["passed"] for v in per.values()) else "not supported",
        "failing": failing,
        "datasets": per,
    }


__all__ = [
    "FrameRows",
    "bootstrap_interval",
    "cluster_bootstrap_interval",
    "confirmatory_verdict",
    "dataset_verdict",
]
