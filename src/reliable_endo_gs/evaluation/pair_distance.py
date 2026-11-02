"""Pairwise 3D distance error: a task-proximate geometry metric for distances between points of the tissue surface.

Systems that warn of instrument-to-vessel proximity compute metric distances between points of the reconstructed surface, so the
quantity that matters is the error of such a distance, not only of a depth per pixel. For one frame, pairs of pixels with valid
ground truth are drawn with a seeded generator (the same pairs for every model: they depend only on the ground truth), the pairs
whose ground-truth 3D distance falls into a band (mm) are kept, and the distance between the two predicted points is compared with
the ground-truth distance. A pair with an invalid predicted point is not scored; the fraction of scored pairs is reported as coverage.

Shapes: points ``[1, 3, H, W]`` in metres (invalid entries may be NaN), masks ``[1, 1, H, W]`` bool. Sampling runs on the CPU so
that the pairs do not depend on the device.
"""

from __future__ import annotations

import hashlib

import torch

from reliable_endo_gs.evaluation.stereo import MetricInputError, MetricResult

DEFAULT_BANDS_MM: tuple[tuple[float, float], ...] = ((5.0, 10.0), (10.0, 20.0))


def seed_from_sample_id(sample_id: str) -> int:
    """Stable 32-bit seed derived from the frame id."""
    return int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:4], "big")


def pair_distance_metrics(
    predicted_points: torch.Tensor,
    gt_points: torch.Tensor,
    gt_valid: torch.Tensor,
    predicted_valid: torch.Tensor,
    *,
    seed: int,
    candidates: int = 20_000,
    max_pairs: int = 2_000,
    max_offset_px: int = 160,
    bands_mm: tuple[tuple[float, float], ...] = DEFAULT_BANDS_MM,
) -> dict[str, MetricResult]:
    """Mean absolute (mm) and relative error of pair distances per band, with coverage."""
    for name, tensor, channels in (
        ("predicted_points", predicted_points, 3),
        ("gt_points", gt_points, 3),
        ("gt_valid", gt_valid, 1),
        ("predicted_valid", predicted_valid, 1),
    ):
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != channels:
            raise MetricInputError(f"{name} must have shape [1, {channels}, H, W]")
    if gt_valid.dtype != torch.bool or predicted_valid.dtype != torch.bool:
        raise MetricInputError("validity masks must be boolean")
    height, width = int(gt_valid.shape[2]), int(gt_valid.shape[3])
    gt_ok = (gt_valid[0, 0] & torch.isfinite(gt_points[0]).all(dim=0)).cpu()
    pred_ok = (predicted_valid[0, 0] & torch.isfinite(predicted_points[0]).all(dim=0)).cpu()
    gt_cloud = gt_points[0].cpu()
    pred_cloud = predicted_points[0].cpu()
    valid_yx = torch.nonzero(gt_ok)
    results: dict[str, MetricResult] = {}
    if len(valid_yx) == 0:
        draw = None
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        first = valid_yx[torch.randint(len(valid_yx), (candidates,), generator=generator)]
        offset = torch.randint(
            -max_offset_px, max_offset_px + 1, (candidates, 2), generator=generator
        )
        second = first + offset
        inside = (
            (second[:, 0] >= 0)
            & (second[:, 0] < height)
            & (second[:, 1] >= 0)
            & (second[:, 1] < width)
        )
        first, second = first[inside], second[inside]
        both = gt_ok[second[:, 0], second[:, 1]]
        first, second = first[both], second[both]
        gt_a, gt_b = gt_cloud[:, first[:, 0], first[:, 1]], gt_cloud[:, second[:, 0], second[:, 1]]
        gt_distance = 1000.0 * torch.linalg.vector_norm(gt_a - gt_b, dim=0)
        pred_a, pred_b = (
            pred_cloud[:, first[:, 0], first[:, 1]],
            pred_cloud[:, second[:, 0], second[:, 1]],
        )
        pred_distance = 1000.0 * torch.linalg.vector_norm(pred_a - pred_b, dim=0)
        scored = (
            pred_ok[first[:, 0], first[:, 1]]
            & pred_ok[second[:, 0], second[:, 1]]
            & torch.isfinite(pred_distance)
        )
        draw = (gt_distance, pred_distance, scored)
    for low, high in bands_mm:
        label = f"{low:g}_{high:g}"
        definition = f"pairs of GT-valid pixels with GT 3D distance in [{low:g},{high:g}) mm, at most {max_pairs} per frame; "
        if draw is None:
            total, valid = 0, 0
            absolute = relative = None
        else:
            gt_distance, pred_distance, scored = draw
            in_band = ((gt_distance >= low) & (gt_distance < high)).nonzero().flatten()[:max_pairs]
            total = int(len(in_band))
            keep = in_band[scored[in_band]]
            valid = int(len(keep))
            if valid:
                error = (pred_distance[keep] - gt_distance[keep]).abs()
                absolute = float(error.mean())
                relative = float((error / gt_distance[keep]).mean())
            else:
                absolute = relative = None
        results[f"pair_dist_mae_mm_{label}"] = MetricResult(
            f"pair_dist_mae_mm_{label}",
            absolute,
            valid,
            total,
            definition + "mean abs(predicted distance - GT distance), mm",
        )
        results[f"pair_dist_rel_{label}"] = MetricResult(
            f"pair_dist_rel_{label}",
            relative,
            valid,
            total,
            definition + "mean abs error / GT distance",
        )
        results[f"pair_dist_coverage_{label}"] = MetricResult(
            f"pair_dist_coverage_{label}",
            valid / total if total else None,
            valid,
            total,
            definition + "scored pairs / drawn pairs",
        )
    return results
