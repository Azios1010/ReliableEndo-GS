"""Offline GT geometry metrics, using the canonical calibrated transformations.

GT is never a deployment input. Length errors are reported in millimetres.
Sampled unsquared Chamfer/nearest-neighbour NC are explicitly diagnostics:
the published paper does not fully specify its point sampling/units protocol.
"""

from __future__ import annotations

import torch

from reliable_endo_gs.contracts.deployment import RectifiedStereoCamera
from reliable_endo_gs.evaluation.depth import depth_mae, depth_rmse
from reliable_endo_gs.evaluation.stereo import MetricInputError, MetricResult, endpoint_error
from reliable_endo_gs.geometry.rectified import reconstruct_rectified
from reliable_endo_gs.geometry.surface import estimate_surface_frame


def _subsample(points: torch.Tensor, maximum: int) -> torch.Tensor:
    if points.shape[0] <= maximum:
        return torch.arange(points.shape[0], device=points.device)
    return torch.linspace(0, points.shape[0] - 1, maximum, device=points.device).round().long()


def cloud_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    max_points: int,
    chunk_size: int,
    predicted_normals: torch.Tensor | None = None,
    target_normals: torch.Tensor | None = None,
) -> dict[str, MetricResult]:
    """Symmetric mean unsquared nearest-point distance (mm), optional abs-dot NC.

    Clouds/normals are finite ``[N,3]`` in the same left-camera frame, in metres.
    Independently keep evenly spaced row-major positions in each cloud. NC
    uses precisely the nearest neighbours used by Chamfer, in both directions.
    Chunking bounds pairwise memory; the cap is a declared approximation.
    """
    for points in (predicted, target):
        if points.ndim != 2 or points.shape[1] != 3 or not bool(torch.isfinite(points).all()):
            raise MetricInputError("clouds must be finite [N,3]")
    if max_points < 1 or chunk_size < 1 or predicted.device != target.device:
        raise MetricInputError("invalid cloud cap/chunk/device")
    definition = (
        f"symmetric_mean_unsquared_nearest_distance_mm rowmajor_equispaced_cap={max_points}"
    )
    if not len(predicted) or not len(target):
        empty = {
            "chamfer_mm": MetricResult(
                "chamfer_mm", None, 0, len(predicted) + len(target), definition
            )
        }
        if predicted_normals is not None and target_normals is not None:
            empty["normal_consistency"] = MetricResult(
                "normal_consistency",
                None,
                0,
                len(predicted) + len(target),
                f"symmetric_nearest_abs_unit_normal_dot cap={max_points}",
            )
        return empty
    pi, ti = _subsample(predicted, max_points), _subsample(target, max_points)
    p, t = predicted[pi].float(), target[ti].float()

    def nearest(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distances = []
        indices = []
        for start in range(0, len(a), chunk_size):
            d, i = torch.cdist(
                a[start : start + chunk_size], b, compute_mode="donot_use_mm_for_euclid_dist"
            ).min(1)
            distances.append(d)
            indices.append(i)
        return torch.cat(distances), torch.cat(indices)

    pd, pnearest = nearest(p, t)
    td, tnearest = nearest(t, p)
    cd = float((pd.mean() + td.mean()) * 500)
    result = {
        "chamfer_mm": MetricResult(
            "chamfer_mm", cd, len(p) + len(t), len(predicted) + len(target), definition
        )
    }
    if predicted_normals is not None and target_normals is not None:
        if predicted_normals.shape != predicted.shape or target_normals.shape != target.shape:
            raise MetricInputError("cloud normals must match their cloud shapes")
        pn, tn = predicted_normals[pi], target_normals[ti]
        for normals in (pn, tn):
            if not bool(torch.isfinite(normals).all()) or not bool((normals.norm(dim=1) > 0).all()):
                raise MetricInputError("cloud normals must be finite and nonzero")
        pn = torch.nn.functional.normalize(pn, dim=1)
        tn = torch.nn.functional.normalize(tn, dim=1)
        nc = float(
            (
                (pn * tn[pnearest]).sum(1).abs().clamp(max=1).mean()
                + (tn * pn[tnearest]).sum(1).abs().clamp(max=1).mean()
            )
            * 0.5
        )
        result["normal_consistency"] = MetricResult(
            "normal_consistency",
            nc,
            len(p) + len(t),
            len(predicted) + len(target),
            f"symmetric_nearest_abs_unit_normal_dot cap={max_points}",
        )
    return result


def evaluate_geometry(
    predicted_disparity: torch.Tensor,
    gt_disparity: torch.Tensor,
    gt_valid: torch.Tensor,
    predicted_valid: torch.Tensor,
    camera: RectifiedStereoCamera,
    *,
    cloud_max_points: int,
    cloud_chunk_size: int,
    gaussian_centres: torch.Tensor | None = None,
    gaussian_valid: torch.Tensor | None = None,
    gaussian_source_index: torch.Tensor | None = None,
    rendered_depth: torch.Tensor | None = None,
) -> tuple[dict[str, MetricResult], dict[str, object]]:
    """Evaluate one sample's offline GT on the exact deployment grid.

    Disparities/masks/depth are ``[1,1,H,W]``. Gaussian centres are ``[1,N,3]``
    in metres. Native RGB-only renderers yield explicit unavailable depth
    metrics; stereo depth and rendered depth are never substituted for one another.
    """
    if predicted_disparity.shape != gt_disparity.shape or predicted_disparity.shape[0] != 1:
        raise MetricInputError("benchmark geometry requires one matching stereo sample")
    if gt_valid.shape != gt_disparity.shape or predicted_valid.shape != gt_disparity.shape:
        raise MetricInputError("benchmark GT/prediction masks must match disparity")
    if gt_valid.dtype != torch.bool or predicted_valid.dtype != torch.bool:
        raise MetricInputError("benchmark geometry masks must be boolean")
    gt = reconstruct_rectified(gt_disparity, camera, gt_valid, min_effective_disparity=1e-6)
    pred = reconstruct_rectified(
        predicted_disparity, camera, predicted_valid, min_effective_disparity=1e-6
    )
    mask = gt.depth.valid_mask & pred.depth.valid_mask
    gt_count = int(gt.depth.valid_mask.sum())
    count = int(mask.sum())
    total = gt_valid.numel()
    errors = (predicted_disparity - gt_disparity).abs()
    metrics = {
        "disparity_epe": endpoint_error(predicted_disparity, gt_disparity, mask),
        "disparity_1px_accuracy": MetricResult(
            "disparity_1px_accuracy",
            float(((errors < 1) & mask).sum()) / gt_count if gt_count else None,
            gt_count,
            total,
            "fraction(abs_error<1px AND valid_prediction) over_all_valid_GT; invalid_prediction_is_failure",
        ),
        "geometry_coverage": MetricResult(
            "geometry_coverage",
            count / gt_count if gt_count else None,
            gt_count,
            total,
            "valid_prediction_and_GT / valid_GT",
        ),
    }
    for key, value in {
        "mae_mm": depth_mae(pred.depth.depth * 1000, gt.depth.depth * 1000, mask),
        "rmse_mm": depth_rmse(pred.depth.depth * 1000, gt.depth.depth * 1000, mask),
    }.items():
        metrics[f"stereo_depth_{key}"] = value
    metrics["stereo_depth_absrel"] = MetricResult(
        "stereo_depth_absrel",
        float(((pred.depth.depth - gt.depth.depth).abs() / gt.depth.depth)[mask].mean())
        if count
        else None,
        count,
        total,
        "mean(abs(predicted_depth-GT_depth)/GT_depth) on_joint_valid",
    )
    # No fallback ray normals are used as evaluated surface normals.
    ps = estimate_surface_frame(pred.points.centers, pred.points.valid_mask, layout="channel_first")
    ts = estimate_surface_frame(gt.points.centers, gt.points.valid_mask, layout="channel_first")
    nmask = mask[:, 0] & ps.normal_valid_mask & ts.normal_valid_mask
    pp = pred.points.centers.permute(0, 2, 3, 1)[nmask]
    tp = gt.points.centers.permute(0, 2, 3, 1)[nmask]
    for key, value in cloud_metrics(
        pp,
        tp,
        max_points=cloud_max_points,
        chunk_size=cloud_chunk_size,
        predicted_normals=ps.normals[nmask],
        target_normals=ts.normals[nmask],
    ).items():
        metrics[f"stereo_surface_{key}"] = value
    if gaussian_centres is not None and gaussian_valid is not None:
        if gaussian_source_index is None or gaussian_source_index.shape != gaussian_valid.shape:
            raise MetricInputError("Gaussian cloud evaluation requires source pixel correspondence")
        if gaussian_source_index.dtype != torch.long:
            raise MetricInputError("Gaussian source index must be int64")
        in_grid = (gaussian_source_index >= 0) & (gaussian_source_index < gt_valid.numel())
        indices = gaussian_source_index.clamp(0, gt_valid.numel() - 1)
        gt_supported = gt.points.valid_mask.flatten()[indices] & in_grid
        cloud = gaussian_centres[
            gaussian_valid & gt_supported & torch.isfinite(gaussian_centres).all(-1)
        ]
        gt_cloud = gt.points.centers.permute(0, 2, 3, 1)[gt.points.valid_mask[:, 0]]
        for key, value in cloud_metrics(
            cloud, gt_cloud, max_points=cloud_max_points, chunk_size=cloud_chunk_size
        ).items():
            metrics[f"gaussian_centres_{key}"] = value
    reason = None
    if rendered_depth is None:
        reason = "renderer_does_not_export_metric_depth; stereo_depth_is_reported_separately"
        for key in ("mae_mm", "rmse_mm", "absrel"):
            name = f"rendered_depth_{key}"
            metrics[name] = MetricResult(name, None, 0, total, reason)
        metrics["rendered_depth_coverage"] = MetricResult(
            "rendered_depth_coverage", None, 0, total, reason
        )
    else:
        rm = gt.depth.valid_mask & torch.isfinite(rendered_depth) & (rendered_depth > 0)
        metrics["rendered_depth_coverage"] = MetricResult(
            "rendered_depth_coverage",
            int(rm.sum()) / gt_count if gt_count else None,
            gt_count,
            total,
            "positive_finite_render_depth_on_valid_GT / valid_GT",
        )
        metrics["rendered_depth_mae_mm"] = depth_mae(
            rendered_depth * 1000, gt.depth.depth * 1000, rm
        )
        metrics["rendered_depth_rmse_mm"] = depth_rmse(
            rendered_depth * 1000, gt.depth.depth * 1000, rm
        )
        metrics["rendered_depth_absrel"] = MetricResult(
            "rendered_depth_absrel",
            float(((rendered_depth - gt.depth.depth).abs() / gt.depth.depth)[rm].mean())
            if bool(rm.any())
            else None,
            int(rm.sum()),
            total,
            "mean(abs(render_depth-GT_depth)/GT_depth) on_joint_valid",
        )
    return metrics, {
        "gt_valid_pixels": gt_count,
        "joint_valid_pixels": count,
        "normal_joint_valid_pixels": int(nmask.sum()),
        "rendered_depth_unavailable": reason,
        "gt_depth_source": "rectified_SCARED_GT_disparity_via_canonical_K_baseline_in_metres",
        "geometry_paper_protocol_equivalence": "unverified_sampling_and_CD_NC_definitions",
    }
