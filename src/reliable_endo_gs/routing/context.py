"""Local/global, spatial and physical pre-action context for balanced routing."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from reliable_endo_gs.actions.repair import RepairState, primitive_core
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.gaussians import quaternion_to_rotation_matrix
from reliable_endo_gs.geometry.rectified import reconstruct_rectified
from reliable_endo_gs.geometry.stereo_matching import correspondence_error, project_left_support
from reliable_endo_gs.regions.grid import Region
from reliable_endo_gs.routing.repairability import (
    REPAIRABILITY_FEATURE_NAMES,
    repairability_features,
)

CONTEXT_NAMES = (
    (
        "global_left_error",
        "global_right_error",
        "global_warp_error",
        "global_warp_support",
        "global_disparity_mean",
        "global_disparity_std",
        "depth_mean_m",
        "depth_std_m",
        "fx_over_width",
        "fy_over_height",
        "baseline_m",
        "principal_offset_over_width",
        "log_scale_x",
        "log_scale_y",
        "log_scale_z",
        "log_anisotropy",
        "primitive_density",
        "normal_x",
        "normal_y",
        "normal_z",
        "visible_fraction",
        "radius_over_diagonal",
        "visibility_available",
        "cost_entropy",
        "cost_margin",
        "feature_energy",
        "matching_evidence_available",
    )
    + tuple(f"patch_warp_{i}" for i in range(4))
    + tuple(f"patch_render_error_{i}" for i in range(4))
)
BALANCED_FEATURE_NAMES = REPAIRABILITY_FEATURE_NAMES + CONTEXT_NAMES
AVAILABILITY_COLUMNS = tuple(
    i for i, name in enumerate(BALANCED_FEATURE_NAMES) if name.endswith("_available")
)


@torch.no_grad()
def balanced_features(
    state: RepairState,
    regions: tuple[Region, ...],
    config: InterventionConfig,
    *,
    scale_unit_m: float,
) -> torch.Tensor:
    """Detached [R,F] inputs; no GT/teacher/post-action targets. Batch one.

    Native visibility holds radii [1,N]; pixel masks are not relabeled as radii.
    Missing evidence has an explicit availability column. Log scales are in m.
    """
    if state.batch.left.shape[0] != 1 or state.base.render_right is None:
        raise ValueError("balanced context needs batch one and both base renders")
    basic = repairability_features(state, regions, config, scale_unit_m=scale_unit_m)
    batch = state.batch
    h, w = batch.left.shape[-2:]
    disparity = state.base.target.disparity
    error, warped_valid = correspondence_error(
        batch.left, batch.right, disparity, right_support=batch.support
    )
    left_error = (state.base.render_left.image - batch.left).abs().mean(1, keepdim=True)
    right_error = (state.base.render_right.image - batch.right).abs()
    geometry = reconstruct_rectified(
        disparity, batch.camera, state.support, min_effective_disparity=1e-6
    )
    depth = geometry.depth.depth
    evidence = getattr(state.base, "matching_evidence", {})
    radii = state.base.render_left.visibility
    visibility_available = (
        radii is not None and radii.shape == state.base.gaussians.valid_mask.shape
    )
    matching_available = all(
        k in evidence for k in ("cost_entropy", "cost_margin", "feature_energy")
    )

    def mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Fixed-shape reductions avoid CUDA nonzero/boolean-index compaction,
        # which must synchronize with the host to determine output size.
        expanded = mask.expand_as(value)
        return torch.where(expanded, value.float(), 0).sum() / expanded.sum().clamp_min(1)

    def std(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Two-pass centred variance avoids cancellation in E[x^2]-E[x]^2.
        centre = mean(value, mask)
        return mean((value.float() - centre).square(), mask).sqrt()

    def primitive_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask[:, None], value.float(), 0).sum(0) / mask.sum().clamp_min(1)

    global_values = torch.stack(
        (
            mean(left_error, state.support),
            mean(right_error, batch.support),
            mean(error, state.support),
            mean(warped_valid.float(), state.support),
            mean(disparity / w, state.support),
            torch.where(state.support.any(), std(disparity, state.support), float("nan")),
        )
    )
    frame = quaternion_to_rotation_matrix(state.base.gaussians.rotations)
    scales = state.base.gaussians.scales[0].float()
    log_scales = scales.clamp_min(1e-8).log()
    log_anisotropy = (scales.max(-1).values / scales.min(-1).values.clamp_min(1e-8)).log()
    rows = []
    for i, region in enumerate(regions):
        mask = region.mask(batch.left) & state.support
        right_mask = project_left_support(mask, disparity) & batch.support
        # Correct right-view region evidence; v1 retains its historical encoding.
        for channel in range(3):
            basic[i, 6 + channel] = mean(right_error[:, channel : channel + 1], right_mask)
        valid_depth = mask & geometry.depth.valid_mask
        slots = primitive_core(state, region)[0]
        zero = disparity.new_zeros((), dtype=torch.float32)
        physical = torch.stack(
            (
                mean(depth, valid_depth),
                std(depth, valid_depth),
                batch.camera.K_L[0, 0, 0].float() / w,
                batch.camera.K_L[0, 1, 1].float() / h,
                -batch.camera.T_R_from_L[0, 0, 3].float(),
                (batch.camera.K_L[0, 0, 2] - batch.camera.K_R[0, 0, 2]).float() / w,
            )
        )
        log_scale = primitive_mean(log_scales, slots)
        anisotropy = primitive_mean(log_anisotropy[:, None], slots)[0]
        density = slots.float().sum() / mask.float().sum().clamp_min(1)
        normal = primitive_mean(frame[0, :, :, 2], slots)
        visible = zero
        radius = zero
        if visibility_available:
            assert radii is not None
            selected = radii[0].float()[:, None]
            visible = primitive_mean((selected > 0).float(), slots)[0]
            radius = primitive_mean(selected, slots)[0] / (h * h + w * w) ** 0.5
        gaussians = torch.cat(
            (
                log_scale,
                torch.stack((anisotropy, density)),
                normal,
                torch.stack((visible, radius, zero.new_tensor(float(visibility_available)))),
            )
        )
        matching = (
            torch.stack(
                tuple(
                    mean(evidence[k], mask)
                    for k in ("cost_entropy", "cost_margin", "feature_energy")
                )
            )
            if matching_available
            else zero.expand(3)
        )
        y0, x0, y1, x1 = region.core

        def patches(
            value: torch.Tensor, bounds: tuple[int, int, int, int], selected: torch.Tensor
        ) -> torch.Tensor:
            y0, x0, y1, x1 = bounds
            crop = value[..., y0:y1, x0:x1]
            support = selected[..., y0:y1, x0:x1].float()
            numerator = F.adaptive_avg_pool2d(crop.float() * support, (2, 2))
            denominator = F.adaptive_avg_pool2d(support, (2, 2)).clamp_min(1e-6)
            return (numerator / denominator).flatten()

        rows.append(
            torch.cat(
                (
                    global_values,
                    physical,
                    gaussians,
                    matching,
                    zero.new_tensor([float(matching_available)]),
                    patches(error, region.core, mask),
                    patches(left_error, region.core, mask),
                )
            )
        )
    result = torch.cat((basic, torch.stack(rows)), 1)
    if result.shape[1] != len(BALANCED_FEATURE_NAMES) or not bool(torch.isfinite(result).all()):
        raise ValueError("invalid balanced context feature encoding")
    return result.detach()
