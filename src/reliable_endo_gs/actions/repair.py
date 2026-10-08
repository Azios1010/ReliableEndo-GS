"""One reusable STOP/stereo/Gaussian implementation for the A04 extension.

Only core pixels/primitives are written. Stereo recomputes canonical A03
geometry on the full grid (including neighbour reads) and then merges by
source_index. Both repairs currently incur full-frame rendering. This is a
quality experiment, not a claim of region-proportional execution cost.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import IntEnum
from typing import Any, cast

import torch
from torch import nn

from reliable_endo_gs.actions.stereo_matching import MatchingStereoRepair
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch
from reliable_endo_gs.contracts.gaussians import GaussianField
from reliable_endo_gs.contracts.rendering import RenderOutput
from reliable_endo_gs.gaussians import (
    AttributeResiduals,
    ReferenceAttributeHead,
    build_geometry_prior,
    compose_gaussian_field,
    context_from_geometry_prior,
    quaternion_to_rotation_matrix,
)
from reliable_endo_gs.reconstruction.model import (
    ReferenceReconstructionModel,
    ReferenceReconstructionResult,
)
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.regions.grid import Region
from reliable_endo_gs.regions.masks import MaskRegion
from reliable_endo_gs.rendering.field import FieldRenderer, FieldRenderRequest


class Action(IntEnum):
    STOP = 0
    STEREO = 1
    GAUSSIAN = 2


@dataclass(frozen=True, slots=True)
class RepairState:
    """Borrowed immutable deployment batch and frozen pre-action evidence."""

    batch: DeploymentStereoBatch
    base: ReferenceReconstructionResult

    @property
    def support(self) -> torch.Tensor:
        """Sensor plus predicted teacher validity; never a GT mask."""
        return self.batch.support & self.base.target.support & self.base.target.validity


@dataclass(frozen=True, slots=True)
class ActionResult:
    action: Action
    field: GaussianField
    disparity: torch.Tensor
    left: RenderOutput
    right: RenderOutput
    changed_fields: tuple[str, ...]


def _base_result(
    state: RepairState, action: Action, disparity: torch.Tensor | None = None
) -> ActionResult:
    """Return the borrowed pre-action result without another renderer pass."""

    base = state.base
    if base.render_right is None:
        raise ValueError("Phase II requires both base renders")
    return ActionResult(
        action,
        base.gaussians,
        base.target.disparity if disparity is None else disparity,
        base.render_left,
        base.render_right,
        (),
    )


def primitive_core(state: RepairState, region: Region) -> torch.Tensor:
    """Map core pixels to Gaussian slots using canonical source identities."""
    flat = region.mask(state.batch.left).reshape(1, -1)
    index = state.base.prior.source_index
    return flat.gather(1, index.clamp_min(0)) & state.base.gaussians.valid_mask & (index >= 0)


def eligible(state: RepairState, region: Region) -> bool:
    """Empty sensor/teacher support or empty primitive cores are ineligible."""
    return bool(
        (state.support & region.mask(state.batch.left)).any() & primitive_core(state, region).any()
    )


def action_eligible(state: RepairState, region: Region, action: Action) -> bool:
    """Check action-specific deployment preconditions before executing a repair.

    ``eligible`` is intentionally retained as the shared support/primitive
    predicate.  Stereo additionally requires a finite disparity map; Gaussian
    repair additionally requires finite, positive attributes on the selected
    primitive slots.  These checks prevent a selector prediction from turning
    missing evidence or padded primitives into a real intervention.
    """

    if action is Action.STOP:
        return True
    if action not in (Action.STEREO, Action.GAUSSIAN) or not eligible(state, region):
        return False
    core = primitive_core(state, region)
    field = state.base.gaussians
    if action is Action.STEREO:
        writable = region.mask(state.batch.left) & state.support
        return bool(torch.isfinite(state.base.target.disparity[writable]).all())
    return bool(
        torch.isfinite(field.means3d[core]).all()
        and torch.isfinite(field.rotations[core]).all()
        and torch.isfinite(field.scales[core]).all()
        and torch.isfinite(field.opacities[core]).all()
        and (field.scales[core] > 0).all()
    )


def merge_field(base: GaussianField, changed: GaussianField, mask: torch.Tensor) -> GaussianField:
    """Preserve every out-of-core slot exactly, including covariance metadata."""
    values: dict[str, Any] = {}
    for item in fields(base):
        name = item.name
        before, after = getattr(base, name), getattr(changed, name)
        if before is None or after is None:
            if before is not None or after is not None:
                raise ValueError(f"repair changed optional field availability: {name}")
            values[name] = None
        else:
            expanded = mask.reshape(*mask.shape, *([1] * (before.ndim - 2)))
            values[name] = torch.where(expanded, after, before)
    return replace(base, **values)


def preserve_zero_gaussian_residual(
    base: GaussianField, changed: GaussianField, residuals: AttributeResiduals
) -> GaussianField:
    """Exact identity at zero [B,N,7] residual, with composition derivatives.

    Quaternion -> matrix -> quaternion is not bitwise identity in FP32. At
    exactly zero residual only, anchor the forward value to the actual base
    field. ``base.detach() + (changed - changed.detach())`` retains the canonical
    composition Jacobian, including its upstream base derivatives; directly
    returning base would suppress repair gradients. Nonzero interventions use
    canonical composition unchanged. No nonzero tolerance/threshold is used.
    """
    zero = (
        (residuals.delta_rotation == 0).all(-1)
        & (residuals.delta_log_scale == 0).all(-1)
        & (residuals.delta_alpha_logit == 0).all(-1)
    )
    values: dict[str, Any] = {}
    for item in fields(base):
        before, after = getattr(base, item.name), getattr(changed, item.name)
        if before is None or after is None:
            if before is not None or after is not None:
                raise ValueError(f"zero residual changed optional field availability: {item.name}")
            values[item.name] = None
        else:
            mask = zero.reshape(*zero.shape, *([1] * (after.ndim - 2)))
            identity = (
                before.detach() + (after - after.detach()) if after.is_floating_point() else before
            )
            values[item.name] = torch.where(mask, identity, after)
    return replace(changed, **values)


class RepairActions(nn.Module):
    """Trainable local disparity head and 13-channel Gaussian residual MLP."""

    def __init__(
        self,
        reference: ReferenceReconstructionModel,
        *,
        width: int = 32,
        max_disparity_px: float = 4.0,
        composition_alpha_epsilon: float = 0.0000001,
        matching_head: MatchingStereoRepair | None = None,
        effective_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        if width < 1 or not 0 < max_disparity_px < float("inf"):
            raise ValueError("repair width and disparity bound must be positive")
        self.max_disparity_px = max_disparity_px
        if not 0 < effective_fraction < 1:
            raise ValueError("effective correction fraction must be in (0,1)")
        self.effective_fraction = effective_fraction
        # Opt-in numerical identity fix; historical repair semantics are retained.
        self.exact_zero_identity = matching_head is not None
        self.gaussian_bounds = replace(
            reference.config.residual_bounds, alpha_logit_epsilon=composition_alpha_epsilon
        )
        legacy_stereo = nn.Sequential(
            nn.Conv2d(10, width, 1),
            nn.ReLU(),
            nn.Conv2d(width, width, 1),
            nn.ReLU(),
            nn.Conv2d(width, 1, 1),
        )
        stereo_output = legacy_stereo[-1]
        if not isinstance(stereo_output, nn.Conv2d) or stereo_output.bias is None:
            raise TypeError("stereo output must be a Conv2d with bias")
        nn.init.zeros_(stereo_output.weight)
        nn.init.zeros_(stereo_output.bias)
        self.stereo: nn.Module = legacy_stereo if matching_head is None else matching_head
        self.gaussian = ReferenceAttributeHead(
            hidden_dims=(width, width), bounds=reference.config.residual_bounds
        )

    def execute(
        self,
        state: RepairState,
        region: Region,
        action: Action,
        reference: ReferenceReconstructionModel,
    ) -> ActionResult:
        """Apply one intervention from the same base; never chain repairs."""
        base, batch = state.base, state.batch
        if base.render_right is None:
            raise ValueError("Phase II requires both base renders")
        if batch.left.shape[0] != 1:
            raise ValueError("Phase II supports batch-size one")
        if action == Action.STOP:
            return _base_result(state, action)
        if base.gaussians.cov_center is not None or base.gaussians.cov_effective is not None:
            raise ValueError("A04 repair does not implement uncertainty covariance updates")
        if not action_eligible(state, region, action):
            raise ValueError(f"{action.name.lower()} repair requested on an ineligible region")
        if not isinstance(reference, StudentReconstructionModel) and any(
            parameter.requires_grad for parameter in reference.parameters()
        ):
            raise ValueError("reference parameters must be frozen before repair")
        disparity = base.target.disparity
        core = primitive_core(state, region)
        names: tuple[str, ...]
        if action == Action.STEREO:
            # Pointwise learned head reads the declared halo; geometry reads the
            # full grid to preserve the canonical surface-neighbour semantics.
            y0, x0, y1, x1 = region.halo
            inputs = torch.cat(
                (
                    batch.left,
                    batch.right,
                    disparity / batch.left.shape[-1],
                    base.render_left.image.detach() - batch.left,
                ),
                dim=1,
            ).float()
            if isinstance(self.stereo, MatchingStereoRepair):
                prediction, matching_valid = self.stereo.forward_pair(
                    batch.left,
                    batch.right,
                    disparity,
                    base.render_left.image.detach() - batch.left,
                    region.halo,
                    batch.support,
                    (batch.camera.K_L[:, 0, 2] - batch.camera.K_R[:, 0, 2])[:, None, None, None],
                    roi_features=isinstance(region, MaskRegion),
                    return_validity=True,
                )
            else:
                prediction = self.stereo(inputs[..., y0:y1, x0:x1])
            delta = torch.nn.functional.pad(
                prediction, (x0, batch.left.shape[-1] - x1, y0, batch.left.shape[-2] - y1)
            )
            # Principal-point offset is not depth; the canonical A03 constructor
            # still owns triangulation. Bound corrections to 25% of positive
            # effective disparity to preserve the base support/sample ordering.
            offset = (batch.camera.K_L[:, 0, 2] - batch.camera.K_R[:, 0, 2])[:, None, None, None]
            limit = torch.minimum(
                (disparity - offset).detach().clamp_min(0) * self.effective_fraction,
                disparity.new_tensor(self.max_disparity_px),
            )
            disparity = disparity + torch.where(
                region.mask(batch.left)
                & state.support
                & (
                    torch.nn.functional.pad(
                        matching_valid,
                        (x0, batch.left.shape[-1] - x1, y0, batch.left.shape[-2] - y1),
                    )
                    if isinstance(self.stereo, MatchingStereoRepair)
                    else torch.ones_like(state.support)
                ),
                delta.tanh().to(disparity.dtype) * limit,
                torch.zeros_like(disparity),
            )
            if not torch.is_grad_enabled() and torch.equal(disparity, base.target.disparity):
                return _base_result(state, action, disparity)
            return geometry_repair_outcome(state, region, disparity, reference)
        elif action == Action.GAUSSIAN:
            field = base.gaussians
            # Compose against actual A04 attributes, rather than replacing the
            # trained head output with an untrained A03 prior.
            prior = replace(
                base.prior,
                mu=field.means3d,
                R0=quaternion_to_rotation_matrix(field.rotations),
                rotations=field.rotations,
                S0=field.scales,
                alpha0=field.opacities,
                rgb=field.colors,
            )
            context = context_from_geometry_prior(
                prior, context_scale_m=reference.config.context_scale_m
            )
            residuals = (
                self.gaussian.forward_selected(context, core)
                if isinstance(region, MaskRegion)
                else self.gaussian(context, core)
            )
            if not torch.is_grad_enabled() and all(
                bool(torch.equal(value, torch.zeros_like(value)))
                for value in (
                    residuals.delta_rotation,
                    residuals.delta_log_scale,
                    residuals.delta_alpha_logit,
                )
            ):
                return _base_result(state, action)
            changed = compose_gaussian_field(
                prior,
                AttributeResiduals(
                    residuals.delta_rotation, residuals.delta_log_scale, residuals.delta_alpha_logit
                ),
                bounds=self.gaussian_bounds,
            )
            if self.exact_zero_identity:
                changed = preserve_zero_gaussian_residual(field, changed, residuals)
            names = ("rotation", "scale", "opacity", "cov_surface")
            if not torch.is_grad_enabled() and all(
                torch.equal(getattr(field, item.name), getattr(changed, item.name))
                for item in fields(field)
                if getattr(field, item.name) is not None
            ):
                return _base_result(state, action)
        else:
            raise ValueError(f"unsupported repair action: {action}")
        return _render_outcome(
            state, merge_field(base.gaussians, changed, core), disparity, reference, action, names
        )


def geometry_repair_outcome(
    state: RepairState,
    region: Region,
    disparity: torch.Tensor,
    reference: ReferenceReconstructionModel,
) -> ActionResult:
    """Shared A03/head/render path for learned and deterministic geometry actions.

    Disparity is raw [1,1,H,W] pixels. Only core values may differ from base;
    Gaussian source indices and every out-of-core slot must remain unchanged.
    The canonical A03 functions own triangulation/surface construction.
    """
    base, batch = state.base, state.batch
    if disparity.shape != base.target.disparity.shape or disparity.device != batch.left.device:
        raise ValueError("geometry repair disparity grid/device mismatch")
    if base.gaussians.cov_center is not None or base.gaussians.cov_effective is not None:
        raise ValueError("A04 repair does not implement uncertainty covariance updates")
    if not action_eligible(state, region, Action.STEREO) or batch.left.shape[0] != 1:
        raise ValueError("ineligible geometry repair")
    if not torch.is_grad_enabled() and torch.equal(disparity, base.target.disparity):
        return _base_result(state, Action.STEREO, disparity)
    outside = ~region.mask(batch.left)
    if not torch.equal(disparity[outside], base.target.disparity[outside]):
        raise ValueError("geometry repair changed disparity outside core")
    prior = build_geometry_prior(
        disparity, batch.camera, state.support, batch.left, policy=reference.config.prior_policy
    )
    if not torch.equal(prior.source_index, base.prior.source_index):
        raise ValueError("stereo repair changed source sampling correspondence")
    context = context_from_geometry_prior(prior, context_scale_m=reference.config.context_scale_m)
    # Preserve the established head AMP policy and canonical composition.
    amp_dtype = torch.float16 if reference.config.runtime.precision == "fp16" else torch.bfloat16
    with torch.autocast(
        device_type=batch.left.device.type,
        dtype=amp_dtype,
        enabled=(
            batch.left.device.type == "cuda"
            and reference.config.runtime.precision in ("fp16", "bf16")
        ),
    ):
        residuals = (
            reference.head.forward_selected(context, primitive_core(state, region))
            if isinstance(region, MaskRegion) and isinstance(reference.head, ReferenceAttributeHead)
            else reference.head(context, prior.valid_mask)
        )
    changed = compose_gaussian_field(prior, residuals, bounds=reference.config.residual_bounds)
    names = ("disparity", "centres", "surface_frame", "scale", "opacity", "cov_surface")
    return _render_outcome(
        state,
        merge_field(base.gaussians, changed, primitive_core(state, region)),
        disparity,
        reference,
        Action.STEREO,
        names,
    )


def _render_outcome(
    state: RepairState,
    field: GaussianField,
    disparity: torch.Tensor,
    reference: ReferenceReconstructionModel,
    action: Action,
    names: tuple[str, ...],
) -> ActionResult:
    # The current renderer consumes only R/S support. Keep optional covariance
    # on the ActionResult for provenance, but never pass unsupported covariance
    # tensors across the renderer boundary.
    render_field = replace(field, cov_surface=None, cov_center=None, cov_effective=None)
    hw = (int(state.batch.left.shape[-2]), int(state.batch.left.shape[-1]))
    if reference.renderer is None:
        raise ValueError("reference renderer is unavailable")
    renderer = cast(FieldRenderer, reference.renderer)
    left = renderer.render(FieldRenderRequest(render_field, state.batch.camera, hw, "left"))
    right = renderer.render(FieldRenderRequest(render_field, state.batch.camera, hw, "right"))
    return ActionResult(action, field, disparity, left, right, names)
