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


class RepairActions(nn.Module):
    """Trainable local disparity head and 13-channel Gaussian residual MLP."""

    def __init__(
        self,
        reference: ReferenceReconstructionModel,
        *,
        width: int = 32,
        max_disparity_px: float = 4.0,
        composition_alpha_epsilon: float = 0.0000001,
    ) -> None:
        super().__init__()
        if width < 1 or not 0 < max_disparity_px < float("inf"):
            raise ValueError("repair width and disparity bound must be positive")
        self.max_disparity_px = max_disparity_px
        self.gaussian_bounds = replace(
            reference.config.residual_bounds, alpha_logit_epsilon=composition_alpha_epsilon
        )
        self.stereo = nn.Sequential(
            nn.Conv2d(10, width, 1),
            nn.ReLU(),
            nn.Conv2d(width, width, 1),
            nn.ReLU(),
            nn.Conv2d(width, 1, 1),
        )
        stereo_output = self.stereo[-1]
        if not isinstance(stereo_output, nn.Conv2d) or stereo_output.bias is None:
            raise TypeError("stereo output must be a Conv2d with bias")
        nn.init.zeros_(stereo_output.weight)
        nn.init.zeros_(stereo_output.bias)
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
            return ActionResult(
                action,
                base.gaussians,
                base.target.disparity,
                base.render_left,
                base.render_right,
                (),
            )
        if not eligible(state, region):
            raise ValueError("repair requested on an ineligible region")
        if not isinstance(reference, StudentReconstructionModel) and any(
            parameter.requires_grad for parameter in reference.parameters()
        ):
            raise ValueError("reference parameters must be frozen before repair")
        if base.gaussians.cov_center is not None or base.gaussians.cov_effective is not None:
            raise ValueError("A04 repair does not implement uncertainty covariance updates")
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
            prediction = self.stereo(inputs[..., y0:y1, x0:x1])
            delta = torch.nn.functional.pad(
                prediction, (x0, batch.left.shape[-1] - x1, y0, batch.left.shape[-2] - y1)
            )
            # Principal-point offset is not depth; the canonical A03 constructor
            # still owns triangulation. Bound corrections to 25% of positive
            # effective disparity to preserve the base support/sample ordering.
            offset = (batch.camera.K_L[:, 0, 2] - batch.camera.K_R[:, 0, 2])[:, None, None, None]
            limit = torch.minimum(
                (disparity - offset).detach().clamp_min(0) * 0.25,
                disparity.new_tensor(self.max_disparity_px),
            )
            disparity = disparity + torch.where(
                region.mask(batch.left) & state.support,
                delta.tanh().to(disparity.dtype) * limit,
                torch.zeros_like(disparity),
            )
            prior = build_geometry_prior(
                disparity,
                batch.camera,
                state.support,
                batch.left,
                policy=reference.config.prior_policy,
            )
            if not torch.equal(prior.source_index, base.prior.source_index):
                raise ValueError("stereo repair changed source sampling correspondence")
            context = context_from_geometry_prior(
                prior, context_scale_m=reference.config.context_scale_m
            )
            # Match the frozen base head's AMP policy. Re-evaluating it in
            # FP32 would make a zero stereo correction change the reference
            # attributes and contaminate counterfactual utility labels.
            amp_dtype = (
                torch.float16 if reference.config.runtime.precision == "fp16" else torch.bfloat16
            )
            with torch.autocast(
                device_type=batch.left.device.type,
                dtype=amp_dtype,
                enabled=(
                    batch.left.device.type == "cuda"
                    and reference.config.runtime.precision in ("fp16", "bf16")
                ),
            ):
                residuals = reference.head(context, prior.valid_mask)
            changed = compose_gaussian_field(
                prior, residuals, bounds=reference.config.residual_bounds
            )
            names = ("disparity", "centres", "surface_frame", "scale", "opacity", "cov_surface")
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
            residuals = self.gaussian(context, core)
            changed = compose_gaussian_field(
                prior,
                AttributeResiduals(
                    residuals.delta_rotation, residuals.delta_log_scale, residuals.delta_alpha_logit
                ),
                bounds=self.gaussian_bounds,
            )
            names = ("rotation", "scale", "opacity", "cov_surface")
        else:
            raise ValueError(f"unsupported repair action: {action}")
        field = merge_field(base.gaussians, changed, core)
        render_field = replace(field, cov_surface=None)
        hw = (int(batch.left.shape[-2]), int(batch.left.shape[-1]))
        if reference.renderer is None:
            raise ValueError("reference renderer is unavailable")
        renderer = cast(FieldRenderer, reference.renderer)
        left = renderer.render(FieldRenderRequest(render_field, batch.camera, hw, "left"))
        right = renderer.render(FieldRenderRequest(render_field, batch.camera, hw, "right"))
        return ActionResult(action, field, disparity, left, right, names)
