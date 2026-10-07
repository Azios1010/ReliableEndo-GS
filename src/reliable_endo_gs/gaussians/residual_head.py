"""Bounded attribute residual head for the A04 quality reference.

The reference head is deliberately independent of any stereo implementation.
It consumes local geometry/RGB context assembled from :class:`GeometryPrior`
and predicts only bounded residuals.  A final zero-initialized projection makes
the initial model exactly the A03 priors-only field while keeping the residual
parameters trainable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn

from reliable_endo_gs.gaussians.priors import GeometryPrior


@dataclass(frozen=True, slots=True)
class ResidualBounds:
    """Declared bounds for local-frame attribute residuals.

    Rotation is an axis-angle vector in radians, scale is a log multiplier,
    and opacity is an additive logit.  Absolute metric scale limits are
    applied again by the canonical composition owner.
    """

    max_rotation_rad: float = 0.35
    max_log_scale: float = 0.75
    max_alpha_logit: float = 2.0
    min_scale_m: float = 1.0e-6
    max_scale_m: float = 0.5
    alpha_logit_epsilon: float = 1.0e-4

    def __post_init__(self) -> None:
        for name in (
            "max_rotation_rad",
            "max_log_scale",
            "max_alpha_logit",
            "min_scale_m",
            "max_scale_m",
            "alpha_logit_epsilon",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError(f"{name} must be a finite positive number")
            if not isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_scale_m < self.min_scale_m:
            raise ValueError("max_scale_m must be >= min_scale_m")
        if self.alpha_logit_epsilon >= 0.5:
            raise ValueError("alpha_logit_epsilon must be less than 0.5")

    def to_dict(self) -> dict[str, float]:
        """Return a JSON/YAML-compatible bound mapping."""

        return {
            "max_rotation_rad": float(self.max_rotation_rad),
            "max_log_scale": float(self.max_log_scale),
            "max_alpha_logit": float(self.max_alpha_logit),
            "min_scale_m": float(self.min_scale_m),
            "max_scale_m": float(self.max_scale_m),
            "alpha_logit_epsilon": float(self.alpha_logit_epsilon),
        }

    @classmethod
    def from_dict(cls, value: object) -> ResidualBounds:
        """Build bounds from a strict mapping."""

        if not isinstance(value, dict):
            raise TypeError("residual bounds must be a mapping")
        expected = {
            "max_rotation_rad",
            "max_log_scale",
            "max_alpha_logit",
            "min_scale_m",
            "max_scale_m",
            "alpha_logit_epsilon",
        }
        if set(value) != expected:
            raise ValueError(f"residual bounds keys must be exactly {sorted(expected)}")
        return cls(**{key: float(value[key]) for key in expected})


@dataclass(frozen=True, slots=True)
class AttributeResiduals:
    """Head output with explicit local-frame residual semantics.

    Tensors have shape ``[B, N, 3]``, ``[B, N, 3]`` and ``[B, N, 1]`` for
    rotation, log-scale and alpha-logit respectively.  The head's outputs
    are already bounded, but the composition owner clamps again at its
    boundary so callers cannot bypass the safety contract.
    """

    delta_rotation: torch.Tensor
    delta_log_scale: torch.Tensor
    delta_alpha_logit: torch.Tensor

    def __post_init__(self) -> None:
        reference = self.delta_rotation
        if not isinstance(reference, torch.Tensor) or reference.ndim != 3:
            raise ValueError("delta_rotation must have shape [B,N,3]")
        if reference.shape[-1] != 3 or not torch.is_floating_point(reference):
            raise ValueError("delta_rotation must be floating point [B,N,3]")
        for name, value, width in (
            ("delta_log_scale", self.delta_log_scale, 3),
            ("delta_alpha_logit", self.delta_alpha_logit, 1),
        ):
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise ValueError(f"{name} must have shape [B,N,{width}]")
            if tuple(value.shape) != (reference.shape[0], reference.shape[1], width):
                raise ValueError(f"{name} must have shape [B,N,{width}]")
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must be floating point")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise TypeError(f"{name} must share dtype/device with delta_rotation")

    @property
    def rotation(self) -> torch.Tensor:
        """Compatibility alias for the axis-angle residual."""

        return self.delta_rotation

    @property
    def log_scale(self) -> torch.Tensor:
        """Compatibility alias for the log-scale residual."""

        return self.delta_log_scale

    @property
    def alpha_logit(self) -> torch.Tensor:
        """Compatibility alias for the opacity-logit residual."""

        return self.delta_alpha_logit


def context_from_geometry_prior(
    prior: GeometryPrior, *, context_scale_m: float = 1.0
) -> torch.Tensor:
    """Assemble local RGB/geometry context from an A03 prior.

    The context is ``[rgb, mu/context_scale, normal, S0/context_scale,
    alpha0]`` with 13 channels.  It intentionally contains no teacher
    feature or hidden upstream state.  ``prior`` is duck-typed to keep this
    head independent of the constructor package at import time.
    """

    if isinstance(context_scale_m, bool) or not isinstance(context_scale_m, (float, int)):
        raise TypeError("context_scale_m must be a finite positive number")
    if not isfinite(float(context_scale_m)) or float(context_scale_m) <= 0:
        raise ValueError("context_scale_m must be a finite positive number")
    rgb = prior.rgb
    means = prior.mu
    frame = prior.R0
    scales = prior.S0
    alpha = prior.alpha0
    if not all(isinstance(value, torch.Tensor) for value in (rgb, means, frame, scales, alpha)):
        raise TypeError("prior geometry fields must be tensors")
    if frame.ndim != 4 or frame.shape[-2:] != (3, 3):
        raise ValueError("prior.R0 must have shape [B,N,3,3]")
    context = torch.cat(
        (
            rgb,
            means / float(context_scale_m),
            frame[..., :, 2],
            scales / float(context_scale_m),
            alpha,
        ),
        dim=-1,
    )
    if context.shape[-1] != 13:
        raise ValueError(f"assembled prior context must have 13 channels; got {context.shape[-1]}")
    valid = prior.valid_mask
    if valid.shape != context.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("prior.valid_mask must be bool [B,N]")
    context = torch.where(valid.unsqueeze(-1), context, torch.zeros_like(context))
    return context


class BoundedAttributeResidualHead(nn.Module):
    """Heavier reference-only MLP for bounded Gaussian attribute residuals."""

    CONTEXT_CHANNELS = 13

    def __init__(
        self,
        input_dim: int = CONTEXT_CHANNELS,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        bounds: ResidualBounds | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        hidden = tuple(hidden_dims)
        if not hidden or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in hidden
        ):
            raise ValueError("hidden_dims must contain positive integers")
        self.input_dim = input_dim
        self.hidden_dims = hidden
        self.bounds = bounds or ResidualBounds()

        layers: list[nn.Module] = []
        in_features = input_dim
        for width in hidden:
            layers.extend((nn.Linear(in_features, width), nn.GELU()))
            in_features = width
        self.body = nn.Sequential(*layers)
        self.output = nn.Linear(in_features, 7)
        # The residual branch starts as an exact identity composition.  The
        # output layer remains trainable and therefore receives gradients from
        # the first backward pass even though its initial value is zero.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward_selected(self, context: torch.Tensor, selected: torch.Tensor) -> AttributeResiduals:
        """Evaluate only selected pointwise contexts and scatter BxNx{3,3,1}.

        This head has no cross-primitive operations, so gathering preserves its
        semantics. Geometry construction and field composition are separate.
        """
        if (
            context.ndim != 3
            or selected.shape != context.shape[:2]
            or selected.dtype != torch.bool
            or selected.device != context.device
        ):
            raise ValueError("selected head requires context BNC and on-device bool BN")
        indices = selected.reshape(-1).nonzero().flatten()
        if not indices.numel():
            raise ValueError("selected head requires at least one primitive")
        compact = context.reshape(-1, context.shape[-1])[indices][None]
        residuals = self(compact, torch.ones_like(indices, dtype=torch.bool)[None])

        def scatter(value: torch.Tensor) -> torch.Tensor:
            flat = value.new_zeros((selected.numel(), value.shape[-1]))
            return flat.index_copy(0, indices, value[0]).reshape(*selected.shape, value.shape[-1])

        return AttributeResiduals(
            scatter(residuals.delta_rotation),
            scatter(residuals.delta_log_scale),
            scatter(residuals.delta_alpha_logit),
        )

    def forward(
        self,
        context: torch.Tensor | object,
        valid_mask: torch.Tensor | None = None,
    ) -> AttributeResiduals:
        """Predict bounded residuals from ``[B,N,C]`` context or an A03 prior."""

        if not isinstance(context, torch.Tensor):
            if not isinstance(context, GeometryPrior):
                raise TypeError("context must be a tensor or GeometryPrior")
            context = context_from_geometry_prior(context)
        if context.ndim != 3 or context.shape[-1] != self.input_dim:
            raise ValueError(
                f"context must have shape [B,N,{self.input_dim}]; got {tuple(context.shape)}"
            )
        if not torch.is_floating_point(context):
            raise TypeError("context must be floating point")
        output_dtype = context.dtype
        parameter = next(self.parameters(), None)
        if parameter is not None and parameter.dtype != context.dtype:
            # Linear layers retain their declared training dtype.  Returning
            # to the caller's context dtype keeps A03 FP64 analytic paths and
            # FP16/BF16 deployment paths connected to one composition owner.
            context = context.to(dtype=parameter.dtype)
        if valid_mask is not None:
            if valid_mask.shape != context.shape[:2] or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be bool [B,N]")
            if valid_mask.device != context.device:
                raise ValueError("valid_mask must share context device")
            context = torch.where(valid_mask.unsqueeze(-1), context, torch.zeros_like(context))

        raw = self.output(self.body(context))
        bounds = self.bounds
        rotation = bounds.max_rotation_rad * torch.tanh(raw[..., 0:3] / bounds.max_rotation_rad)
        log_scale = bounds.max_log_scale * torch.tanh(raw[..., 3:6] / bounds.max_log_scale)
        alpha = bounds.max_alpha_logit * torch.tanh(raw[..., 6:7] / bounds.max_alpha_logit)
        rotation = rotation.to(dtype=output_dtype)
        log_scale = log_scale.to(dtype=output_dtype)
        alpha = alpha.to(dtype=output_dtype)
        return AttributeResiduals(rotation, log_scale, alpha)


class ReferenceAttributeHead(BoundedAttributeResidualHead):
    """Stable semantic name for the quality-reference attribute head."""


__all__ = [
    "AttributeResiduals",
    "BoundedAttributeResidualHead",
    "ReferenceAttributeHead",
    "ResidualBounds",
    "context_from_geometry_prior",
]
