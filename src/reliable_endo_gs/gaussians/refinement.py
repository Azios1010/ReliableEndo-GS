"""Trainable per-primitive Gaussian field refinement (configurable successor of the Stage A Gaussian repair head).

Same pointwise structure as :class:`BoundedAttributeResidualHead` (GELU MLP, zero-initialised output layer so the
initial model is the identity, tanh-bounded residuals, identical ``body`` / ``output`` parameter names so Stage A
head weights load directly) with the knobs a capacity study needs: hidden widths (including none), which of the
13 context channels are visible, an optional bounded colour residual and configurable bounds. The 13 channels are
``[rgb(3), mu/scale(3), normal(3), S0/scale(3), alpha0(1)]`` (see ``context_from_geometry_prior``): purely local to
one primitive, no neighbours, no image or student features.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from reliable_endo_gs.gaussians.residual_head import AttributeResiduals, ResidualBounds

CONTEXT_CHANNEL_NAMES: tuple[str, ...] = (
    "rgb_r", "rgb_g", "rgb_b",
    "mu_x", "mu_y", "mu_z",
    "normal_x", "normal_y", "normal_z",
    "scale_a", "scale_b", "scale_c",
    "alpha0",
)  # fmt: skip
RESIDUAL_GROUPS: tuple[str, ...] = ("rotation", "scale", "opacity")
ALL_CONTEXT_CHANNELS: tuple[int, ...] = tuple(range(len(CONTEXT_CHANNEL_NAMES)))
# Absolute image position enters only through mu_x / mu_y; mu_z (depth) is kept.
NO_ABSOLUTE_XY_CHANNELS: tuple[int, ...] = tuple(i for i in ALL_CONTEXT_CHANNELS if i not in (3, 4))


@dataclass(frozen=True, slots=True)
class FieldRefinerConfig:
    """Architecture of the refiner; ``input_channels=()`` is a learned constant per output."""

    hidden_dims: tuple[int, ...] = (32, 32)
    input_channels: tuple[int, ...] = ALL_CONTEXT_CHANNELS
    color_residual: bool = False
    max_color_delta: float = 0.1
    bounds: ResidualBounds = field(default_factory=ResidualBounds)
    # Residual groups that may be non-zero; a group left out is multiplied by zero after its bound, so its
    # output rows receive no gradient (they stay at their zero initialisation).
    active_outputs: tuple[str, ...] = RESIDUAL_GROUPS

    def __post_init__(self) -> None:
        if (
            not self.active_outputs
            or len(set(self.active_outputs)) != len(self.active_outputs)
            or any(group not in RESIDUAL_GROUPS for group in self.active_outputs)
        ):
            raise ValueError(f"active_outputs must be a non-empty subset of {RESIDUAL_GROUPS}")
        if any(type(w) is not int or w < 1 for w in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        if any(
            type(c) is not int or not 0 <= c < len(CONTEXT_CHANNEL_NAMES)
            for c in self.input_channels
        ):
            raise ValueError("input_channels must index the 13 context channels")
        if len(set(self.input_channels)) != len(self.input_channels):
            raise ValueError("input_channels must be unique")
        if not 0 < self.max_color_delta <= 1:
            raise ValueError("max_color_delta must lie in (0, 1]")

    def to_dict(self) -> dict[str, object]:
        values: dict[str, object] = {
            "hidden_dims": list(self.hidden_dims),
            "input_channels": [CONTEXT_CHANNEL_NAMES[c] for c in self.input_channels],
            "color_residual": self.color_residual,
            "max_color_delta": self.max_color_delta,
            "bounds": self.bounds.to_dict(),
        }
        # The neutral default is omitted so that historical identities stay valid.
        if self.active_outputs != RESIDUAL_GROUPS:
            values["active_outputs"] = list(self.active_outputs)
        return values


@dataclass(frozen=True, slots=True)
class FieldResiduals:
    """Bounded residuals; ``delta_color`` [B,N,3] is None unless the colour residual is enabled."""

    attributes: AttributeResiduals
    delta_color: torch.Tensor | None


class FieldRefiner(nn.Module):
    """Context [B,N,13] -> bounded rotation [B,N,3], log-scale [B,N,3], alpha-logit [B,N,1] (+ colour [B,N,3])."""

    def __init__(self, config: FieldRefinerConfig | None = None) -> None:
        super().__init__()
        self.config = config or FieldRefinerConfig()
        self.register_buffer(
            "_channels",
            torch.tensor(self.config.input_channels, dtype=torch.long),
            persistent=False,
        )
        in_features = max(len(self.config.input_channels), 1)  # a constant refiner sees a single 1
        layers: list[nn.Module] = []
        for width in self.config.hidden_dims:
            layers.extend((nn.Linear(in_features, width), nn.GELU()))
            in_features = width
        self.body = nn.Sequential(*layers)
        self.output = nn.Linear(in_features, 7)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.color_output: nn.Linear | None = None
        if self.config.color_residual:
            self.color_output = nn.Linear(in_features, 3)
            nn.init.zeros_(self.color_output.weight)
            nn.init.zeros_(self.color_output.bias)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, context: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> FieldResiduals:
        if context.ndim != 3 or context.shape[-1] != len(CONTEXT_CHANNEL_NAMES):
            raise ValueError(f"context must have shape [B,N,13]; got {tuple(context.shape)}")
        if not torch.is_floating_point(context):
            raise TypeError("context must be floating point")
        output_dtype = context.dtype
        parameter = next(self.parameters())
        if parameter.dtype != context.dtype:
            context = context.to(dtype=parameter.dtype)
        if valid_mask is not None:
            if valid_mask.shape != context.shape[:2] or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be bool [B,N]")
            context = torch.where(valid_mask.unsqueeze(-1), context, torch.zeros_like(context))
        features = (
            context[..., self._channels]
            if self.config.input_channels
            else torch.ones_like(context[..., :1])
        )
        hidden = self.body(features)
        raw = self.output(hidden)
        bounds = self.config.bounds
        rotation = bounds.max_rotation_rad * torch.tanh(raw[..., 0:3] / bounds.max_rotation_rad)
        log_scale = bounds.max_log_scale * torch.tanh(raw[..., 3:6] / bounds.max_log_scale)
        alpha = bounds.max_alpha_logit * torch.tanh(raw[..., 6:7] / bounds.max_alpha_logit)
        if len(self.config.active_outputs) != len(RESIDUAL_GROUPS):
            groups = self.config.active_outputs
            rotation = rotation if "rotation" in groups else torch.zeros_like(rotation)
            log_scale = log_scale if "scale" in groups else torch.zeros_like(log_scale)
            alpha = alpha if "opacity" in groups else torch.zeros_like(alpha)
        delta_color = None
        if self.color_output is not None:
            limit = self.config.max_color_delta
            delta_color = (limit * torch.tanh(self.color_output(hidden) / limit)).to(
                dtype=output_dtype
            )
            if valid_mask is not None:
                delta_color = torch.where(
                    valid_mask.unsqueeze(-1), delta_color, torch.zeros_like(delta_color)
                )
        return FieldResiduals(
            AttributeResiduals(
                rotation.to(output_dtype), log_scale.to(output_dtype), alpha.to(output_dtype)
            ),
            delta_color,
        )


def channel_names(channels: Sequence[int]) -> tuple[str, ...]:
    """Human-readable names for a channel selection."""
    return tuple(CONTEXT_CHANNEL_NAMES[c] for c in channels)
