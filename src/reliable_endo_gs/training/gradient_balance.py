"""Detached bounded weights measured on declared actual shared parameters."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from reliable_endo_gs.config.intervention import InterventionConfig


class GradientBalance(nn.Module):
    """No trainable parameters. State commits only after successful optimizer step."""

    ema: torch.Tensor
    weights: torch.Tensor
    observations: torch.Tensor

    def __init__(self, names: tuple[str, ...], config: InterventionConfig) -> None:
        super().__init__()
        self.names = names
        self.config = config
        self.register_buffer("ema", torch.zeros(len(names)))
        self.register_buffer("weights", torch.ones(len(names)))
        self.register_buffer("observations", torch.tensor(0, dtype=torch.long))

    def measure(
        self, losses: Sequence[torch.Tensor], shared: Sequence[nn.Parameter]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Probe loss gradients before clipping; not an optimizer update."""
        if len(losses) != len(self.names) or not shared:
            raise ValueError("gradient balance ownership mismatch")
        gradients = []
        for loss in losses:
            grads = (
                torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
                if loss.requires_grad
                else [None] * len(shared)
            )
            gradients.append(
                torch.cat(
                    [
                        torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                        for p, g in zip(shared, grads, strict=True)
                    ]
                )
            )
        norms = torch.stack([g.norm() for g in gradients])
        if not bool(torch.isfinite(norms).all()):
            raise ValueError("nonfinite gradient balancing observation")
        proposed = torch.where(
            self.ema > 0,
            self.config.gradient_ema_decay * self.ema
            + (1 - self.config.gradient_ema_decay) * norms,
            norms,
        )
        active = (proposed > self.config.normalization_floor) & (
            norms > self.config.normalization_floor
        )
        reference = proposed[active].median() if bool(active.any()) else proposed.new_tensor(1.0)
        weights = (
            torch.where(
                active,
                reference / proposed.clamp_min(self.config.normalization_floor),
                torch.ones_like(proposed),
            )
            .clamp(self.config.balance_minimum, self.config.balance_maximum)
            .detach()
        )
        diag = {f"shared_grad_{name}": float(norms[i]) for i, name in enumerate(self.names)}
        diag.update({f"balance_{name}": float(weights[i]) for i, name in enumerate(self.names)})
        for a in range(len(self.names)):
            for b in range(a + 1, len(self.names)):
                denominator = norms[a] * norms[b]
                diag[f"grad_cosine_{self.names[a]}_{self.names[b]}"] = (
                    float((gradients[a] * gradients[b]).sum() / denominator)
                    if float(denominator) > 1e-12
                    else 0.0
                )
        return torch.stack((proposed, weights)), diag

    @torch.no_grad()
    def commit(self, pending: torch.Tensor) -> None:
        self.ema.copy_(pending[0])
        self.weights.copy_(pending[1])
        self.observations.add_(1)

    def validate_state(self) -> None:
        if (
            bool((self.ema < 0).any())
            or int(self.observations) < 0
            or not bool(
                (
                    (self.weights >= self.config.balance_minimum)
                    & (self.weights <= self.config.balance_maximum)
                ).all()
            )
        ):
            raise ValueError("invalid gradient balancing state")

    def weighted_loss(
        self,
        losses: Sequence[torch.Tensor],
        priorities: torch.Tensor,
        pending: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply explicit [N] priorities AFTER measured balancing coefficients.

        Priorities never enter the norm/EMA probe and cannot be inversely
        absorbed by it. Subsequent optimizer clipping may still bound updates.
        """
        if len(losses) != len(self.names) or priorities.shape != self.weights.shape:
            raise ValueError("objective priority shape mismatch")
        if not bool(torch.isfinite(priorities).all() & (priorities >= 0).all()):
            raise ValueError("objective priorities must be finite and nonnegative")
        if pending is not None and pending.shape != (2, len(self.names)):
            raise ValueError("pending gradient balance shape mismatch")
        weights = self.weights if pending is None else pending[1]
        return (weights.detach() * priorities.detach() * torch.stack(tuple(losses))).sum()
