"""Signed expected utility; positive gain is distinct from physical safety."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SignedGainPrediction:
    """[R,2] components in stereo/Gaussian order, magnitudes in utility units."""

    positive_logits: torch.Tensor
    positive_gain: torch.Tensor
    negative_cost: torch.Tensor

    def __post_init__(self) -> None:
        values = (self.positive_logits, self.positive_gain, self.negative_cost)
        if (
            self.positive_logits.ndim != 2
            or self.positive_logits.shape[1] != 2
            or any(
                value.shape != self.positive_logits.shape
                or value.device != self.positive_logits.device
                or value.dtype != self.positive_logits.dtype
                or not value.is_floating_point()
                for value in values
            )
            or not bool(
                torch.stack([torch.isfinite(value).all() for value in values]).all()
                & (self.positive_gain >= 0).all()
                & (self.negative_cost >= 0).all()
            )
        ):
            raise ValueError(
                "signed gain components must share finite floating [R,2] and nonnegative magnitudes"
            )

    @property
    def positive_probability(self) -> torch.Tensor:
        return self.positive_logits.sigmoid()

    @property
    def expected_gain(self) -> torch.Tensor:
        probability = self.positive_probability
        result: torch.Tensor = (
            probability * self.positive_gain - (1 - probability) * self.negative_cost
        )
        return result

    def to_report(self) -> dict[str, list[list[float]]]:
        """Detached [R,3] diagnostics in STOP/stereo/Gaussian order; STOP is zero."""
        zero = torch.zeros_like(self.positive_gain[:, :1])
        return {
            name: torch.cat((zero, value), 1).detach().cpu().tolist()
            for name, value in (
                ("positive_probability", self.positive_probability),
                ("positive_gain", self.positive_gain),
                ("negative_cost", self.negative_cost),
                ("expected_gain", self.expected_gain),
            )
        }


def decode_signed_gain(raw: torch.Tensor, scale: torch.Tensor) -> SignedGainPrediction:
    """[R,2,3] logits/positive/negative -> nonnegative magnitudes, common scale.

    Softplus retains gradients for both magnitude heads. The original physical
    utility and its sign are unchanged; no action-specific preference multiplier.
    """
    if (
        raw.ndim != 3
        or raw.shape[1:] != (2, 3)
        or scale.numel() != 1
        or scale.device != raw.device
        or not bool(torch.isfinite(raw).all() & torch.isfinite(scale).all() & (scale > 0).all())
    ):
        raise ValueError("signed gain requires finite [R,2,3] and positive shared scale")
    return SignedGainPrediction(
        raw[..., 0], F.softplus(raw[..., 1]) * scale, F.softplus(raw[..., 2]) * scale
    )


def signed_gain_losses(
    prediction: SignedGainPrediction,
    targets: torch.Tensor,
    scale: torch.Tensor,
    row_weights: torch.Tensor | None = None,
    action_available: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-action [2] proper sign BCE and conditional magnitude mean-squared loss.

    P(U>0) uses the original row distribution, with no class-frequency multiplier.
    Magnitudes use only their own sign class and divide by that class's weight.
    A missing class has zero loss/gradient. U=0 belongs to the nonpositive class
    with magnitude zero. Unknown physical outcomes must be excluded by caller.
    This factorization estimates E[U|x], including negative costs, not oracle gain.
    """
    if (
        targets.shape != prediction.positive_gain.shape
        or targets.ndim != 2
        or targets.shape[1] != 2
        or targets.shape[0] == 0
        or targets.device != prediction.positive_gain.device
        or not bool(torch.isfinite(targets).all())
        or scale.numel() != 1
        or scale.device != targets.device
        or not bool(torch.isfinite(scale).all() & (scale > 0).all())
    ):
        raise ValueError("signed gain targets must be finite nonempty [R,2]")
    weights = targets.new_ones((targets.shape[0], 1))
    if row_weights is not None:
        if (
            row_weights.shape != targets.shape[:1]
            or row_weights.device != targets.device
            or not bool(torch.isfinite(row_weights).all() & (row_weights > 0).all())
        ):
            raise ValueError("signed gain row weights must be positive finite [R]")
        weights = row_weights[:, None].detach()
    if action_available is None:
        action_available = torch.ones_like(targets, dtype=torch.bool)
    if (
        action_available.shape != targets.shape
        or action_available.dtype != torch.bool
        or action_available.device != targets.device
    ):
        raise ValueError("signed gain action availability must be bool [R,2]")
    weights = weights * action_available.detach()
    positive = targets.detach() > 0
    sign = F.binary_cross_entropy_with_logits(
        prediction.positive_logits, positive.float(), reduction="none"
    )
    positive_error = ((prediction.positive_gain - targets.detach().clamp_min(0)) / scale).square()
    negative_error = (
        (prediction.negative_cost - (-targets.detach()).clamp_min(0)) / scale
    ).square()

    def conditional(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weighted_mask = weights * mask
        denominator = weighted_mask.sum(0)
        # Clamp only the missing-class denominator. Tiny valid weights retain
        # their exact conditional mean; multiplying all weights changes nothing.
        return (error * weighted_mask).sum(0) / torch.where(
            denominator > 0, denominator, torch.ones_like(denominator)
        )

    return {
        "probability": (sign * weights).sum(0) / weights.sum(0).clamp_min(1e-12),
        "positive": conditional(positive_error, positive),
        "negative": conditional(negative_error, ~positive),
        "positive_count": (positive & action_available).sum(0),
        "negative_count": ((~positive) & action_available).sum(0),
    }
