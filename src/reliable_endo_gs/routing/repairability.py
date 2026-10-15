"""Deployment-only evidence, gain/harm prediction and single-action allocation."""

from __future__ import annotations

import math

import torch
from torch import nn

from reliable_endo_gs.actions.repair import Action, RepairState, primitive_core
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.geometry.stereo_matching import correspondence_error
from reliable_endo_gs.regions.grid import Region
from reliable_endo_gs.routing.selector import FEATURE_NAMES, ActionSelector, Decision

REPAIRABILITY_FEATURE_NAMES = FEATURE_NAMES + (
    "warp_error_mean",
    "warp_error_std",
    "warp_valid_fraction",
    "texture_std",
    "local_matching_margin",
    "local_matching_entropy_proxy",
    "disparity_relative_jump",
    "matching_candidate_support_fraction",
)


@torch.no_grad()
def repairability_features(
    state: RepairState,
    regions: tuple[Region, ...],
    config: InterventionConfig,
    *,
    scale_unit_m: float,
) -> torch.Tensor:
    """Return detached [R,26] evidence; no teacher/GT/action outcomes.

    Reductions are batched across regions to avoid per-feature GPU host sync.
    Entropy uses three RGB warp candidates; it is not calibrated sigma_d.
    """
    if state.base.render_right is None or not regions or scale_unit_m <= 0:
        raise ValueError("two renders, regions and positive scale normalization required")
    height, width = state.batch.left.shape[-2:]
    masks = (
        torch.cat([r.mask(state.batch.left) & state.support for r in regions], 0).flatten(1).float()
    )
    count = masks.sum(1).clamp_min(1)

    def mean(image: torch.Tensor) -> torch.Tensor:
        return masks @ image[0].flatten(1).float().T / count[:, None]

    raw_disparity = state.base.target.disparity
    d = raw_disparity.float() / width
    dm = mean(d)
    ds = (mean(d.square()) - dm.square()).clamp_min(0).sqrt()
    core_area = masks.new_tensor([r.pixel_count for r in regions])
    primitive_masks = torch.stack([primitive_core(state, r)[0] for r in regions]).float()
    pc = primitive_masks.sum(1).clamp_min(1)
    opacity = (
        primitive_masks @ state.base.gaussians.opacities[0].reshape(-1, 1).float() / pc[:, None]
    )
    scale = primitive_masks @ state.base.gaussians.scales[0].float() / pc[:, None] / scale_unit_m
    xy = masks.new_tensor(
        [
            [(r.core[1] + r.core[3]) / (2 * width), (r.core[0] + r.core[2]) / (2 * height)]
            for r in regions
        ]
    )
    errors, validities = [], []
    for offset in (-config.search_step_px, 0, config.search_step_px):
        e, v = correspondence_error(
            state.batch.left,
            state.batch.right,
            raw_disparity + offset,
            right_support=state.batch.support,
        )
        errors.append(e)
        validities.append(v)
    candidates = torch.cat(errors, 1)
    cv = torch.cat(validities, 1)
    scores = (-candidates / config.matching_temperature).masked_fill(~cv, -1e4)
    probabilities = scores.softmax(1) * cv.to(scores.dtype)
    probabilities = probabilities / probabilities.sum(1, keepdim=True).clamp_min(1e-6)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
        1, keepdim=True
    ) / math.log(3)
    evidence_available = cv.any(1, keepdim=True)
    # Entropy zero is the algebraic result of an all-invalid row, not a
    # confident match.  Keep the fixed feature schema and expose the state as
    # maximal ambiguity; the candidate-support feature remains zero.
    entropy = torch.where(evidence_available, entropy, torch.ones_like(entropy))
    ordered = candidates.sort(1).values
    margin = (ordered[:, 1:2] - ordered[:, :1]) / config.stereo_scale
    error = errors[1]
    em = mean(error)
    es = (mean(error.square()) - em.square()).clamp_min(0).sqrt()
    gray = state.batch.left.mean(1, keepdim=True)
    texture = (mean(gray.square()) - mean(gray).square()).clamp_min(0).sqrt()
    jump = torch.zeros_like(raw_disparity)
    jump[..., 1:] = (
        (raw_disparity[..., 1:] - raw_disparity[..., :-1]).abs()
        / raw_disparity[..., 1:].abs().clamp_min(1e-6)
    ).clamp(max=1)
    legacy = torch.cat(
        (
            mean(state.batch.left),
            mean((state.base.render_left.image - state.batch.left).abs()),
            mean((state.base.render_right.image - state.batch.right).abs()),
            dm,
            ds,
            (masks.sum(1) / core_area)[:, None],
            opacity,
            scale,
            xy,
        ),
        1,
    )
    extra = torch.cat(
        (
            em / config.stereo_scale,
            es / config.stereo_scale,
            mean(validities[1].float()),
            texture,
            mean(margin),
            mean(entropy),
            mean(jump),
            mean(cv.float().mean(1, keepdim=True)),
        ),
        1,
    )
    return torch.cat((legacy, extra), 1).detach()


class RepairabilitySelector(ActionSelector):
    """Small MLP: [R,26] -> gains and harm logits for two repair actions."""

    def __init__(self, width: int = 32) -> None:
        super().__init__(width)
        self.network = nn.Sequential(
            nn.Linear(len(REPAIRABILITY_FEATURE_NAMES), width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 4),
        )
        output = self.network[-1]
        if not isinstance(output, nn.Linear) or output.bias is None:
            raise TypeError("selector output must have bias")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return gain and harm probability [R,3]; STOP is exactly zero."""
        if (
            features.ndim != 2
            or features.shape[1] != len(REPAIRABILITY_FEATURE_NAMES)
            or not bool(torch.isfinite(features).all())
        ):
            raise ValueError("repairability features must be finite [R,26]")
        raw = self.network(features.float())
        zero = torch.zeros_like(raw[:, :1])
        return torch.cat((zero, raw[:, :2]), 1), torch.cat((zero, raw[:, 2:].sigmoid()), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.predict(features)[0]

    def loss(
        self,
        features: torch.Tensor,
        gains: torch.Tensor,
        harms: torch.Tensor,
        config: InterventionConfig,
        *,
        action_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Detached counterfactual targets; ranking weights measured gain margins."""
        if gains.shape != (features.shape[0], 3) or harms.shape != gains.shape:
            raise ValueError("selector targets must be [R,3]")
        if action_available is None:
            action_available = torch.ones_like(gains, dtype=torch.bool)
        if action_available.shape != gains.shape or action_available.dtype != torch.bool:
            raise ValueError("selector action availability must be bool [R,3]")
        raw = self.network(features.float())
        prediction = torch.cat((torch.zeros_like(raw[:, :1]), raw[:, :2]), 1)
        gain_target = gains.detach()
        available = action_available[:, 1:]
        regression = (
            nn.functional.smooth_l1_loss(prediction[:, 1:], gain_target[:, 1:], reduction="none")
            * available
        ).sum() / available.sum().clamp_min(1)
        rank_terms = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            margin = gain_target[:, i] - gain_target[:, j]
            pair_available = action_available[:, i] & action_available[:, j]
            informative = (margin.abs() > config.ranking_margin) & pair_available
            error = (
                config.ranking_margin - margin.sign() * (prediction[:, i] - prediction[:, j])
            ).clamp_min(0)
            rank_terms.append(
                (error * margin.abs() * informative).sum() / pair_available.sum().clamp_min(1)
            )
        ranking = torch.stack(rank_terms).mean()
        harm = (
            nn.functional.binary_cross_entropy_with_logits(
                raw[:, 2:], harms.detach()[:, 1:].float(), reduction="none"
            )
            * available
        ).sum() / available.sum().clamp_min(1)
        return regression + config.ranking_weight * ranking + config.harm_weight * harm


def select_repairability(
    gains: torch.Tensor,
    harms: torch.Tensor,
    costs_ms: torch.Tensor,
    valid: torch.Tensor,
    remaining_ms: float,
    config: InterventionConfig,
    *,
    risk_available: torch.Tensor | None = None,
) -> Decision:
    """Exact highest gain among feasible safe single interventions; else STOP.

    Cost appears once, as a budget constraint. An estimate is not a guarantee
    of actual runtime; overruns must be reported by evaluation.
    """
    if (
        gains.ndim != 2
        or gains.shape[1] != 3
        or harms.shape != gains.shape
        or costs_ms.shape != gains.shape
    ):
        raise ValueError("gain/harm/cost must share [R,3]")
    if valid.dtype != torch.bool or valid.shape not in (gains.shape[:1], gains.shape):
        raise ValueError("eligibility must be bool [R] or action-specific [R,3]")
    if risk_available is None:
        risk_available = torch.ones_like(harms, dtype=torch.bool)
    if risk_available.shape != harms.shape or risk_available.dtype != torch.bool:
        raise ValueError("risk_available must be bool [R,3]")
    if not math.isfinite(remaining_ms) or remaining_ms < 0:
        raise ValueError("remaining budget must be finite and nonnegative")
    values_valid = (
        torch.isfinite(gains).all() & torch.isfinite(harms).all() & torch.isfinite(costs_ms).all()
    ) & (((harms >= 0) & (harms <= 1)).all() & (costs_ms >= 0).all() & (gains[:, 0] == 0).all())
    action_valid = valid[:, None] if valid.ndim == 1 else valid
    cutoffs = harms.new_tensor(
        [
            0.0,
            config.maximum_harm_probability
            if config.maximum_stereo_harm_probability is None
            else config.maximum_stereo_harm_probability,
            config.maximum_harm_probability
            if config.maximum_gaussian_harm_probability is None
            else config.maximum_gaussian_harm_probability,
        ]
    )
    allowed = (
        action_valid
        & (costs_ms > 0)
        & (costs_ms <= remaining_ms)
        & risk_available
        & (harms <= cutoffs)
        & (gains > config.minimum_gain)
    )
    allowed[:, 0] = False
    # Cost remains a hard budget constraint.  An optional measured-latency
    # penalty also participates in the ranking so an expensive action cannot
    # win solely because its raw gain is marginally larger.  The default zero
    # keeps historical gain-only semantics for legacy configurations.
    utility = gains - config.cost_weight_per_ms * costs_ms
    scores = torch.cat((gains.new_zeros(1), torch.where(allowed, utility, -torch.inf).flatten()))
    winner = scores.argmax()
    selected_gain = torch.cat((gains.new_zeros(1), gains.flatten()))[winner]
    selected_cost = torch.cat((costs_ms.new_zeros(1), costs_ms.flatten()))[winner]
    status, encoded, gain, cost = (
        torch.stack(
            (values_valid.double(), winner.double(), selected_gain.double(), selected_cost.double())
        )
        .detach()
        .cpu()
        .tolist()
    )
    if not status:
        raise ValueError("invalid gain/harm/cost values")
    if encoded == 0:
        return Decision(None, Action.STOP, 0.0, 0.0)
    index = int(encoded) - 1
    row, action = divmod(index, 3)
    return Decision(row, Action(action), gain, cost)
