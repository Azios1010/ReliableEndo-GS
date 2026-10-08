"""Separated gain/harm MLPs and bounded balancing on actual shared parameters."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.routing.context import AVAILABILITY_COLUMNS, BALANCED_FEATURE_NAMES
from reliable_endo_gs.routing.normalization import TrainingNormalization
from reliable_endo_gs.routing.repairability import RepairabilitySelector


class BalancedSelector(RepairabilitySelector):
    """Global/local utility and harm [R,3], independently encoded.

    Harm has no shared parameters with gain, so BCE cannot dominate gain's
    encoder. Gain/local/ranking use bounded detached EMA gradient balancing.
    """

    def __init__(self, width: int, config: InterventionConfig) -> None:
        if config.schema != "intervention_selection.v2":
            raise ValueError("balanced selector requires the v2 protocol")
        super().__init__(width)
        del self.network
        self.config = config
        self.normalization = TrainingNormalization(
            len(BALANCED_FEATURE_NAMES), AVAILABILITY_COLUMNS, config
        )

        def network(outputs: int) -> nn.Sequential:
            net = nn.Sequential(
                nn.Linear(len(BALANCED_FEATURE_NAMES), width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, outputs),
            )
            output = net[-1]
            assert isinstance(output, nn.Linear) and output.bias is not None
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
            return net

        self.gain_network = network(4)
        self.harm_network = network(2)

    def outputs(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.normalization(features.detach())
        gain = self.gain_network(x)
        harm = self.harm_network(x)
        zero = torch.zeros_like(gain[:, :1])
        global_gain = torch.cat((zero, gain[:, :2] * self.normalization.gain_scales[0]), 1)
        local_gain = torch.cat((zero, gain[:, 2:] * self.normalization.gain_scales[1]), 1)
        return global_gain, local_gain, torch.cat((zero, harm.sigmoid()), 1)

    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_gain, _, harm = self.outputs(features)
        return global_gain, harm

    def loss(
        self,
        features: torch.Tensor,
        gains: torch.Tensor,
        harms: torch.Tensor,
        config: InterventionConfig,
    ) -> torch.Tensor:
        raise ValueError("balanced selector requires explicit local/global component targets")

    def objective(
        self,
        features: torch.Tensor,
        global_target: torch.Tensor,
        local_target: torch.Tensor,
        harms: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        """Return component losses; the training layer owns gradient balancing."""
        global_prediction, local_prediction, risk = self.outputs(features)
        if (
            global_target.shape != global_prediction.shape
            or local_target.shape != global_target.shape
            or harms.shape != global_target.shape
        ):
            raise ValueError("balanced supervision must share [R,3]")
        if not bool(torch.isfinite(global_target).all() & torch.isfinite(local_target).all()):
            raise ValueError("balanced gain targets must be finite")
        if not bool((global_target[:, 0] == 0).all() & (local_target[:, 0] == 0).all()):
            raise ValueError("STOP gain targets must be exactly zero")
        if not bool(torch.isfinite(harms).all()):
            raise ValueError("balanced harm targets must be finite")
        if not bool(((harms >= 0) & (harms <= 1)).all()):
            raise ValueError("balanced harm targets must be binary probabilities")
        if not bool((harms[:, 0] == 0).all()):
            raise ValueError("STOP harm target must be exactly zero")
        global_scale, local_scale = self.normalization.gain_scales.unbind()
        g = global_prediction / global_scale
        local_normalized = local_prediction / local_scale
        t = global_target.detach() / global_scale
        global_loss = F.smooth_l1_loss(g[:, 1:], t[:, 1:])
        local_loss = F.smooth_l1_loss(
            local_normalized[:, 1:], local_target.detach()[:, 1:] / local_scale
        )
        # Ranking units are explicit in the config.  Raw units preserve the
        # historical objective; train_scale compares labels after the common
        # TRAIN-fitted gain transform.  This avoids changing old checkpoint
        # semantics while making the remediation experiment deliberate.
        rank_terms = []
        informative_pairs = []
        for a, b in ((0, 1), (0, 2), (1, 2)):
            if self.config.ranking_margin_units == "train_scale":
                margin_values = t[:, a] - t[:, b]
                margin_scale = global_scale.new_tensor(1.0)
            else:
                margin_values = global_target.detach()[:, a] - global_target.detach()[:, b]
                margin_scale = global_scale
            informative = margin_values.abs() > (
                self.config.ranking_margin
                if self.config.ranking_margin_units == "raw"
                else self.config.ranking_margin
            )
            # Bounded normalized weights keep a few very large proxy labels
            # from dominating the pair objective while retaining their order.
            normalized_margin = margin_values / margin_scale
            weight = normalized_margin.abs().clamp(max=5)
            error = (
                (
                    self.config.ranking_margin / margin_scale
                    if self.config.ranking_margin_units == "raw"
                    else self.config.ranking_margin
                )
                - normalized_margin.sign() * (g[:, a] - g[:, b])
            ).clamp_min(0)
            rank_terms.append((error * weight * informative).mean())
            informative_pairs.append(informative.float().mean())
        ranking = torch.stack(rank_terms).mean()
        # Use logits directly rather than probability BCE for numerical stability.
        logits = self.harm_network(self.normalization(features.detach()))
        harm_loss = F.binary_cross_entropy_with_logits(logits, harms.detach()[:, 1:].float())
        diagnostics = {
            "selector_global_loss": float(global_loss.detach()),
            "selector_local_loss": float(local_loss.detach()),
            "selector_ranking_loss": float(ranking.detach()),
            "selector_harm_loss": float(harm_loss.detach()),
            "selector_ranking_informative_fraction": float(
                torch.stack(informative_pairs).mean().detach()
            ),
            "selector_ranking_margin_units": self.config.ranking_margin_units,
            "selector_ranking_margin": float(self.config.ranking_margin),
            "selector_harm_positive_fraction": float(harms[:, 1:].detach().mean()),
            # Local gain is deliberately an auxiliary target.  Global gain is
            # the sole decision utility; local supervision shapes the shared
            # encoder without counting the same render reward twice.
            "selector_local_auxiliary_only": 1.0,
        }
        diagnostics.update(
            gain_scale_global=float(global_scale),
            gain_scale_local=float(local_scale),
            normalization_fitted=float(self.normalization.fitted),
            feature_clip_fraction=float(
                ((features - self.normalization.center) / self.normalization.scale)
                .abs()
                .gt(self.config.feature_clip)
                .float()
                .mean()
            )
            if bool(self.normalization.fitted)
            else 0.0,
        )
        return {
            "global": global_loss,
            "local": local_loss,
            "ranking": ranking,
            "harm": harm_loss,
        }, diagnostics

    def validate_state(self) -> None:
        self.normalization.validate_state()
