"""Separated gain/harm MLPs and bounded balancing on actual shared parameters."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.routing.context import AVAILABILITY_COLUMNS, BALANCED_FEATURE_NAMES
from reliable_endo_gs.routing.gain import (
    SignedGainPrediction,
    decode_signed_gain,
    signed_gain_losses,
)
from reliable_endo_gs.routing.normalization import TrainingNormalization
from reliable_endo_gs.routing.repairability import RepairabilitySelector
from reliable_endo_gs.routing.tokens import selector_feature_names


def _network(
    width: int, outputs: int, feature_count: int = len(BALANCED_FEATURE_NAMES)
) -> nn.Sequential:
    net = nn.Sequential(
        nn.Linear(feature_count, width),
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


class _ActionNetworks(nn.Module):
    """Independent action MLPs: [R,F] -> [R,2*outputs], component-major order."""

    def __init__(
        self, width: int, outputs: int, feature_count: int = len(BALANCED_FEATURE_NAMES)
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleDict(
            {action: _network(width, outputs, feature_count) for action in ("stereo", "gaussian")}
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Gain packing is global_ST, global_GS, local_ST, local_GS;
        # risk packing is ST, GS. STOP has no parameters in either network.
        values = [branch(features) for branch in self.branches.values()]
        return torch.stack(values, dim=-1).flatten(1)


class BalancedSelector(RepairabilitySelector):
    """Global/local utility and harm [R,3], independently encoded.

    Harm has no shared parameters with gain. The separate_action_heads topology
    also shares no trainable parameters between actions; numerical normalization
    and the utility scale stay common. Gain/local/ranking use EMA balancing.
    """

    def __init__(self, width: int, config: InterventionConfig) -> None:
        if config.schema != "intervention_selection.v2":
            raise ValueError("balanced selector requires the v2 protocol")
        super().__init__(width)
        del self.network
        self.config = config
        feature_count = len(selector_feature_names(config))
        self.normalization = TrainingNormalization(feature_count, AVAILABILITY_COLUMNS, config)

        self.gain_network: nn.Module
        self.harm_network: nn.Module
        if config.selector_architecture == "separate_action_heads":
            self.gain_network = _ActionNetworks(width, 2, feature_count)
            if config.selector_gain_mode == "signed_expectation":
                # Keep the matched gain trunks, risk initialization and caller
                # RNG stream. Only the zero-initialized output readouts expand.
                with torch.random.fork_rng(devices=[]):
                    for branch in self.gain_network.branches.values():
                        assert isinstance(branch, nn.Sequential)
                        output = nn.Linear(width, 4)
                        assert output.bias is not None
                        nn.init.zeros_(output.weight)
                        nn.init.zeros_(output.bias)
                        branch[-1] = output
            self.harm_network = _ActionNetworks(
                width, 3 if config.selector_risk_mode == "local_global" else 1, feature_count
            )
        else:
            self.gain_network = _network(width, 4)
            self.harm_network = _network(width, 2)

    def optimization_networks(self) -> tuple[nn.Sequential, ...]:
        """MLPs clipped independently; no action consumes another's norm budget."""
        networks: list[nn.Sequential] = []
        for network in (self.gain_network, self.harm_network):
            if isinstance(network, _ActionNetworks):
                for branch in network.branches.values():
                    assert isinstance(branch, nn.Sequential)
                    networks.append(branch)
            else:
                assert isinstance(network, nn.Sequential)
                networks.append(network)
        return tuple(networks)

    def optimization_components(self) -> tuple[tuple[str, nn.Sequential], ...]:
        """Stable per-network diagnostic names; gain/risk ownership stays separate."""
        names = (
            ("stereo_gain", "gaussian_gain", "stereo_risk", "gaussian_risk")
            if self.config.selector_architecture == "separate_action_heads"
            else ("gain", "risk")
        )
        return tuple(zip(names, self.optimization_networks(), strict=True))

    def gain_shared_parameters(self) -> tuple[nn.Parameter, ...]:
        """Both gain trunks shared by global/local/ranking, excluding outputs."""
        network = self.gain_network
        branches = (
            tuple(network.branches.values()) if isinstance(network, _ActionNetworks) else (network,)
        )
        return tuple(
            p
            for branch in branches
            for layer in list(branch.children())[:-1]
            for p in layer.parameters()
        )

    def utility_and_risk_logits(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Global/local utility [R,3] and harm logits [R,2], one encode per head.

        Consumers needing BCE or calibrated probabilities reuse these logits;
        they must not run normalization and the risk network a second time.
        """
        global_gain, local_gain, logits = self._encoded_outputs(features)
        return (
            global_gain,
            local_gain,
            logits[:, :, 2] if self.config.selector_risk_mode == "local_global" else logits,
        )

    def _encoded_outputs(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Historical output interface; extra gain components are explicit elsewhere."""
        global_gain, local_gain, harm, _ = self._encoded_outputs_with_gain(features)
        return global_gain, local_gain, harm

    def _encoded_outputs_with_gain(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SignedGainPrediction | None]:
        """Single encoding; grouped risk logits [R,2,3] in local/global/any order.

        Union probability lies within Frechet bounds max(pL,pG)..min(1,pL+pG).
        A learned third scalar locates it within those bounds. This learns
        dependence instead of assuming independence; supervision is the real OR.
        """
        x = self.normalization(features.detach())
        gain = self.gain_network(x)
        harm = self.harm_network(x)
        if self.config.selector_risk_mode == "local_global":
            raw = harm.reshape(-1, 3, 2).transpose(1, 2)
            probability = raw[:, :, :2].sigmoid()
            lower = probability.amax(-1)
            upper = probability.sum(-1).clamp(max=1)
            union = lower + (upper - lower) * raw[:, :, 2].sigmoid()
            epsilon = torch.finfo(union.dtype).eps
            harm = torch.cat(
                (raw[:, :, :2], torch.logit(union.clamp(epsilon, 1 - epsilon))[..., None]), -1
            )
        zero = torch.zeros_like(gain[:, :1])
        components = None
        if self.config.selector_gain_mode == "signed_expectation":
            # Component-major packing: sign ST/GS, positive ST/GS,
            # negative ST/GS, local auxiliary ST/GS.
            raw = gain.reshape(-1, 4, 2).transpose(1, 2)
            components = decode_signed_gain(raw[:, :, :3], self.normalization.gain_scales[0])
            global_gain = torch.cat((zero, components.expected_gain), 1)
            local_gain = torch.cat((zero, raw[:, :, 3] * self.normalization.gain_scales[1]), 1)
        else:
            global_gain = torch.cat((zero, gain[:, :2] * self.normalization.gain_scales[0]), 1)
            local_gain = torch.cat((zero, gain[:, 2:] * self.normalization.gain_scales[1]), 1)
        return global_gain, local_gain, harm, components

    def risk_event_outputs(self, features: torch.Tensor) -> torch.Tensor:
        """Explicit probabilities [R,3,3]: STOP/ST/GS x local/global/any."""
        if self.config.selector_risk_mode != "local_global":
            raise ValueError("historical scalar risk has no local/global predictions")
        events = self.outputs_with_events(features)[3]
        assert events is not None
        return events

    def outputs_with_events(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """One encode for decision and diagnostics; events [R,3,3] or legacy None."""
        global_gain, local_gain, harm, events, _ = self.outputs_with_components(features)
        return global_gain, local_gain, harm, events

    def outputs_with_components(
        self, features: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, SignedGainPrediction | None
    ]:
        """One encode for decision gains, risks and optional sign/magnitude outputs."""
        global_gain, local_gain, logits, components = self._encoded_outputs_with_gain(features)
        if self.config.selector_risk_mode == "local_global":
            events = torch.cat((logits.new_zeros((logits.shape[0], 1, 3)), logits.sigmoid()), 1)
            return global_gain, local_gain, events[:, :, 2], events, components
        return (
            global_gain,
            local_gain,
            torch.cat((global_gain[:, :1], logits.sigmoid()), 1),
            None,
            components,
        )

    def outputs(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_gain, local_gain, harm, _ = self.outputs_with_events(features)
        return global_gain, local_gain, harm

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
        *,
        row_weights: torch.Tensor | None = None,
        risk_components: torch.Tensor | None = None,
        action_available: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        """Return component losses; the training layer owns gradient balancing."""
        global_prediction, local_prediction, event_logits, gain_components = (
            self._encoded_outputs_with_gain(features)
        )
        grouped = self.config.selector_risk_mode == "local_global"
        logits = event_logits[:, :, 2] if grouped else event_logits
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
        if row_weights is not None and (
            row_weights.shape != (features.shape[0],)
            or row_weights.device != features.device
            or not bool(torch.isfinite(row_weights).all() & (row_weights > 0).all())
        ):
            raise ValueError("selector row weights must be positive finite [R]")

        if action_available is None:
            action_available = torch.ones_like(global_target, dtype=torch.bool)
        if (
            action_available.shape != global_target.shape
            or action_available.dtype != torch.bool
            or action_available.device != features.device
            or not bool(action_available[:, 0].all())
        ):
            raise ValueError("selector action availability must be bool [R,3] with STOP available")
        available = action_available[:, 1:]

        def reduce_rows(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            if mask is None:
                mask = torch.ones_like(values, dtype=torch.bool)
            if mask.shape != values.shape:
                if values.ndim == 2 and mask.shape == (values.shape[0],):
                    mask = mask[:, None].expand_as(values)
                else:
                    raise ValueError("selector reduction mask shape mismatch")
            weights = mask.to(values.dtype)
            if row_weights is not None:
                weights = weights * row_weights.detach().reshape(-1, *([1] * (values.ndim - 1)))
            denominator = weights.sum()
            return (values * weights).sum() / denominator.clamp_min(1e-12)

        g = global_prediction / global_scale
        local_normalized = local_prediction / local_scale
        t = global_target.detach() / global_scale
        gain_terms = None
        if gain_components is None:
            global_errors = F.smooth_l1_loss(g[:, 1:], t[:, 1:], reduction="none")
            global_action_losses = torch.stack(
                [reduce_rows(global_errors[:, i], available[:, i]) for i in (0, 1)]
            )
            global_loss = reduce_rows(global_errors, available)
            gain_probability_loss = global_prediction.new_zeros(())
        else:
            gain_terms = signed_gain_losses(
                gain_components,
                global_target[:, 1:],
                global_scale,
                row_weights,
                available,
            )
            # Sign BCE stays outside the magnitude/local gradient balancer.
            global_action_losses = (gain_terms["positive"] + gain_terms["negative"]) / 2
            global_loss = global_action_losses.mean()
            gain_probability_loss = gain_terms["probability"].mean()
        local_errors = F.smooth_l1_loss(
            local_normalized[:, 1:],
            local_target.detach()[:, 1:] / local_scale,
            reduction="none",
        )
        local_loss = reduce_rows(local_errors, available)
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
            if self.config.metric_quality is not None:
                # Rank only physically safe observed alternatives. Risk owns
                # harmful actions; render reward cannot make them rank targets.
                informative = informative & ~harms[:, a].bool() & ~harms[:, b].bool()
            # Bounded normalized weights keep a few very large proxy labels
            # from dominating the pair objective while retaining their order.
            normalized_margin = margin_values / margin_scale
            weight = normalized_margin.abs().clamp(max=5)
            separation = (
                self.config.ranking_margin / margin_scale
                if self.config.ranking_margin_units == "raw"
                else self.config.ranking_margin
            )
            # Large verified advantages should not be satisfied by an almost
            # tied prediction. This changes optimization only, not utility or
            # the raw-unit safety guard. Harmful pairs stay excluded above.
            separation = torch.maximum(
                normalized_margin.abs() * self.config.ranking_gap_fraction,
                torch.as_tensor(separation, device=g.device, dtype=g.dtype),
            )
            error = (separation - normalized_margin.sign() * (g[:, a] - g[:, b])).clamp_min(0)
            pair_available = action_available[:, a] & action_available[:, b]
            rank_terms.append(reduce_rows(error * weight * informative, pair_available))
            informative_pairs.append(
                (informative & pair_available).float().sum()
                / pair_available.float().sum().clamp_min(1)
            )
        ranking = torch.stack(rank_terms).mean()
        # Use logits directly rather than probability BCE for numerical stability.
        harm_errors = F.binary_cross_entropy_with_logits(
            logits, harms.detach()[:, 1:].float(), reduction="none"
        )
        event_errors = None
        if grouped:
            if (
                risk_components is None
                or risk_components.shape != (*harms.shape, 2)
                or risk_components.device != harms.device
                or not bool(((risk_components == 0) | (risk_components == 1)).all())
                or not bool((risk_components[:, 0] == 0).all())
                or not torch.equal(risk_components.bool().any(-1), harms.bool())
                or not bool(((harms == 0) | (harms == 1)).all())
            ):
                raise ValueError(
                    "grouped risk requires binary [R,3,2] local/global labels with exact OR and STOP zero"
                )
            local_global_errors = F.binary_cross_entropy_with_logits(
                event_logits[:, :, :2], risk_components.detach()[:, 1:].float(), reduction="none"
            )
            event_errors = torch.cat((local_global_errors, harm_errors[..., None]), -1)
            # Fixed mean over three dimensionless events, retaining total harm weight.
            harm_errors = event_errors.mean(-1)
        harm_loss = reduce_rows(harm_errors, available)
        action_losses = {}
        for i, action in enumerate(("stereo", "gaussian")):
            action_losses[f"{action}_global"] = global_action_losses[i]
            action_losses[f"{action}_local"] = reduce_rows(local_errors[:, i], available[:, i])
            action_losses[f"{action}_harm"] = reduce_rows(harm_errors[:, i], available[:, i])
            if gain_terms is not None:
                action_losses[f"{action}_gain_probability"] = gain_terms["probability"][i]
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
            "selector_ranking_gap_fraction": self.config.ranking_gap_fraction,
            "selector_harm_positive_fraction": float(
                (harms[:, 1:] * available.to(harms.dtype)).sum()
                / available.to(harms.dtype).sum().clamp_min(1)
            ),
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
        if gain_terms is not None:
            diagnostics.update(
                selector_gain_mode=self.config.selector_gain_mode,
                selector_gain_probability_loss=float(gain_probability_loss.detach()),
                selector_gain_probability_weight=self.config.gain_probability_weight,
            )
        # Read the added per-action metrics back together, rather than adding
        # one GPU synchronization for every new diagnostic scalar.
        with torch.no_grad():
            action_metrics: dict[str, torch.Tensor] = {}
            for i, action in enumerate(("stereo", "gaussian")):
                action_metrics.update(
                    {
                        f"selector_{action}_{name}_loss": action_losses[f"{action}_{name}"]
                        for name in ("global", "local", "harm")
                    }
                )
                action_metrics[f"selector_{action}_harm_brier"] = reduce_rows(
                    (logits[:, i].sigmoid() - harms[:, i + 1]).square(), available[:, i]
                )
                action_metrics[f"selector_{action}_harm_rate"] = reduce_rows(
                    harms[:, i + 1], available[:, i]
                )
                action_metrics[f"selector_{action}_global_gain_mae"] = reduce_rows(
                    (global_prediction[:, i + 1] - global_target[:, i + 1]).abs(), available[:, i]
                )
                if gain_terms is not None:
                    assert gain_components is not None
                    probability = gain_components.positive_probability[:, i]
                    positive = (global_target[:, i + 1] > 0).float()
                    for name in ("probability", "positive", "negative"):
                        action_metrics[f"selector_{action}_gain_{name}_loss"] = gain_terms[name][i]
                    action_metrics[f"selector_{action}_gain_positive_brier"] = reduce_rows(
                        (probability - positive).square(), available[:, i]
                    )
                    action_metrics[f"selector_{action}_gain_positive_rate"] = reduce_rows(
                        positive, available[:, i]
                    )
                    for sign in ("positive", "negative"):
                        action_metrics[f"selector_{action}_gain_{sign}_count"] = gain_terms[
                            f"{sign}_count"
                        ][i].float()
                if event_errors is not None:
                    assert risk_components is not None
                    event_target = torch.cat((risk_components[:, i + 1], harms[:, i + 1, None]), -1)
                    for j, name in enumerate(("local", "global", "any")):
                        action_metrics[f"selector_{action}_risk_{name}_loss"] = reduce_rows(
                            event_errors[:, i, j], available[:, i]
                        )
                        action_metrics[f"selector_{action}_risk_{name}_brier"] = reduce_rows(
                            (event_logits[:, i, j].sigmoid() - event_target[:, j]).square(),
                            available[:, i],
                        )
                        action_metrics[f"selector_{action}_risk_{name}_rate"] = reduce_rows(
                            event_target[:, j], available[:, i]
                        )
            values = torch.stack(tuple(action_metrics.values())).detach().cpu().tolist()
            diagnostics.update(dict(zip(action_metrics, values, strict=True)))
        return {
            "global": global_loss,
            "local": local_loss,
            "ranking": ranking,
            "harm": harm_loss,
            "gain_probability": gain_probability_loss,
            **action_losses,
        }, diagnostics

    def validate_state(self) -> None:
        self.normalization.validate_state()
