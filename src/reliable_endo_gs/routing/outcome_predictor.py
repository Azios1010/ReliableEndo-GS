"""Small pre-action predictor of separate metric changes and constraint risk.

Fitted only from an offline training intervention table. No GT, teacher or
post-action outcome is a predictor input; GT is detached supervision only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from reliable_endo_gs.config.outcome_training import OutcomeTrainingConfig
from reliable_endo_gs.routing.context import AVAILABILITY_COLUMNS, BALANCED_FEATURE_NAMES
from reliable_endo_gs.routing.tradeoff import (
    ACTION_NAMES,
    COMPONENT_NAMES,
    SCALE_FLOORS,
    TradeoffConfig,
    constraint_margins_tensor,
)


class OutcomeNetwork(nn.Sequential):
    """Hidden dropout without changing historical parameter names/layouts."""

    def __init__(self, *layers: nn.Module, dropout: float = 0.0) -> None:
        super().__init__(*layers)
        self.dropout = dropout

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self:
            value = layer(value)
            if isinstance(layer, nn.ReLU) and self.dropout:
                value = F.dropout(value, p=self.dropout, training=self.training)
        return value


class OutcomePredictor(nn.Module):
    """[R,61] -> changes [R,3,12] and violation probabilities [R,3].

    Gaussian repair never changes disparity, hence its stereo geometry changes
    are exactly zero by action semantics, not learned from GT or clipped to win.
    Independent risk/gain encoders prevent BCE changing the gain representation.
    """

    feature_center: torch.Tensor
    feature_scale: torch.Tensor
    metric_scale: torch.Tensor
    fitted: torch.Tensor
    risk_positive_weight: torch.Tensor
    structural: torch.Tensor
    risk_temperature: torch.Tensor
    risk_bias: torch.Tensor
    risk_calibrated: torch.Tensor

    def __init__(
        self,
        width: int = 32,
        *,
        linear: bool = False,
        component_risk: bool = False,
        dropout: float = 0.0,
        risk_context_config: TradeoffConfig | None = None,
    ) -> None:
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("predictor width must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("predictor dropout must lie in [0,1)")
        features = len(BALANCED_FEATURE_NAMES)
        actions, components = len(ACTION_NAMES), len(COMPONENT_NAMES)
        self.register_buffer("feature_center", torch.zeros(features))
        self.register_buffer("feature_scale", torch.ones(features))
        self.register_buffer("metric_scale", torch.ones(components))
        self.register_buffer("fitted", torch.tensor(False))
        self.register_buffer("risk_positive_weight", torch.ones(actions))
        self.component_risk = component_risk
        if risk_context_config is not None and (
            not component_risk or not risk_context_config.schema.endswith(".v2")
        ):
            raise ValueError("conditional risk needs v2 component-risk policy")
        self.risk_context_config = risk_context_config
        if component_risk:
            self.register_buffer("risk_temperature", torch.ones(actions))
            self.register_buffer("risk_bias", torch.zeros(actions))
            self.register_buffer("risk_calibrated", torch.zeros(actions, dtype=torch.bool))
        structural = torch.ones(actions, components)
        structural[2, :5] = 0
        structural[2, -1] = 0
        self.register_buffer("structural", structural)

        def network(outputs: int, inputs: int = features) -> nn.Sequential:
            layers: list[nn.Module] = (
                []
                if linear
                else [nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()]
            )
            output = nn.Linear(inputs if linear else width, outputs)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
            return OutcomeNetwork(*layers, output, dropout=dropout)

        self.change_network = network(actions * components)
        self.risk_network = network(
            actions * (components + 1) if component_risk else actions,
            features + actions * components if risk_context_config is not None else features,
        )

    @property
    def risk_feature_names(self) -> tuple[str, ...]:
        """Version-bound encoder inputs; conditional features are predictions."""
        if self.risk_context_config is None:
            return BALANCED_FEATURE_NAMES
        return BALANCED_FEATURE_NAMES + tuple(
            f"predicted_guard_margin:{action}:{metric}"
            for action in ACTION_NAMES
            for metric in COMPONENT_NAMES
        )

    @torch.no_grad()
    def fit_normalization(
        self, features: torch.Tensor, changes: torch.Tensor, observed: torch.Tensor
    ) -> None:
        """Training [R,F]/[R,A,C]/bool[R,A]; one scale per metric across actions."""
        if bool(self.fitted):
            raise ValueError("predictor normalization is immutable once fitted")
        if (
            features.ndim != 2
            or features.shape[1] != self.feature_center.numel()
            or changes.shape != (features.shape[0], len(ACTION_NAMES), len(COMPONENT_NAMES))
            or observed.shape != changes.shape[:2]
            or observed.dtype != torch.bool
            or not bool(observed.any())
            or not bool(torch.isfinite(features).all() & torch.isfinite(changes).all())
        ):
            raise ValueError("invalid training-only normalization inputs")
        quantiles = features.float().quantile(features.new_tensor([0.25, 0.5, 0.75]), dim=0)
        spread = (quantiles[2] - quantiles[0]) / 1.349
        # Constants retain their raw variation rather than being permanently suppressed.
        self.feature_center.copy_(quantiles[1])
        self.feature_scale.copy_(torch.where(spread > 1e-6, spread, torch.ones_like(spread)))
        for column in AVAILABILITY_COLUMNS:
            self.feature_center[column] = 0
            self.feature_scale[column] = 1
        metric = changes[observed].abs().quantile(0.75, dim=0)
        self.metric_scale.copy_(torch.maximum(metric, metric.new_tensor(SCALE_FLOORS)))
        self.fitted.fill_(True)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized changes and risk logits, used by offline supervised fitting."""
        changes, joint, _ = self.forward_details(features)
        return changes, joint

    def forward_details(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Changes [R,A,C], joint logits [R,A], optional metric logits [R,A,C].

        The joint event is learned directly; correlated metric risks are never
        combined using an unjustified independence assumption.
        """
        if (
            not bool(self.fitted)
            or features.ndim != 2
            or features.shape[1] != self.feature_center.numel()
            or not bool(torch.isfinite(features).all())
        ):
            raise ValueError("predictor needs fitted finite pre-action [R,F] features")
        values = ((features.detach().float() - self.feature_center) / self.feature_scale).clamp(
            -5, 5
        )
        changes = self.change_network(values).reshape(-1, len(ACTION_NAMES), len(COMPONENT_NAMES))
        changes = changes * self.structural
        if self.risk_context_config is not None:
            # Detached predictions are available before the action. This is not
            # teacher forcing or GT input; risk cannot update the gain encoder.
            margins = constraint_margins_tensor(
                changes.detach() * self.metric_scale,
                self.metric_scale.tolist(),
                self.risk_context_config,
            ).clamp(-5, 5)
            values = torch.cat((values, margins.flatten(1)), dim=1)
        risk = self.risk_network(values)
        if self.component_risk:
            risk = risk.reshape(-1, len(ACTION_NAMES), len(COMPONENT_NAMES) + 1)
            return changes, risk[..., 0], risk[..., 1:]
        return changes, risk, None

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        changes, logits = self(features)
        # Positive-weighted BCE shifts optimal odds by its training weight.
        # Undo that known shift; held-out calibration is still required.
        logits = logits - self.risk_positive_weight.log()
        if self.component_risk:
            probability = (logits / self.risk_temperature + self.risk_bias).sigmoid()
            # Insufficient held-out class/group support means abstention, not
            # a fabricated low-risk estimate for an unvalidated repair source.
            probability = torch.where(
                self.risk_calibrated, probability, torch.ones_like(probability)
            )
        else:
            probability = logits.sigmoid()
        return changes * self.metric_scale, probability

    @torch.no_grad()
    def predict_component_risks(self, features: torch.Tensor) -> torch.Tensor:
        """Auxiliary per-metric risks [R,A,C], diagnostic, not joint probabilities."""
        _, _, logits = self.forward_details(features)
        if logits is None:
            raise ValueError("historical predictor has no component risk outputs")
        return logits.sigmoid() * self.structural


def load_outcome_predictor(
    path: Path,
    *,
    expected_config: TradeoffConfig,
    expected_action_identity: Mapping[str, object],
    expected_candidate_hash: str,
) -> OutcomePredictor:
    """Strict CPU import for the experimental predictor; no optimizer/upstream import.

    Action identity must be supplied by the frozen experiment. A different
    action checkpoint/candidate cannot silently reuse these outcome predictions.
    """
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(raw, dict)
        or raw.get("schema")
        != (
            "metric_outcome_predictor.v2"
            if expected_config.schema.endswith(".v2")
            else "metric_outcome_predictor.v1"
        )
        or raw.get("kind") not in ("linear", "mlp")
        or raw.get("feature_names") != list(BALANCED_FEATURE_NAMES)
        or raw.get("component_names") != list(COMPONENT_NAMES)
        or raw.get("action_names") != list(ACTION_NAMES)
        or raw.get("frozen_action_identity") != expected_action_identity
        or raw.get("candidate_config_hash") != expected_candidate_hash
        or raw.get("config_hash") != expected_config.config_hash
    ):
        raise ValueError("outcome predictor/action identity mismatch")
    config = TradeoffConfig(**raw["config"])
    if config != expected_config:
        raise ValueError("outcome predictor contract mismatch")
    training = raw.get("training_policy")
    training_config = OutcomeTrainingConfig(**training) if isinstance(training, dict) else None
    if training is not None and training_config is None:
        raise ValueError("invalid predictor training policy")
    if training_config is not None and (
        raw.get("training_policy_hash") != training_config.config_hash
        or not config.schema.endswith(".v2")
    ):
        raise ValueError("predictor training policy identity mismatch")
    model = OutcomePredictor(
        config.width,
        linear=raw["kind"] == "linear",
        component_risk=config.schema.endswith(".v2"),
        dropout=training_config.dropout if training_config is not None else 0.0,
        risk_context_config=config
        if training_config is not None and training_config.risk_feature_mode == "predicted_margins"
        else None,
    )
    if model.risk_context_config is not None and raw.get("risk_feature_names") != list(
        model.risk_feature_names
    ):
        raise ValueError("conditional risk feature provenance mismatch")
    payload = raw.get("state_dict")
    expected = model.state_dict()
    if (
        not isinstance(payload, dict)
        or set(payload) != set(expected)
        or any(
            not isinstance(v, torch.Tensor)
            or v.shape != expected[k].shape
            or v.dtype != expected[k].dtype
            or not bool(torch.isfinite(v).all())
            for k, v in payload.items()
        )
    ):
        raise ValueError("invalid outcome predictor tensors")
    if (
        not bool(payload["fitted"])
        or not bool((payload["feature_scale"] > 0).all())
        or not bool((payload["metric_scale"] > 0).all())
        or not bool(
            (
                (payload["risk_positive_weight"] >= 0.25) & (payload["risk_positive_weight"] <= 4)
            ).all()
        )
        or not torch.equal(payload["structural"], expected["structural"])
    ):
        raise ValueError("invalid outcome predictor normalization/action semantics")
    if model.component_risk and (
        not bool((payload["risk_temperature"] > 0).all())
        or not torch.equal(
            payload["risk_positive_weight"], torch.ones_like(payload["risk_positive_weight"])
        )
    ):
        raise ValueError(
            "v2 risk requires positive calibration temperatures and proper unweighted BCE"
        )
    if model.component_risk:
        partition = raw.get("training_partition")
        calibration = raw.get("training_risk_calibration")
        if not isinstance(partition, dict) or not isinstance(calibration, dict):
            raise ValueError("v2 predictor lacks training-only risk calibration provenance")
        definition = {k: v for k, v in partition.items() if k != "partition_sha256"}
        if (
            partition.get("partition_sha256")
            != hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
            or partition.get("source_training_table_sha256") != raw.get("training_table_sha256")
            or partition.get("development_used_for_fitting") is not False
        ):
            raise ValueError("invalid risk calibration partition identity")
        fit_ids, cal_ids = partition.get("fit_sample_ids"), partition.get("calibration_sample_ids")
        if (
            not isinstance(fit_ids, list)
            or not isinstance(cal_ids, list)
            or not fit_ids
            or not cal_ids
            or any(not isinstance(s, str) for s in fit_ids + cal_ids)
            or len(set(fit_ids + cal_ids)) != len(fit_ids) + len(cal_ids)
            or any(
                s.split("/")[0] not in {f"dataset_{i}" for i in range(1, 8)}
                for s in fit_ids + cal_ids
            )
            or set(s.rsplit("/", 1)[0] for s in fit_ids) & set(s.rsplit("/", 1)[0] for s in cal_ids)
        ):
            raise ValueError("invalid training/calibration sample separation")
        if training_config is not None:
            validation_ids = partition.get("validation_sample_ids")
            if (
                not isinstance(validation_ids, list)
                or not validation_ids
                or any(not isinstance(s, str) for s in validation_ids)
                or len(set(fit_ids + cal_ids + validation_ids))
                != len(fit_ids + cal_ids + validation_ids)
                or any(
                    s.split("/")[0] not in {f"dataset_{i}" for i in range(1, 8)}
                    for s in validation_ids
                )
                or set(s.rsplit("/", 1)[0] for s in validation_ids)
                & set(s.rsplit("/", 1)[0] for s in fit_ids + cal_ids)
            ):
                raise ValueError("invalid fit/validation/calibration keyframe separation")
            selection = raw.get("training_selection")
            if (
                not isinstance(selection, dict)
                or selection.get("enabled") is not True
                or selection.get("fit_sample_ids") != fit_ids
                or selection.get("validation_sample_ids") != validation_ids
                or selection.get("training_policy_hash") != training_config.config_hash
                or selection.get("best_state_restored") is not True
                or selection.get("development_used_for_selection") is not False
            ):
                raise ValueError("missing grouped checkpoint selection provenance")
            best_step = selection.get("best_step")
            completed_steps = selection.get("completed_steps")
            score = selection.get("best_validation_score")
            if (
                type(best_step) is not int
                or type(completed_steps) is not int
                or not training_config.min_steps
                <= best_step
                <= completed_steps
                <= training_config.max_steps
                * (2 if training_config.risk_feature_mode == "predicted_margins" else 1)
                or not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
                or selection.get("stop_reason") not in ("validation_patience", "max_steps")
            ):
                raise ValueError("invalid grouped checkpoint selection results")
            if training_config.independent_encoder_stopping:
                encoders = selection.get("encoder_selection")
                if not isinstance(encoders, dict) or set(encoders) != {"changes", "risk"}:
                    raise ValueError("missing independent encoder selection provenance")
                for item in encoders.values():
                    if (
                        not isinstance(item, dict)
                        or type(item.get("best_step")) is not int
                        or not training_config.min_steps <= item["best_step"] <= completed_steps
                        or not isinstance(item.get("best_score"), (int, float))
                        or not math.isfinite(item["best_score"])
                        or item.get("best_state_restored") is not True
                    ):
                        raise ValueError("invalid independent encoder selection provenance")
            if training_config.risk_feature_mode == "predicted_margins":
                refit = selection.get("risk_refit")
                first = selection.get("first_stage_completed_steps")
                if (
                    not isinstance(refit, dict)
                    or type(first) is not int
                    or not training_config.min_steps <= first <= training_config.max_steps
                    or type(refit.get("completed_steps")) is not int
                    or type(refit.get("best_step")) is not int
                    or not training_config.min_steps
                    <= refit["best_step"]
                    <= refit["completed_steps"]
                    <= training_config.max_steps
                    or completed_steps != first + refit["completed_steps"]
                    or best_step != first + refit["best_step"]
                    or refit.get("gain_unchanged") is not True
                    or refit.get("best_state_restored") is not True
                    or refit.get("seed") != config.seed
                ):
                    raise ValueError("invalid fixed-gain conditional risk refit provenance")
        expected_calibration = (
            training_config.calibration_method if training_config is not None else "grid"
        )
        if calibration.get("method", "grid") != expected_calibration:
            raise ValueError("risk calibration method/training policy mismatch")
        if expected_calibration == "platt" and (
            calibration.get("objective_population") != "equal_keyframe"
            or training_config is None
            or calibration.get("regularization") != training_config.calibration_regularization
            or calibration.get("platt_bounds")
            != {"log_slope": [-2, 2], "bias": [-4, 4], "max_iter": 80}
        ):
            raise ValueError("risk calibration regularization/provenance mismatch")
        actions = calibration.get("actions")
        if (
            not isinstance(actions, dict)
            or set(actions) != set(ACTION_NAMES)
            or calibration.get("cutoff") != config.maximum_predicted_violation_probability
            or calibration.get("cutoff_tuned") is not False
        ):
            raise ValueError("missing per-action calibration status")
        for ai, action in enumerate(ACTION_NAMES):
            item = actions[action]
            supported = item.get("supported") if isinstance(item, dict) else None
            if type(supported) is not bool or bool(payload["risk_calibrated"][ai]) != supported:
                raise ValueError("risk calibration availability mismatch")
            if supported and any(
                type(item.get(key)) is not int or item[key] < config.calibration_min_class_groups
                for key in ("safe_keyframes", "unsafe_keyframes")
            ):
                raise ValueError("insufficient per-action calibration group evidence")
            temperature = item.get("temperature", 1.0)
            bias = item.get("bias", 0.0)
            if (
                float(payload["risk_temperature"][ai]) != temperature
                or float(payload["risk_bias"][ai]) != bias
            ):
                raise ValueError("risk calibration parameters mismatch")
    model.load_state_dict(payload, strict=True)
    return model.eval().requires_grad_(False)
