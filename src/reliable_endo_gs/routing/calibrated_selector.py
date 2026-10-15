"""Frozen gain/risk calibration bound to the exact repair artifact."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.contracts.interventions import RISK_EVENT_NAMES
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.gain import SignedGainPrediction
from reliable_endo_gs.routing.repairability import select_repairability
from reliable_endo_gs.routing.selector import Decision


class CalibratedSelector(nn.Module):
    """TRAIN-fitted [R,F] -> [R,3] gain/risk; unavailable calibration abstains."""

    gain_slope: torch.Tensor
    gain_bias: torch.Tensor
    risk_temperature: torch.Tensor
    risk_bias: torch.Tensor
    calibration_available: torch.Tensor
    event_temperature: torch.Tensor
    event_bias: torch.Tensor
    event_calibration_available: torch.Tensor

    def __init__(
        self,
        head: BalancedSelector,
        decision_config: InterventionConfig,
        action_identity: Mapping[str, object],
        mask_identity: str,
    ) -> None:
        super().__init__()
        # The gate uses the directly supervised any-event logit. Local/global
        # events are calibrated independently for diagnostics; this does not
        # impose a new decision rule or certify a coherent joint distribution.
        self.head = head.eval().requires_grad_(False)
        self.config = decision_config
        self.action_identity = dict(action_identity)
        self.mask_identity = mask_identity
        self.register_buffer("gain_slope", torch.ones(2))
        self.register_buffer("gain_bias", torch.zeros(2))
        self.register_buffer("risk_temperature", torch.ones(2))
        self.register_buffer("risk_bias", torch.zeros(2))
        self.register_buffer("calibration_available", torch.zeros(2, dtype=torch.bool))
        if head.config.selector_risk_mode == "local_global":
            self.register_buffer("event_temperature", torch.ones(2, 2))
            self.register_buffer("event_bias", torch.zeros(2, 2))
            self.register_buffer("event_calibration_available", torch.zeros(2, 2, dtype=torch.bool))

    @property
    def artifact_schema(self) -> str:
        """Historical scalar artifacts retain their tensor schema."""
        return (
            "calibrated_selector.v2"
            if self.config.selector_risk_mode == "local_global"
            else "calibrated_selector.v1"
        )

    def prediction_metadata(self) -> dict[str, object]:
        """Identify transformed outputs and unsupported identity fallbacks."""
        events = None
        if self.config.selector_risk_mode == "local_global":
            events = (
                torch.cat(
                    (self.event_calibration_available, self.calibration_available[:, None]), -1
                )
                .cpu()
                .tolist()
            )
        return {
            "source": "TRAIN_calibration_artifact",
            "global_gain": "affine_expected_utility",
            "global_gain_calibration_available": self.calibration_available.cpu().tolist(),
            "risk_any": "platt_probability",
            "risk_any_calibration_available": self.calibration_available.cpu().tolist(),
            "risk_event_names": list(RISK_EVENT_NAMES) if events is not None else None,
            "risk_event_calibration_available": events,
            "unsupported_event_output": "raw_identity_transform",
            "gain_components": "raw_head_sign_and_magnitudes",
            "gain_components_reconstruct": "raw_expected_gain_before_affine",
            "action_order": ["STEREO", "GAUSSIAN"],
        }

    @torch.no_grad()
    def outputs_with_components(
        self, features: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, SignedGainPrediction | None
    ]:
        """One encode: calibrated gains/risks, raw local gain and signed components.

        Gains/risks are [R,3], events [R,3,3]; signed components remain [R,2]
        and reconstruct the raw expected gain, before its affine calibration.
        Unsupported event transforms are identity and explicitly marked in
        ``prediction_metadata``. Only supported any-event calibration gates actions.
        """
        gain, local, logits, components = self.head.outputs_with_logits(features)
        zero = gain[:, :1]
        gain = torch.cat((zero, gain[:, 1:] * self.gain_slope + self.gain_bias), 1)
        events = None
        if self.config.selector_risk_mode == "local_global":
            temperature = torch.cat((self.event_temperature, self.risk_temperature[:, None]), -1)
            bias = torch.cat((self.event_bias, self.risk_bias[:, None]), -1)
            events = torch.cat(
                (logits.new_zeros((len(features), 1, 3)), (logits / temperature + bias).sigmoid()),
                1,
            )
            risks = events[:, :, 2]
        else:
            risks = torch.cat(
                (zero, (logits / self.risk_temperature + self.risk_bias).sigmoid()), 1
            )
        return gain, local, risks, events, components

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gain, _, risks, _, _ = self.outputs_with_components(features)
        return gain, risks

    @torch.no_grad()
    def select(
        self,
        features: torch.Tensor,
        costs_ms: torch.Tensor,
        valid: torch.Tensor,
        remaining_ms: float,
    ) -> Decision:
        gains, risks = self.predict(features)
        available = torch.cat(
            (self.calibration_available.new_ones(1), self.calibration_available)
        ).expand_as(risks)
        return select_repairability(
            gains, risks, costs_ms, valid, remaining_ms, self.config, risk_available=available
        )

    def validate_state(self) -> None:
        self.head.validate_state()
        if (
            not bool(self.head.normalization.fitted)
            or not all(bool(torch.isfinite(x).all()) for x in self.state_dict().values())
            or not bool(
                (self.risk_temperature > 0).all()
                & (self.gain_slope >= 0).all()
                & (self.gain_slope <= 2).all()
            )
        ):
            raise ValueError("invalid calibrated selector state")
        for key in (
            "selector_architecture",
            "selector_balance_mode",
            "selector_gain_mode",
            "selector_risk_mode",
            "metric_quality",
            "metric_scales",
            "stereo_utility_multiplier",
            "gaussian_guard_multiplier",
            "repair_max_disparity_px",
            "cost_weight_per_ms",
            "maximum_stereo_harm_probability",
            "maximum_gaussian_harm_probability",
        ):
            if getattr(self.head.config, key) != getattr(self.config, key):
                raise ValueError(
                    "calibrated head architecture/utility/guard differs from deployment"
                )
        if not bool(
            (self.risk_temperature >= math.exp(-2) - 1e-6).all()
            & (self.risk_temperature <= math.exp(2) + 1e-6).all()
            & (self.risk_bias.abs() <= 4).all()
            & (self.gain_bias.abs() <= self.head.normalization.gain_scales[0] + 1e-6).all()
        ):
            raise ValueError("gain/risk calibration outside declared bounds")
        unsupported = ~self.calibration_available
        if not bool(
            (self.risk_temperature[unsupported] == 1).all()
            & (self.risk_bias[unsupported] == 0).all()
            & (self.gain_slope[unsupported] == 1).all()
            & (self.gain_bias[unsupported] == 0).all()
        ):
            raise ValueError("unsupported calibration must retain the raw identity transform")
        if self.config.selector_risk_mode == "local_global" and not bool(
            (self.event_temperature >= math.exp(-2) - 1e-6).all()
            & (self.event_temperature <= math.exp(2) + 1e-6).all()
            & (self.event_bias.abs() <= 4).all()
        ):
            raise ValueError("event calibration outside declared bounds")
        if self.config.selector_risk_mode == "local_global" and not bool(
            (self.event_temperature[~self.event_calibration_available] == 1).all()
            & (self.event_bias[~self.event_calibration_available] == 0).all()
        ):
            raise ValueError("unsupported event calibration must retain the raw identity transform")


def load_calibrated_selector(
    path: Path,
    *,
    expected_action_identity: Mapping[str, object],
    expected_config: InterventionConfig,
    expected_mask_identity: str,
    expected_source_sha256: str,
) -> CalibratedSelector:
    """Validate a new immutable artifact; never resume old joint optimizer state."""
    raw: Any = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(raw, dict)
        or raw.get("schema") not in ("calibrated_selector.v1", "calibrated_selector.v2")
        or raw.get("action_identity") != dict(expected_action_identity)
        or raw.get("decision_config") != expected_config.to_dict()
        or raw.get("target_mask_index_sha256") != expected_mask_identity
        or raw.get("source_sha256") != expected_source_sha256
        or raw.get("development_used_for_fitting") is not False
    ):
        raise ValueError("calibrated selector artifact binding mismatch")
    head = BalancedSelector(raw["width"], InterventionConfig(**raw["head_config"]))
    model = CalibratedSelector(
        head, expected_config, expected_action_identity, expected_mask_identity
    )
    if raw["schema"] != model.artifact_schema:
        raise ValueError("calibrated selector event schema mismatch")
    if set(raw["state"]) != set(model.state_dict()) or any(
        not isinstance(v, torch.Tensor)
        or v.shape != model.state_dict()[k].shape
        or v.dtype != model.state_dict()[k].dtype
        for k, v in raw["state"].items()
    ):
        raise ValueError("invalid calibrated selector tensor schema")
    model.load_state_dict(raw["state"], strict=True)
    model.validate_state()
    report = raw.get("calibration", {})
    partition = raw.get("partition", {})
    if (
        report.get("development_used_for_calibration") is not False
        or report.get("cutoffs_tuned") is not False
        or partition.get("parent_roles_changed") is not False
    ):
        raise ValueError("missing TRAIN calibration/partition provenance")
    groups = [
        set(partition.get(k + "_keyframes", [])) for k in ("fit", "validation", "calibration")
    ]
    if any(len(g) < 2 for g in groups) or any(
        groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("overlapping or missing internal TRAIN groups")
    for i, name in enumerate(("STEREO", "GAUSSIAN")):
        row = report.get("actions", {}).get(name, {})
        if row.get("supported") is not bool(model.calibration_available[i]):
            raise ValueError("calibration availability disagrees with support evidence")
        if bool(model.calibration_available[i]) and (
            row.get("safe_keyframes", 0) < 2
            or row.get("unsafe_keyframes", 0) < 2
            or row.get("observations", 0) < 8
        ):
            raise ValueError("insufficient independent safe/harm calibration support")
        if model.config.selector_risk_mode == "local_global":
            for j, event in enumerate(RISK_EVENT_NAMES):
                event_row = row.get("events", {}).get(event, {})
                supported = (
                    bool(model.event_calibration_available[i, j])
                    if j < 2
                    else bool(model.calibration_available[i])
                )
                if event_row.get("supported") is not supported:
                    raise ValueError(
                        "event calibration availability disagrees with support evidence"
                    )
                if supported and (
                    event_row.get("safe_keyframes", 0) < 2
                    or event_row.get("unsafe_keyframes", 0) < 2
                    or event_row.get("observations", 0) < 8
                ):
                    raise ValueError("insufficient independent event calibration support")
    return model.eval().requires_grad_(False)
