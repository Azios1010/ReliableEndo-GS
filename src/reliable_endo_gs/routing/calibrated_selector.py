"""Frozen scalar gain/risk selector bound to the exact repair artifact."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.repairability import select_repairability
from reliable_endo_gs.routing.selector import Decision


class CalibratedSelector(nn.Module):
    """TRAIN-fitted [R,F] -> [R,3] gain/risk; unavailable calibration abstains."""

    gain_slope: torch.Tensor
    gain_bias: torch.Tensor
    risk_temperature: torch.Tensor
    risk_bias: torch.Tensor
    calibration_available: torch.Tensor

    def __init__(
        self,
        head: BalancedSelector,
        decision_config: InterventionConfig,
        action_identity: Mapping[str, object],
        mask_identity: str,
    ) -> None:
        super().__init__()
        if head.config.selector_gain_mode != "regression":
            raise ValueError("signed expected gain requires component-aware calibration")
        if head.config.selector_risk_mode != "union":
            raise ValueError(
                "local/global/any risk requires event-aware calibration; scalar Platt artifacts are incompatible"
            )
        self.head = head.eval().requires_grad_(False)
        self.config = decision_config
        self.action_identity = dict(action_identity)
        self.mask_identity = mask_identity
        self.register_buffer("gain_slope", torch.ones(2))
        self.register_buffer("gain_bias", torch.zeros(2))
        self.register_buffer("risk_temperature", torch.ones(2))
        self.register_buffer("risk_bias", torch.zeros(2))
        self.register_buffer("calibration_available", torch.zeros(2, dtype=torch.bool))

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gain, _, logits = self.head.utility_and_risk_logits(features)
        zero = gain[:, :1]
        return torch.cat((zero, gain[:, 1:] * self.gain_slope + self.gain_bias), 1), torch.cat(
            (zero, (logits / self.risk_temperature + self.risk_bias).sigmoid()), 1
        )

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
            "metric_quality",
            "metric_scales",
            "stereo_utility_multiplier",
            "gaussian_guard_multiplier",
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
        or raw.get("schema") != "calibrated_selector.v1"
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
    return model.eval().requires_grad_(False)
