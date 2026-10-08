"""Explicit experimental repairability protocol; legacy routing is unchanged."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class InterventionConfig:
    """Pilot settings, not fitted confidence calibration or gate certification."""

    schema: str = "intervention_selection.v1"
    matching_width: int = 8
    search_radius_px: int = 16
    search_step_px: int = 2
    matching_temperature: float = 0.05
    max_effective_fraction: float = 0.5
    kd_min_depth_m: float = 0.01
    kd_max_photometric_error: float | None = None
    kd_scale_px: float = 16.0
    rgb_scale: float = 0.07
    stereo_scale: float = 0.1
    stereo_weight: float = 1.0
    collateral_weight: float = 1.0
    ranking_weight: float = 1.0
    harm_weight: float = 1.0
    # Ranking margins are versioned because changing their units changes the
    # selector objective.  ``raw`` preserves historical v1/v2 behaviour;
    # ``train_scale`` is an opt-in remediation using the fitted gain scale.
    ranking_margin: float = 0.02
    ranking_margin_units: str = "raw"
    harm_tolerance: float = 0.001
    maximum_harm_probability: float = 0.5
    minimum_gain: float = 0.0
    total_budget_ms: float = 1000.0
    # v2 optimization settings; v1 config hashes omit these fields.
    normalization_frames: int = 32
    normalization_rows: int = 256
    # Legacy normalization feeds raw features during collection.  ``burn_in``
    # collects TRAIN statistics first and suppresses selector optimizer steps
    # until the transform is fitted.
    normalization_mode: str = "legacy"
    feature_clip: float = 5.0
    normalization_floor: float = 1e-6
    initial_gain_scale: float = 0.05
    minimum_gain_scale: float = 0.01
    gradient_ema_decay: float = 0.9
    balance_minimum: float = 0.05
    balance_maximum: float = 20.0
    local_auxiliary_weight: float = 1.0
    local_repair_weight: float = 1.0
    global_repair_weight: float = 1.0
    gradient_diagnostics_every: int = 25
    # Applied after model gradient balancing; independent of proxy utility units.
    stereo_training_priority: float = 1.0

    def __post_init__(self) -> None:
        if self.schema not in ("intervention_selection.v1", "intervention_selection.v2"):
            raise ValueError("unsupported intervention selection schema")
        if self.ranking_margin_units not in ("raw", "train_scale"):
            raise ValueError("ranking_margin_units must be raw or train_scale")
        if self.normalization_mode not in ("legacy", "burn_in"):
            raise ValueError("normalization_mode must be legacy or burn_in")
        if self.schema == "intervention_selection.v1" and (
            self.ranking_margin_units != "raw" or self.normalization_mode != "legacy"
        ):
            raise ValueError("v1 requires historical raw ranking and legacy normalization")
        for name in (
            "matching_width",
            "search_radius_px",
            "search_step_px",
            "normalization_frames",
            "normalization_rows",
            "gradient_diagnostics_every",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.search_radius_px % self.search_step_px:
            raise ValueError("search step must divide radius")
        if self.schema == "intervention_selection.v2" and self.normalization_rows < 2:
            raise ValueError("balanced normalization requires capacity for at least two regions")
        for name, value in asdict(self).items():
            if (
                name in {"schema", "ranking_margin_units", "normalization_mode"}
                or value is None
            ):
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if min(self.matching_temperature, self.kd_scale_px, self.rgb_scale, self.stereo_scale) <= 0:
            raise ValueError("temperature and normalization scales must be positive")
        if not 0 < self.max_effective_fraction < 1:
            raise ValueError("effective disparity correction fraction must be in (0,1)")
        if not 0 < self.maximum_harm_probability < 1:
            raise ValueError("harm probability cutoff must be in (0,1)")
        if not 0 <= self.gradient_ema_decay < 1:
            raise ValueError("gradient EMA decay must be in [0,1)")
        if not 0 < self.balance_minimum <= 1 <= self.balance_maximum:
            raise ValueError("gradient balancing bounds must include one")
        if (
            min(
                self.feature_clip,
                self.normalization_floor,
                self.initial_gain_scale,
                self.minimum_gain_scale,
                self.local_repair_weight,
                self.global_repair_weight,
                self.stereo_training_priority,
            )
            <= 0
        ):
            raise ValueError("normalization and local/global repair settings must be positive")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Preserve historical v1 serialization and identity exactly."""
        data = asdict(self)
        if self.schema == "intervention_selection.v1":
            data = {k: v for k, v in data.items() if k not in V2_FIELDS}
        elif self.stereo_training_priority == 1.0:
            # Preserve existing v2 config/checkpoint identities at neutral priority.
            data.pop("stereo_training_priority")
        # Omit neutral defaults so existing v2 config/checkpoint hashes remain
        # stable.  Explicit remediation settings are serialized and hashed.
        if self.ranking_margin_units == "raw":
            data.pop("ranking_margin_units", None)
        if self.normalization_mode == "legacy":
            data.pop("normalization_mode", None)
        return data


V2_FIELDS = frozenset(
    {
        "normalization_frames",
        "normalization_rows",
        "feature_clip",
        "normalization_floor",
        "initial_gain_scale",
        "minimum_gain_scale",
        "gradient_ema_decay",
        "balance_minimum",
        "balance_maximum",
        "local_auxiliary_weight",
        "local_repair_weight",
        "global_repair_weight",
        "gradient_diagnostics_every",
        "stereo_training_priority",
        "ranking_margin_units",
        "normalization_mode",
    }
)


def load_intervention_config(path: Path) -> InterventionConfig:
    """Reject unknown policy settings instead of silently changing objectives."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - set(InterventionConfig.__dataclass_fields__):
        raise ValueError("intervention config contains unknown fields")
    if (
        raw.get("schema", "intervention_selection.v1") == "intervention_selection.v1"
        and set(raw) & V2_FIELDS
    ):
        raise ValueError("balanced settings require intervention_selection.v2")
    return InterventionConfig(**raw)
