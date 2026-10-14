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
    # Candidate-search extent is independent from the maximum correction that
    # a stereo repair may write into the canonical disparity grid. ``None``
    # keeps the routing config as the owner for legacy callers.
    repair_max_disparity_px: float | None = None
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
    # Optional measured-latency penalty used after the hard remaining-budget
    # check. Zero preserves historical gain-only ranking.
    cost_weight_per_ms: float = 0.0
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
    # Opt-in physical outcome supervision, carried in checkpoint identity.
    # Scales are declared raw-unit references, never fitted on development.
    metric_quality: dict[str, object] | None = None
    metric_scales: tuple[float, ...] | None = None
    gaussian_guard_multiplier: float = 1.0
    maximum_stereo_harm_probability: float | None = None
    maximum_gaussian_harm_probability: float | None = None
    # Explicit action preference in physical utility labels, not a measured
    # improvement or risk discount. Neutral default preserves prior identities.
    stereo_utility_multiplier: float = 1.0
    # Extra head optimization uses only the current detached TRAIN snapshot;
    # it never replays outcomes of older model/action parameters.
    selector_updates_per_snapshot: int = 1
    selector_weight_decay: float = 0.0
    selector_gain_only_refinement: bool = False
    # Fraction of the measured, normalized safe-pair gap used as a ranking
    # separation target. Zero retains the historical constant margin.
    ranking_gap_fraction: float = 0.0
    # Opt-in action-specific encoders; legacy topology and hashes stay available.
    selector_architecture: str = "shared_actions"
    selector_balance_mode: str = "shared"
    # Explicit TRAIN-only metric supervision; zero keeps historical objectives.
    geometry_supervision_weight: float = 0.0
    stereo_repair_geometry_weight: float = 1.0
    geometry_loss_scale_mm: float = 1.0
    selector_context_frames: int = 1
    # Controlled ablation: freeze model updates, preserve gradient probes/losses.
    student_update_mode: str = "joint"
    selector_feature_mode: str = "statistics"
    selector_token_channels: int = 0
    selector_risk_mode: str = "union"
    selector_gain_mode: str = "regression"
    gain_probability_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.selector_gain_mode not in ("regression", "signed_expectation"):
            raise ValueError("unsupported selector gain mode")
        if self.selector_gain_mode == "signed_expectation" and (
            self.schema != "intervention_selection.v2"
            or self.selector_architecture != "separate_action_heads"
            or self.selector_balance_mode != "per_action"
            or self.metric_quality is None
        ):
            raise ValueError("signed expectation requires independent per-action physical v2 heads")
        weight = self.gain_probability_weight
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0
            or (self.selector_gain_mode == "regression" and weight != 1.0)
        ):
            raise ValueError(
                "gain probability weight requires signed expectation and must be positive"
            )
        if self.selector_feature_mode not in ("statistics", "mask_tokens"):
            raise ValueError("unsupported selector feature mode")
        if self.selector_risk_mode not in ("union", "local_global"):
            raise ValueError("unsupported selector risk mode")
        if type(self.selector_token_channels) is not int or self.selector_token_channels < 0:
            raise ValueError("selector token channels must be a nonnegative integer")
        if (self.selector_feature_mode == "mask_tokens") != (self.selector_token_channels > 0):
            raise ValueError("mask tokens require explicit positive student feature channels")
        if self.selector_feature_mode != "statistics" or self.selector_risk_mode != "union":
            if (
                self.schema != "intervention_selection.v2"
                or self.selector_architecture != "separate_action_heads"
            ):
                raise ValueError("token/grouped risk requires independent balanced v2 heads")
            if self.metric_quality is None:
                raise ValueError("token/grouped risk requires physical outcome supervision")
        if self.student_update_mode not in ("joint", "frozen"):
            raise ValueError("student update mode must be joint or frozen")
        if self.student_update_mode == "frozen" and self.schema != "intervention_selection.v2":
            raise ValueError("frozen student comparison requires balanced v2 learning")
        if (
            type(self.selector_context_frames) is not int
            or not 1 <= self.selector_context_frames <= 16
        ):
            raise ValueError("selector context frames must be an integer in [1,16]")
        if self.selector_context_frames != 1 and self.metric_quality is None:
            raise ValueError("fresh selector context requires explicit physical TRAIN truth")
        if self.geometry_supervision_weight and self.metric_quality is None:
            raise ValueError("geometry supervision requires explicit physical TRAIN truth")
        if self.selector_architecture not in ("shared_actions", "separate_action_heads"):
            raise ValueError("unsupported selector architecture")
        if self.selector_balance_mode not in ("shared", "per_action"):
            raise ValueError("unsupported selector balance mode")
        if self.selector_balance_mode == "per_action" and (
            self.schema != "intervention_selection.v2"
            or self.selector_architecture != "separate_action_heads"
        ):
            raise ValueError("per-action balance requires separate balanced v2 action heads")
        if (
            self.schema != "intervention_selection.v2"
            and self.selector_architecture != "shared_actions"
        ):
            raise ValueError("separate action heads require balanced v2 learning")
        if type(self.selector_gain_only_refinement) is not bool:
            raise ValueError("selector gain-only refinement must be boolean")
        if (
            type(self.selector_updates_per_snapshot) is not int
            or not 1 <= self.selector_updates_per_snapshot <= 16
        ):
            raise ValueError("selector updates per snapshot must be an integer in [1,16]")
        if not 0 <= self.ranking_gap_fraction <= 1:
            raise ValueError("ranking gap fraction must be in [0,1]")
        if self.schema != "intervention_selection.v2" and (
            self.selector_updates_per_snapshot != 1
            or self.selector_weight_decay != 0
            or self.ranking_gap_fraction != 0
            or self.selector_gain_only_refinement
        ):
            raise ValueError("selector optimization settings require balanced v2 learning")
        factor = self.stereo_utility_multiplier
        if (
            isinstance(factor, bool)
            or not isinstance(factor, (int, float))
            or not math.isfinite(factor)
            or factor <= 0
        ):
            raise ValueError("stereo utility multiplier must be finite and positive")
        if self.stereo_utility_multiplier != 1.0 and self.metric_quality is None:
            raise ValueError("stereo utility multiplier requires physical metric supervision")
        if (self.metric_quality is None) != (self.metric_scales is None):
            raise ValueError("metric quality requires explicit per-component scales")
        if self.metric_quality is not None:
            from reliable_endo_gs.routing.tradeoff import TradeoffConfig, utility_coefficients

            if self.schema != "intervention_selection.v2":
                raise ValueError("metric quality requires balanced v2 learning")
            if self.normalization_mode != "burn_in":
                raise ValueError("metric quality requires stationary burn-in normalization")
            policy = TradeoffConfig(**self.metric_quality)  # type: ignore[arg-type]
            if policy.schema != "metric_action_tradeoff.v2":
                raise ValueError("metric quality requires bounded v2 utility")
            assert self.metric_scales is not None
            utility_coefficients(self.metric_scales, policy)
            object.__setattr__(self, "metric_scales", tuple(self.metric_scales))
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
        if self.repair_max_disparity_px is not None and (
            not math.isfinite(self.repair_max_disparity_px) or self.repair_max_disparity_px <= 0
        ):
            raise ValueError("repair_max_disparity_px must be finite and positive when provided")
        if self.search_radius_px % self.search_step_px:
            raise ValueError("search step must divide radius")
        if self.schema == "intervention_selection.v2" and self.normalization_rows < 2:
            raise ValueError("balanced normalization requires capacity for at least two regions")
        for name, value in asdict(self).items():
            if (
                name
                in {
                    "schema",
                    "ranking_margin_units",
                    "normalization_mode",
                    "metric_quality",
                    "metric_scales",
                    "selector_gain_only_refinement",
                    "selector_architecture",
                    "selector_balance_mode",
                    "student_update_mode",
                    "selector_feature_mode",
                    "selector_risk_mode",
                    "selector_gain_mode",
                    "repair_max_disparity_px",
                }
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
        if not 0 <= self.gaussian_guard_multiplier <= 1:
            raise ValueError("Gaussian guard multiplier must be in [0,1]")
        for cutoff in (
            self.maximum_stereo_harm_probability,
            self.maximum_gaussian_harm_probability,
        ):
            if cutoff is not None and not 0 < cutoff < 1:
                raise ValueError("action harm probability cutoff must be in (0,1)")
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
                self.stereo_utility_multiplier,
                self.stereo_repair_geometry_weight,
                self.geometry_loss_scale_mm,
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
        for key, neutral in (
            ("metric_quality", None),
            ("metric_scales", None),
            ("gaussian_guard_multiplier", 1.0),
            ("maximum_stereo_harm_probability", None),
            ("maximum_gaussian_harm_probability", None),
            ("stereo_utility_multiplier", 1.0),
            ("selector_updates_per_snapshot", 1),
            ("selector_weight_decay", 0.0),
            ("selector_gain_only_refinement", False),
            ("ranking_gap_fraction", 0.0),
            ("selector_architecture", "shared_actions"),
            ("selector_balance_mode", "shared"),
            ("geometry_supervision_weight", 0.0),
            ("stereo_repair_geometry_weight", 1.0),
            ("geometry_loss_scale_mm", 1.0),
            ("selector_context_frames", 1),
            ("student_update_mode", "joint"),
            ("selector_feature_mode", "statistics"),
            ("selector_token_channels", 0),
            ("selector_risk_mode", "union"),
            ("selector_gain_mode", "regression"),
            ("gain_probability_weight", 1.0),
            ("repair_max_disparity_px", None),
            ("cost_weight_per_ms", 0.0),
        ):
            if data.get(key) == neutral:
                data.pop(key, None)
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
        "metric_quality",
        "metric_scales",
        "gaussian_guard_multiplier",
        "maximum_stereo_harm_probability",
        "maximum_gaussian_harm_probability",
        "stereo_utility_multiplier",
        "selector_updates_per_snapshot",
        "selector_weight_decay",
        "selector_gain_only_refinement",
        "ranking_gap_fraction",
        "selector_architecture",
        "selector_balance_mode",
        "geometry_supervision_weight",
        "stereo_repair_geometry_weight",
        "geometry_loss_scale_mm",
        "selector_context_frames",
        "student_update_mode",
        "selector_feature_mode",
        "selector_token_channels",
        "selector_risk_mode",
        "selector_gain_mode",
        "gain_probability_weight",
        "repair_max_disparity_px",
        "cost_weight_per_ms",
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
