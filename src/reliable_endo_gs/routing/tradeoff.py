"""Versioned metric-outcome constraints; warp is never relabeled as depth risk.

GT-derived outcomes are offline oracle/training targets. Deployment consumes
only predicted changes from pre-action evidence, not these measured outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from reliable_endo_gs.contracts.interventions import ACTION_NAMES as ACTION_NAMES
from reliable_endo_gs.contracts.interventions import COMPONENT_NAMES as COMPONENT_NAMES

# Declared numerical floors in the units of COMPONENT_NAMES, not utility weights.
SCALE_FLOORS = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001)

V2_CONFIG_FIELDS = (
    "utility_gain_cap",
    "component_risk_weight",
    "calibration_min_class_groups",
    "constraint_margin_weight",
)


@dataclass(frozen=True, slots=True)
class TradeoffConfig:
    """Pilot contract. Zero tolerance is a strict non-regression reference.

    Nonzero tolerances require a new config identity and development evidence.
    Predictive feasibility is not a guarantee of actual metric safety.
    """

    schema: str = "metric_action_tradeoff.v1"
    maximum_depth_mae_increase_mm: float = 0.0
    maximum_local_depth_mae_increase_mm: float = 0.0
    maximum_depth_rmse_increase_mm: float = 0.0
    maximum_local_depth_rmse_increase_mm: float = 0.0
    maximum_depth_p95_increase_mm: float = 0.0
    maximum_psnr_drop_db: float = 0.0
    maximum_ssim_drop: float = 0.0
    maximum_outside_rgb_increase: float = 0.0
    maximum_local_rgb_increase: float = 0.0
    maximum_coverage_drop: float = 0.0
    geometry_weight: float = 1.0
    local_geometry_weight: float = 0.25
    psnr_weight: float = 0.5
    ssim_weight: float = 0.5
    collateral_weight: float = 1.0
    local_render_weight: float = 0.25
    maximum_predicted_violation_probability: float = 0.1
    minimum_utility: float = 0.0
    comparison_epsilon: float = 1e-6
    ranking_epsilon: float = 1e-6
    ranking_weight: float = 0.25
    width: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    seed: int = 0
    bootstrap_repetitions: int = 200
    utility_gain_cap: float = 1.0
    component_risk_weight: float = 0.25
    calibration_min_class_groups: int = 2
    constraint_margin_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.schema not in ("metric_action_tradeoff.v1", "metric_action_tradeoff.v2"):
            raise ValueError("unsupported metric tradeoff schema")
        for name in ("width", "epochs", "bootstrap_repetitions", "calibration_min_class_groups"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name, value in asdict(self).items():
            if name == "schema":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.maximum_predicted_violation_probability < 1:
            raise ValueError("violation probability cutoff must lie in (0,1)")
        if self.learning_rate <= 0 or self.comparison_epsilon > 1e-5:
            raise ValueError("positive learning rate and a small numerical epsilon required")
        if self.geometry_weight <= 0 or self.psnr_weight + self.ssim_weight <= 0:
            raise ValueError("both geometry and render utility must remain active")
        if self.utility_gain_cap <= 0:
            raise ValueError("utility gain cap must be positive")
        if self.schema == "metric_action_tradeoff.v2" and (
            self.collateral_weight != 0 or self.local_render_weight != 0
        ):
            raise ValueError("v2 local/outside RGB are guards, not duplicate utility rewards")

    @property
    def minimum_changes(self) -> tuple[float, ...]:
        """Minimum permitted gains, in raw metric units; no scale coupling."""
        return tuple(
            -v
            for v in (
                self.maximum_depth_mae_increase_mm,
                self.maximum_local_depth_mae_increase_mm,
                self.maximum_depth_rmse_increase_mm,
                self.maximum_local_depth_rmse_increase_mm,
                self.maximum_depth_p95_increase_mm,
                self.maximum_psnr_drop_db,
                self.maximum_psnr_drop_db,
                self.maximum_ssim_drop,
                self.maximum_ssim_drop,
                self.maximum_outside_rgb_increase,
                self.maximum_local_rgb_increase,
                self.maximum_coverage_drop,
            )
        )

    @property
    def score_weights(self) -> tuple[float, ...]:
        """One score owner for training, oracle and prediction; tail metrics constrain."""
        if self.schema == "metric_action_tradeoff.v2":
            geometry = self.geometry_weight + self.local_geometry_weight
            render = self.psnr_weight + self.ssim_weight
            return (
                self.geometry_weight / geometry,
                self.local_geometry_weight / geometry,
                0.0,
                0.0,
                0.0,
                self.psnr_weight / (2 * render),
                self.psnr_weight / (2 * render),
                self.ssim_weight / (2 * render),
                self.ssim_weight / (2 * render),
                0.0,
                0.0,
                0.0,
            )
        return (
            self.geometry_weight,
            self.local_geometry_weight,
            0.0,
            0.0,
            0.0,
            self.psnr_weight / 2,
            self.psnr_weight / 2,
            self.ssim_weight / 2,
            self.ssim_weight / 2,
            self.collateral_weight,
            self.local_render_weight,
            0.0,
        )

    @property
    def config_hash(self) -> str:
        raw = asdict(self)
        if self.schema == "metric_action_tradeoff.v1":
            # Preserve historical table/checkpoint identities when adding v2 fields.
            for key in V2_CONFIG_FIELDS:
                raw.pop(key)
        return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()


def load_tradeoff_config(path: Path) -> TradeoffConfig:
    content = path.read_text(encoding="utf-8")
    # JSON's exponent-only floats (1e-6) are strings under PyYAML's YAML 1.1
    # resolver. Parse explicit JSON as JSON; never coerce arbitrary YAML strings.
    raw = json.loads(content) if content.lstrip().startswith("{") else yaml.safe_load(content)
    if not isinstance(raw, dict) or set(raw) - set(TradeoffConfig.__dataclass_fields__):
        raise ValueError("tradeoff config contains unknown fields")
    return TradeoffConfig(**raw)


def component_feasibility(changes: Sequence[float], config: TradeoffConfig) -> tuple[bool, ...]:
    """Per-metric raw-unit guard owner, shared by joint and auxiliary risk labels."""
    if len(changes) != len(COMPONENT_NAMES):
        return (False,) * len(COMPONENT_NAMES)
    return tuple(
        math.isfinite(v) and v >= bound - config.comparison_epsilon
        for v, bound in zip(changes, config.minimum_changes, strict=True)
    )


def feasible_changes(changes: Sequence[float], config: TradeoffConfig) -> bool:
    """Measured feasibility; missing/NaN metrics cannot certify safety."""
    return all(component_feasibility(changes, config))


def utility_coefficients(scales: Sequence[float], config: TradeoffConfig) -> tuple[float, ...]:
    """Common train-fitted scales across actions preserve component units."""
    if len(scales) != len(COMPONENT_NAMES) or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("tradeoff scales must be finite positive per-component values")
    return tuple(w / s for w, s in zip(config.score_weights, scales, strict=True))


def outcome_utility(
    changes: Sequence[float], scales: Sequence[float], config: TradeoffConfig
) -> float:
    if len(changes) != len(COMPONENT_NAMES) or not all(math.isfinite(v) for v in changes):
        raise ValueError("utility requires a complete finite metric vector")
    if config.schema == "metric_action_tradeoff.v1":
        return sum(
            v * w for v, w in zip(changes, utility_coefficients(scales, config), strict=True)
        )
    return float(outcome_utility_tensor(torch.tensor(changes, dtype=torch.float64), scales, config))


def outcome_utility_tensor(
    changes: torch.Tensor, scales: Sequence[float], config: TradeoffConfig
) -> torch.Tensor:
    """Differentiable canonical score for [...,12] raw-unit metric changes.

    v2 bounds each normalized gain before weighting: geometry and render each
    have a total weight of one. Local/outside RGB stay raw-unit constraints.
    Statistical scales remain train-only numerical references, not safety limits.
    """
    coefficients = utility_coefficients(scales, config)
    if changes.shape[-1] != len(COMPONENT_NAMES):
        raise ValueError("utility tensor must end in the component dimension")
    if config.schema == "metric_action_tradeoff.v1":
        return changes @ changes.new_tensor(coefficients)
    normalized = changes / changes.new_tensor(scales)
    # Smooth bounded gain x/(1+|x|/cap): monotone, signed, no hard-clamp
    # ranking ties or zero training gradients at ordinary large gains.
    gains = normalized / (1 + normalized.abs() / config.utility_gain_cap)
    return gains @ changes.new_tensor(config.score_weights)


def constraint_margins_tensor(
    changes: torch.Tensor, scales: Sequence[float], config: TradeoffConfig
) -> torch.Tensor:
    """Normalized signed distance to each raw-unit guard for [...,12] changes.

    Positive means feasible, zero is the inclusive boundary. Used to supervise
    predicted changes near guard crossings, without changing the physical limits.
    """
    utility_coefficients(scales, config)
    if changes.shape[-1] != len(COMPONENT_NAMES):
        raise ValueError("constraint margins must end in the component dimension")
    return (
        changes - changes.new_tensor(config.minimum_changes) + config.comparison_epsilon
    ) / changes.new_tensor(scales)


@dataclass(frozen=True, slots=True)
class TradeoffDecision:
    region_index: int | None
    action: str
    utility: float
    estimated_action_ms: float


def select_tradeoff(
    changes: Sequence[Sequence[Sequence[float]]],
    violations: Sequence[Sequence[float]],
    costs_ms: Sequence[Sequence[float]],
    valid: Sequence[Sequence[bool]],
    *,
    remaining_ms: float,
    scales: Sequence[float],
    config: TradeoffConfig,
) -> TradeoffDecision:
    """One [R,A,C] decision, shared by measured oracle and predictive pilot.

    STOP is a measured zero-change alternative. No finite feasible positive
    utility, no available observation, or no remaining budget means STOP.
    """
    if not math.isfinite(remaining_ms) or remaining_ms < 0:
        raise ValueError("remaining budget must be finite nonnegative")
    utility_coefficients(scales, config)
    if not len(changes) == len(violations) == len(costs_ms) == len(valid):
        raise ValueError("tradeoff candidates must share region count")
    best = TradeoffDecision(None, "STOP", 0.0, 0.0)
    for ri, (row, risks, costs, eligible) in enumerate(
        zip(changes, violations, costs_ms, valid, strict=True)
    ):
        if not len(row) == len(risks) == len(costs) == len(eligible) == len(ACTION_NAMES):
            raise ValueError("tradeoff candidates require the pinned action set")
        for ai, (values, risk, cost, ok) in enumerate(
            zip(row, risks, costs, eligible, strict=True)
        ):
            if not math.isfinite(risk) or not 0 <= risk <= 1:
                raise ValueError("invalid violation probability")
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("invalid action cost")
            if (
                not ok
                or not 0 < cost <= remaining_ms
                or risk > config.maximum_predicted_violation_probability
                or not feasible_changes(values, config)
            ):
                continue
            score = outcome_utility(values, scales, config)
            if score > max(best.utility, config.minimum_utility) + config.comparison_epsilon:
                best = TradeoffDecision(ri, ACTION_NAMES[ai], score, cost)
    return best
