"""Strict, portable settings for the experimental frozen-A04 routing path."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    schema: str = "frozen_a04_action_selector.v1"
    tile_hw: tuple[int, int] = (256, 320)
    halo: int = 4
    width: int = 32
    max_disparity_px: float = 4.0
    composition_alpha_epsilon: float = 0.0000001
    learning_rate: float = 0.001
    cost_weight_per_ms: float = 0.000001
    budget_ms: float = 1000.0
    action_epochs: int = 1
    selector_epochs: int = 1
    checkpoint_every: int = 100
    max_consecutive_skips: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        if self.schema != "frozen_a04_action_selector.v1":
            raise ValueError("unsupported routing schema")
        if len(self.tile_hw) != 2 or any(type(x) is not int or x <= 0 for x in self.tile_hw):
            raise ValueError("tile_hw must be two positive integers")
        for name in (
            "width",
            "action_epochs",
            "selector_epochs",
            "checkpoint_every",
            "max_consecutive_skips",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.halo) is not int
            or self.halo < 0
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise ValueError("halo and seed must be nonnegative integers")
        for name in (
            "max_disparity_px",
            "composition_alpha_epsilon",
            "learning_rate",
            "budget_ms",
            "cost_weight_per_ms",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.max_disparity_px == 0 or self.learning_rate == 0:
            raise ValueError("disparity bound and learning rate must be positive")
        if not 0 < self.composition_alpha_epsilon < 0.5:
            raise ValueError("composition_alpha_epsilon must lie in (0,0.5)")

    @property
    def config_hash(self) -> str:
        """Stable identity of all resolved scientific/operational settings."""
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def load_routing_config(path: Path) -> RoutingConfig:
    """Reject unknown keys; no absolute data paths are stored in config."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - set(RoutingConfig.__dataclass_fields__):
        raise ValueError("routing config must contain only declared fields")
    if "tile_hw" in raw:
        raw["tile_hw"] = tuple(raw["tile_hw"])
    return RoutingConfig(**raw)
