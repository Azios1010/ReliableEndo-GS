"""Predeclared development comparison; no outcome-dependent default choices."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class SelectorStudyConfig:
    schema: str = "frozen_selector_study.v1"
    fit_seeds: tuple[int, ...] = (0, 1, 2)
    utility_modes: tuple[str, ...] = ("geometry", "render_psnr")
    harm_budgets: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)
    cutoff_candidates: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 0.95)
    bootstrap_repetitions: int = 2000
    bootstrap_seed: int = 0
    minimum_unique_win_keyframes: int = 2
    minimum_oracle_headroom: float = 0.0
    minimum_selected_keyframes: int = 2
    minimum_selected_observations: int = 8
    repair_budget_ms: float = 1000.0
    # dB/ms. Zero preserves gain-only ranking until a latency tradeoff is declared.
    render_cost_weight_per_ms: float = 0.0
    grid_tile_hw: tuple[int, int] = (32, 32)
    max_regions: int = 4
    max_pixels: int = 4096
    max_primitives: int = 1024
    max_read_pixels: int = 65536

    def __post_init__(self) -> None:
        for name in (
            "fit_seeds",
            "utility_modes",
            "harm_budgets",
            "cutoff_candidates",
            "grid_tile_hw",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.schema != "frozen_selector_study.v1":
            raise ValueError("unknown selector study schema")
        if (
            not self.fit_seeds
            or len(set(self.fit_seeds)) != len(self.fit_seeds)
            or any(type(s) is not int or s < 0 for s in self.fit_seeds)
        ):
            raise ValueError("invalid study fit seeds")
        if (
            not self.utility_modes
            or len(set(self.utility_modes)) != len(self.utility_modes)
            or set(self.utility_modes) - {"geometry", "render_psnr"}
        ):
            raise ValueError("unknown utility arm")
        for name in ("harm_budgets", "cutoff_candidates"):
            values = getattr(self, name)
            if (
                not values
                or tuple(sorted(set(values))) != values
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1
                    for v in values
                )
            ):
                raise ValueError("budgets and cutoffs must be ordered unique probabilities")
        for name in (
            "bootstrap_repetitions",
            "minimum_unique_win_keyframes",
            "minimum_selected_keyframes",
            "minimum_selected_observations",
            "max_regions",
            "max_pixels",
            "max_primitives",
            "max_read_pixels",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("invalid study count/bound: " + name)
        if (
            self.minimum_unique_win_keyframes < 2
            or self.minimum_selected_keyframes < 2
            or self.minimum_selected_observations < 8
        ):
            raise ValueError(
                "operating points require two keyframes and eight selected observations"
            )
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise ValueError("invalid bootstrap seed")
        if (
            any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in (
                    self.minimum_oracle_headroom,
                    self.repair_budget_ms,
                    self.render_cost_weight_per_ms,
                )
            )
            or self.repair_budget_ms <= 0
        ):
            raise ValueError("invalid study headroom/budget")
        if len(self.grid_tile_hw) != 2 or any(
            type(v) is not int or v < 1 for v in self.grid_tile_hw
        ):
            raise ValueError("invalid study grid")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def load_selector_study_config(path: Path) -> SelectorStudyConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) - set(SelectorStudyConfig.__dataclass_fields__):
        raise ValueError("unknown selector study controls")
    return SelectorStudyConfig(**raw)
