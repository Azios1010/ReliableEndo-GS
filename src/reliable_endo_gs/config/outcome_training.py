"""Opt-in predictor training controls, separate from physical policy semantics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class OutcomeTrainingConfig:
    """Bounded grouped early stopping; no threshold or action tuning."""

    schema: str = "outcome_training.v1"
    max_steps: int = 1000
    min_steps: int = 100
    validate_every: int = 25
    patience: int = 6
    min_delta: float = 0.001
    weight_decay: float = 0.01
    dropout: float = 0.1
    risk_ranking_weight: float = 0.0
    independent_encoder_stopping: bool = False
    calibration_method: str = "grid"
    calibration_regularization: float = 0.01
    risk_feature_mode: str = "raw"

    def __post_init__(self) -> None:
        if self.schema not in ("outcome_training.v1", "outcome_training.v2"):
            raise ValueError("unsupported outcome training policy")
        for name in ("max_steps", "min_steps", "validate_every", "patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_steps > self.max_steps:
            raise ValueError("minimum steps cannot exceed the bounded maximum")
        for name in (
            "min_delta",
            "weight_decay",
            "dropout",
            "risk_ranking_weight",
            "calibration_regularization",
        ):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.dropout >= 1:
            raise ValueError("dropout must be below one")
        if type(self.independent_encoder_stopping) is not bool:
            raise ValueError("independent encoder stopping must be boolean")
        if self.calibration_method not in ("grid", "platt"):
            raise ValueError("unsupported risk calibration method")
        if self.calibration_regularization <= 0:
            raise ValueError("risk calibration regularization must be positive")
        if self.risk_feature_mode not in ("raw", "predicted_margins"):
            raise ValueError("unsupported risk feature mode")
        if self.risk_feature_mode == "predicted_margins" and not self.independent_encoder_stopping:
            raise ValueError("conditional risk requires independent encoder selection")
        if self.schema == "outcome_training.v1" and (
            self.risk_ranking_weight != 0
            or self.independent_encoder_stopping
            or self.calibration_method != "grid"
            or self.calibration_regularization != 0.01
            or self.risk_feature_mode != "raw"
        ):
            raise ValueError("risk discrimination changes require outcome_training.v2")

    @property
    def config_hash(self) -> str:
        raw = asdict(self)
        if self.risk_feature_mode == "raw":
            raw.pop("risk_feature_mode")
        if self.schema == "outcome_training.v1":
            for name in (
                "risk_ranking_weight",
                "independent_encoder_stopping",
                "calibration_method",
                "calibration_regularization",
            ):
                raw.pop(name)
        return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()


def load_outcome_training_config(path: Path) -> OutcomeTrainingConfig:
    content = path.read_text(encoding="utf-8")
    raw = json.loads(content) if content.lstrip().startswith("{") else yaml.safe_load(content)
    if not isinstance(raw, dict) or set(raw) - set(OutcomeTrainingConfig.__dataclass_fields__):
        raise ValueError("invalid outcome training controls")
    return OutcomeTrainingConfig(**raw)
