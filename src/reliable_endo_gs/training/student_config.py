"""Separate strict configuration for the authorized student/joint experiment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class StudentLearningConfig:
    schema: str = "student_joint_learning.v1"
    width: int = 32
    bins: int = 32
    minimum_effective_px: float = 0.25
    student_epochs: int = 1
    joint_epochs: int = 1
    learning_rate: float = 0.001
    reconstruction_weight: float = 1.0
    kd_weight: float = 1.0
    repair_weight: float = 1.0
    checkpoint_every: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        if self.schema != "student_joint_learning.v1":
            raise ValueError("unsupported student schema")
        for name in ("width", "bins", "checkpoint_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("student_epochs", "joint_epochs", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.student_epochs + self.joint_epochs < 1:
            raise ValueError("at least one student or joint epoch is required")
        for name in (
            "minimum_effective_px",
            "learning_rate",
            "reconstruction_weight",
            "kd_weight",
            "repair_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if min(self.minimum_effective_px, self.learning_rate, self.reconstruction_weight) <= 0:
            raise ValueError(
                "minimum disparity, learning rate and reconstruction weight must be positive"
            )
        if self.joint_epochs and self.repair_weight <= 0:
            raise ValueError("joint learning requires positive repair weight")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def load_student_config(path: Path) -> StudentLearningConfig:
    """Fail on unknown fields rather than silently changing the experiment."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - set(StudentLearningConfig.__dataclass_fields__):
        raise ValueError("student config must contain only declared fields")
    return StudentLearningConfig(**raw)
