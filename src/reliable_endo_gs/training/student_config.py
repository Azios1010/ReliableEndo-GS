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
    gt_weight: float = 0.0
    repair_weight: float = 1.0
    checkpoint_every: int = 100
    seed: int = 0
    model_lr_schedule: str = "constant"
    model_lr_decay_epochs: int = 10
    model_lr_min_ratio: float = 0.1
    stereo_architecture: str = "compact"

    def __post_init__(self) -> None:
        if self.schema != "student_joint_learning.v1":
            raise ValueError("unsupported student schema")
        if self.stereo_architecture not in ("compact", "multiscale"):
            raise ValueError("unsupported stereo student architecture")
        for name in ("width", "bins", "checkpoint_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.stereo_architecture == "multiscale" and self.width < 8:
            raise ValueError("multiscale stereo width must be at least 8")
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
            "gt_weight",
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
        if self.model_lr_schedule not in ("constant", "cosine_epochs"):
            raise ValueError("unsupported model learning-rate schedule")
        if type(self.model_lr_decay_epochs) is not int or self.model_lr_decay_epochs < 2:
            raise ValueError("model_lr_decay_epochs must be an integer >= 2")
        ratio = self.model_lr_min_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or not 0 < ratio <= 1
        ):
            raise ValueError("model_lr_min_ratio must be finite in (0, 1]")
        if self.model_lr_schedule == "constant" and (
            self.model_lr_decay_epochs != 10 or self.model_lr_min_ratio != 0.1
        ):
            raise ValueError("constant model LR cannot carry unused decay settings")

    def to_dict(self) -> dict[str, object]:
        """Omit neutral schedule fields to preserve historical checkpoint identities."""
        values = asdict(self)
        if not self.gt_weight:
            values.pop("gt_weight")  # GT-disparity supervision is opt-in; historical identities stay valid
        if self.stereo_architecture == "compact":
            values.pop("stereo_architecture")
        if self.model_lr_schedule == "constant":
            for name in ("model_lr_schedule", "model_lr_decay_epochs", "model_lr_min_ratio"):
                values.pop(name)
        return values

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def load_student_config(path: Path) -> StudentLearningConfig:
    """Fail on unknown fields rather than silently changing the experiment."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - set(StudentLearningConfig.__dataclass_fields__):
        raise ValueError("student config must contain only declared fields")
    return StudentLearningConfig(**raw)
