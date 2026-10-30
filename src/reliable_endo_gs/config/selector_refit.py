"""Declared TRAIN-only selector refitting and calibration controls."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class SelectorRefitConfig:
    schema: str = "selector_refit.v1"
    seed: int = 0
    # Legacy seed is the fallback; study replicates vary ONLY fit_seed.
    partition_seed: int | None = None
    fit_seed: int | None = None
    fit_fraction: float = 0.6
    validation_fraction: float = 0.15
    max_steps: int = 1000
    min_steps: int = 50
    validate_every: int = 25
    patience: int = 6
    min_delta: float = 0.0001
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    calibration_regularization: float = 0.01
    gain_regularization: float = 0.05
    calibration_min_class_groups: int = 2
    calibration_min_observations: int = 8
    partition_policy: str = "global_keyframe_hash"
    max_training_samples: int = 112
    max_development_samples: int = 56
    collection_shard_size: int = 64
    validation_guard_mode: str = "component_minima"

    def __post_init__(self) -> None:
        if self.schema != "selector_refit.v1" or type(self.seed) is not int or self.seed < 0:
            raise ValueError("invalid selector refit identity")
        for value in (self.partition_seed, self.fit_seed):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("partition/fit seed must be a nonnegative integer")
        if self.validation_guard_mode not in ("component_minima", "action_objectives"):
            raise ValueError("unsupported selector validation guard mode")
        for key in (
            "max_steps",
            "min_steps",
            "validate_every",
            "patience",
            "calibration_min_class_groups",
            "calibration_min_observations",
            "max_training_samples",
            "max_development_samples",
            "collection_shard_size",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not self.min_steps <= self.max_steps <= 2000:
            raise ValueError("selector refit must be bounded to at most 2000 steps")
        if self.calibration_min_class_groups < 2 or self.calibration_min_observations < 8:
            raise ValueError("calibration needs at least two keyframes per class and eight regions")
        if (
            self.partition_policy not in ("global_keyframe_hash", "stratified_keyframe_hash")
            or self.collection_shard_size > 256
        ):
            raise ValueError("invalid cohort partition or collection bounds")
        for key in (
            "fit_fraction",
            "validation_fraction",
            "min_delta",
            "learning_rate",
            "weight_decay",
            "calibration_regularization",
            "gain_regularization",
        ):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {key}")
        if (
            not 0 < self.fit_fraction < 1
            or not 0 < self.validation_fraction < 1 - self.fit_fraction
        ):
            raise ValueError("fit/validation must leave a nonempty calibration fraction")
        if min(self.learning_rate, self.calibration_regularization, self.gain_regularization) <= 0:
            raise ValueError("learning rate/calibration regularization must be positive")

    @property
    def config_hash(self) -> str:
        raw = asdict(self)
        for key in (
            "partition_policy",
            "max_training_samples",
            "max_development_samples",
            "collection_shard_size",
            "validation_guard_mode",
            "partition_seed",
            "fit_seed",
        ):
            if raw[key] == self.__dataclass_fields__[key].default:
                raw.pop(key)
        return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()

    @property
    def resolved_partition_seed(self) -> int:
        return self.seed if self.partition_seed is None else self.partition_seed

    @property
    def resolved_fit_seed(self) -> int:
        return self.seed if self.fit_seed is None else self.fit_seed


def load_selector_refit_config(path: Path) -> SelectorRefitConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) - set(SelectorRefitConfig.__dataclass_fields__):
        raise ValueError("unknown selector refit controls")
    return SelectorRefitConfig(**raw)


def partition_selector_training(
    ids: tuple[str, ...], config: SelectorRefitConfig
) -> dict[str, Any]:
    """Hash whole TRAIN keyframes BEFORE reading labels; preserve parent roles."""
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(
            s.split("/")[0] not in {f"dataset_{i}" for i in range(1, 8)} or len(s.split("/")) != 3
            for s in ids
        )
    ):
        raise ValueError("partition requires unique original-SCARED TRAIN identities")
    keys = sorted(
        {s.rsplit("/", 1)[0] for s in ids},
        key=lambda k: hashlib.sha256(f"{config.resolved_partition_seed}:{k}".encode()).hexdigest(),
    )
    nfit = int(len(keys) * config.fit_fraction)
    nval = int(len(keys) * config.validation_fraction)
    groups = (keys[:nfit], keys[nfit : nfit + nval], keys[nfit + nval :])
    if config.partition_policy == "stratified_keyframe_hash":
        fit_keys = []
        validation_keys = []
        calibration_keys = []
        # Derive folds from the sealed manifest, including an explicit train_1236 subset.
        for fold in sorted({s.split("/")[0] for s in ids}):
            candidates = [k for k in keys if k.startswith(fold + "/")]
            if len(candidates) < 3:
                raise ValueError("stratified roles require three keyframes in every training fold")
            validation_keys.append(candidates[0])
            calibration_keys.append(candidates[1])
            fit_keys.extend(candidates[2:])
        groups = (fit_keys, validation_keys, calibration_keys)
    if min(map(len, groups)) < 2:
        raise ValueError("need at least two whole keyframes in each internal TRAIN role")
    report: dict[str, Any] = {
        "policy": "sha256_whole_TRAIN_keyframes_before_labels"
        if config.partition_policy == "global_keyframe_hash"
        else "sha256_within_each_fold_whole_TRAIN_keyframes_before_labels",
        "seed": config.resolved_partition_seed,
        "parent_roles_changed": False,
    }
    for name, group in zip(("fit", "validation", "calibration"), groups, strict=True):
        report[name + "_keyframes"] = group
        report[name + "_sample_ids"] = [s for s in ids if s.rsplit("/", 1)[0] in group]
    report["partition_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode()
    ).hexdigest()
    return report
