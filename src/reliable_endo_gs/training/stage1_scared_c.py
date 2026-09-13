"""Stage1 SCARED-C corrected-video training entry point.

This adapter keeps the pinned upstream stereo architecture and loss while
owning the SCARED-C split, cache, deterministic frame partition, and training
artifacts.  The bounded smoke remains available for legacy mean configuration;
the full-data configuration opts into its own explicit frame split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from reliable_endo_gs.baseline.upstream import import_upstream_module
from reliable_endo_gs.data.scared_c import (
    CORRECTED_VIDEO_MODE,
    build_scared_c_index,
    materialize_scared_c_rectified_rgb,
)
from reliable_endo_gs.data.scared_c_stage1_cache import (
    STAGE1_MEAN_TRAIN,
    STAGE1_MEAN_VALIDATION,
    Stage1GTCache,
    build_stage1_gt_cache,
)
from reliable_endo_gs.data.scared_c_stage1_quality import (
    Stage1GTQualityManifest,
    audit_stage1_gt_cache,
    load_stage1_gt_quality_manifest,
    write_stage1_gt_quality_manifest,
)
from reliable_endo_gs.data.scared_c_stage1_split import (
    STAGE1_FORBIDDEN_DATASET_IDS,
    STAGE1_TRAIN_DATASET_IDS,
    Stage1FrameSplit,
    Stage1FrameSplitError,
    load_stage1_frame_split,
    make_stage1_frame_split,
    normalize_dataset_ids,
    normalize_keyframe_entries,
    normalize_sample_ids,
    select_stage1_records,
    write_stage1_frame_split,
)
from reliable_endo_gs.data.scared_c_stereo import RectificationCache, StackedStereoVideo

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = REPOSITORY_ROOT / "third_party" / "endo_e2e_gs"
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "training" / "stage1_scared_c_mean_v1.yaml"
DEFAULT_CACHE_ROOT = Path(r"E:\runtime-tmp\reliable-endo-gs\scared-c-stage1-cache")
DEFAULT_FULL_FRAME_SPLIT_PATH = Path(
    r"E:\runtime-tmp\reliable-endo-gs\stage1-scared-c-full-v1\split_manifest.json"
)
STAGE1_VALIDATION_STEPS = (
    1000,
    2000,
    3000,
    5000,
    7000,
    10000,
    15000,
    20000,
    30000,
    40000,
    50000,
    60000,
)


class Stage1ScaredCConfigError(ValueError):
    """Raised when a Stage1 SCARED-C config is ambiguous or unsafe."""


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Stage1ScaredCConfigError(f"{name} must be a mapping")
    return value


def _tuple_of_strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise Stage1ScaredCConfigError(f"{name} must be a list of strings")
    return tuple(value)


def _tuple_of_positive_ints(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value
    ):
        raise Stage1ScaredCConfigError(f"{name} must be a list of positive integers")
    return tuple(value)


def _expand_runtime_path(value: object, *, name: str, default: Path | None = None) -> Path:
    if value is None:
        if default is None:
            raise Stage1ScaredCConfigError(f"{name} is required")
        return default
    if not isinstance(value, str) or not value.strip():
        raise Stage1ScaredCConfigError(f"{name} must be a non-empty path string")
    text = value.strip()

    def replace(match: re.Match[str]) -> str:
        variable = match.group(1)
        fallback = match.group(2)
        resolved = os.environ.get(variable)
        if resolved is None or not resolved.strip():
            if fallback is None:
                raise Stage1ScaredCConfigError(f"environment variable {variable} is not set")
            return fallback
        return resolved

    text = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}", replace, text)
    path = Path(text)
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _parse_optional_checkpoint(value: object) -> Path | None:
    """Treat YAML null and legacy textual nulls as a real Python ``None``."""

    if value is None:
        return None
    if isinstance(value, str) and value.strip().casefold() in {"", "none", "null", "~"}:
        return None
    if not isinstance(value, str):
        raise Stage1ScaredCConfigError("checkpoint must be null or a path string")
    return _expand_runtime_path(value, name="restore_ckpt")


@dataclass(frozen=True, slots=True)
class Stage1ScaredCConfig:
    """Resolved declarative Stage1 configuration."""

    name: str
    dataset: str
    mode: str
    data_root: Path
    split_path: Path
    split_active: bool
    cache_root: Path
    train_sequences: tuple[str, ...]
    validation_sequences: tuple[str, ...]
    initialization: str
    from_scratch: bool
    restore_ckpt: Path | None
    batch_size: int
    num_steps: int
    learning_rate: float
    weight_decay: float
    train_iters: int
    val_iters: int
    mixed_precision: bool
    correlation_implementation: str
    correlation_levels: int
    correlation_radius: int
    n_downsample: int
    n_gru_layers: int
    encoder_dims: tuple[int, ...]
    hidden_dims: tuple[int, ...]
    gradient_clip: float
    scheduler_pct_start: float
    scheduler_anneal_strategy: str
    seed: int
    output_dir: Path
    training_mode: str = "legacy_mean"
    train_dataset_ids: tuple[str, ...] = STAGE1_TRAIN_DATASET_IDS
    train_keyframe_entries: tuple[str, ...] = ()
    train_sample_ids: tuple[str, ...] = ()
    frame_split_path: Path = DEFAULT_FULL_FRAME_SPLIT_PATH
    validation_fraction: float = 0.05
    validation_block_policy: str = "tail"
    validation_rounding: str = "nearest_half_up"
    min_validation_frames_per_keyframe: int = 1
    transfer_keyframe_entries: tuple[str, ...] = ()
    excluded_dataset_ids: tuple[str, ...] = STAGE1_FORBIDDEN_DATASET_IDS
    expected_keyframe_count: int | None = None
    expected_total_samples: int | None = None
    expected_train_samples: int | None = None
    expected_validation_samples: int | None = None
    validation_steps: tuple[int, ...] = STAGE1_VALIDATION_STEPS
    gt_quality_policy: str = "reject"

    @property
    def gt_quality_manifest_path(self) -> Path:
        return self.frame_split_path.with_name("gt_quality_manifest.json")

    @property
    def dataset_root(self) -> Path:
        return self.data_root / self.dataset

    @property
    def is_full_data(self) -> bool:
        return self.training_mode == "full_data"

    def require_safe_mode(self, *, steps: int) -> None:
        if steps < 1:
            raise Stage1ScaredCConfigError("steps must be positive")
        if not self.from_scratch or self.initialization != "scratch":
            raise Stage1ScaredCConfigError(
                "Stage1 SCARED-C requires explicit scratch initialization"
            )
        if self.restore_ckpt is not None:
            raise Stage1ScaredCConfigError(
                f"scratch mode cannot carry restore_ckpt={self.restore_ckpt}"
            )
        if self.is_full_data:
            if not (1 <= steps <= 100 or steps == 60000):
                raise Stage1ScaredCConfigError(
                    "full-data Stage1 permits a 1..100-step smoke or the fixed 60000-step run"
                )
            if steps == 60000 and self.num_steps != 60000:
                raise Stage1ScaredCConfigError(
                    "full-data Stage1 must retain the fixed 60000-step recipe"
                )
            return
        if steps != 3 and not self.split_active:
            raise Stage1ScaredCConfigError(
                "the Stage1 mean split is inactive; only the bounded three-step smoke is allowed"
            )


def _manifest_is_active(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage1ScaredCConfigError(f"unable to read split manifest {path}: {error}") from error
    return bool(payload.get("active")) if isinstance(payload, dict) else False


def load_stage1_scared_c_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Stage1ScaredCConfig:
    """Load and validate a legacy smoke or explicit full-data Stage1 config."""

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (REPOSITORY_ROOT / config_path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise Stage1ScaredCConfigError(
            f"unable to read Stage1 config {config_path}: {error}"
        ) from error
    root = _require_mapping(raw, "Stage1 config")
    dataset = str(root.get("dataset", ""))
    mode = str(root.get("mode", ""))
    if dataset != "scared_c" or mode != CORRECTED_VIDEO_MODE:
        raise Stage1ScaredCConfigError(
            "Stage1 config must select dataset=scared_c and corrected_video"
        )

    initialization = _require_mapping(root.get("initialization"), "initialization")
    init_mode = str(initialization.get("mode", ""))
    from_scratch = initialization.get("from_scratch") is True
    restore_ckpt = _parse_optional_checkpoint(initialization.get("restore_ckpt"))
    if init_mode != "scratch" or not from_scratch or restore_ckpt is not None:
        raise Stage1ScaredCConfigError(
            "initialization must be mode=scratch, from_scratch=true, restore_ckpt=null"
        )

    split_path = _expand_runtime_path(root.get("split_manifest"), name="split_manifest")
    try:
        split_payload = _require_mapping(
            json.loads(split_path.read_text(encoding="utf-8")), "split manifest"
        )
    except (OSError, json.JSONDecodeError) as error:
        raise Stage1ScaredCConfigError(
            f"unable to read split manifest {split_path}: {error}"
        ) from error
    split_active = split_payload.get("active")
    if not isinstance(split_active, bool):
        raise Stage1ScaredCConfigError("split manifest active must be boolean")
    if split_active:
        raise Stage1ScaredCConfigError(
            "the Stage1 role guard manifest must remain inactive; frame split activation is separate"
        )
    roles = _require_mapping(split_payload.get("roles"), "split roles")
    final_guard_ids: list[str] = []
    for role_name in ("FINAL_UNTOUCHED", "FINAL_STATIC_UNTOUCHED"):
        role_values = roles.get(role_name, [])
        final_guard_ids.extend(_tuple_of_strings(role_values, f"split roles.{role_name}"))
    if not any(role_id.startswith("6_") for role_id in final_guard_ids):
        raise Stage1ScaredCConfigError("split roles must retain a dataset_6 final guard")
    training_mode = str(root.get("training_mode", "legacy_mean"))
    if training_mode not in {"legacy_mean", "full_data"}:
        raise Stage1ScaredCConfigError("training_mode must be legacy_mean or full_data")

    train_sequences = _tuple_of_strings(
        root.get("train_sequences", [])
        if training_mode == "full_data"
        else root.get("train_sequences"),
        "train_sequences",
    )
    validation_sequences = _tuple_of_strings(
        root.get("validation_sequences", [])
        if training_mode == "full_data"
        else root.get("validation_sequences"),
        "validation_sequences",
    )
    train_dataset_ids: tuple[str, ...] = STAGE1_TRAIN_DATASET_IDS
    train_keyframe_entries: tuple[str, ...] = ()
    train_sample_ids: tuple[str, ...] = ()
    transfer_keyframe_entries: tuple[str, ...] = ()
    excluded_dataset_ids: tuple[str, ...] = STAGE1_FORBIDDEN_DATASET_IDS
    validation_fraction = 0.05
    validation_block_policy = "tail"
    validation_rounding = "nearest_half_up"
    min_validation_frames_per_keyframe = 1
    expected_keyframe_count: int | None = None
    expected_total_samples: int | None = None
    expected_train_samples: int | None = None
    expected_validation_samples: int | None = None
    if training_mode == "legacy_mean":
        if train_sequences != STAGE1_MEAN_TRAIN or validation_sequences != STAGE1_MEAN_VALIDATION:
            raise Stage1ScaredCConfigError(
                "legacy Stage1 train/validation roles do not match stage1_mean_v1"
            )
        if tuple(roles.get("MEAN_TRAIN", ())) != STAGE1_MEAN_TRAIN:
            raise Stage1ScaredCConfigError(
                "split MEAN_TRAIN does not match the approved sequence list"
            )
        if tuple(roles.get("MEAN_VALIDATION", ())) != STAGE1_MEAN_VALIDATION:
            raise Stage1ScaredCConfigError(
                "split MEAN_VALIDATION does not match the approved sequence list"
            )
    else:
        if train_sequences or validation_sequences:
            raise Stage1ScaredCConfigError(
                "full-data Stage1 uses training_selection and an internal frame split, "
                "not sequence-level train/validation lists"
            )
        selection = _require_mapping(root.get("training_selection"), "training_selection")
        raw_dataset_ids = selection.get("dataset_ids", [])
        raw_keyframe_entries = selection.get("keyframe_entries", [])
        raw_sample_ids = selection.get("sample_ids", [])
        try:
            train_dataset_ids = normalize_dataset_ids(raw_dataset_ids) if raw_dataset_ids else ()
            train_keyframe_entries = (
                normalize_keyframe_entries(raw_keyframe_entries) if raw_keyframe_entries else ()
            )
            train_sample_ids = normalize_sample_ids(raw_sample_ids) if raw_sample_ids else ()
        except (Stage1FrameSplitError, TypeError) as error:
            raise Stage1ScaredCConfigError(str(error)) from error
        selector_count = sum(
            bool(value) for value in (train_dataset_ids, train_keyframe_entries, train_sample_ids)
        )
        if selector_count != 1 or train_dataset_ids != STAGE1_TRAIN_DATASET_IDS:
            raise Stage1ScaredCConfigError(
                "full-data training_selection must select exactly dataset_1, dataset_2, dataset_3"
            )
        raw_validation = _require_mapping(root.get("internal_validation"), "internal_validation")
        try:
            validation_fraction = float(raw_validation.get("fraction", 0.05))
            min_validation_frames_per_keyframe = int(
                raw_validation.get("min_frames_per_keyframe", 1)
            )
        except (TypeError, ValueError) as error:
            raise Stage1ScaredCConfigError(
                "internal_validation fraction and min_frames_per_keyframe must be numeric"
            ) from error
        validation_block_policy = str(raw_validation.get("block_policy", "tail"))
        validation_rounding = str(raw_validation.get("rounding", "nearest_half_up"))
        if (
            not np.isfinite(validation_fraction)
            or not 0.0 < validation_fraction < 1.0
            or validation_block_policy != "tail"
            or validation_rounding != "nearest_half_up"
            or min_validation_frames_per_keyframe < 1
        ):
            raise Stage1ScaredCConfigError(
                "internal_validation must use fraction in (0,1), tail blocks, "
                "nearest_half_up rounding, and a positive minimum"
            )
        transfer_keyframe_entries = normalize_keyframe_entries(
            _tuple_of_strings(
                root.get("transfer_keyframe_entries", ["dataset_7/keyframe_2"]),
                "transfer_keyframe_entries",
            ),
            name="transfer_keyframe_entries",
        )
        if not transfer_keyframe_entries or any(
            not entry.startswith("dataset_7/") for entry in transfer_keyframe_entries
        ):
            raise Stage1ScaredCConfigError(
                "transfer_keyframe_entries must be non-empty and dataset_7-only"
            )
        excluded_dataset_ids = normalize_dataset_ids(
            _tuple_of_strings(
                root.get("excluded_dataset_ids", list(STAGE1_FORBIDDEN_DATASET_IDS)),
                "excluded_dataset_ids",
            ),
            name="excluded_dataset_ids",
        )
        if not set(STAGE1_FORBIDDEN_DATASET_IDS) <= set(excluded_dataset_ids):
            raise Stage1ScaredCConfigError(
                "excluded_dataset_ids must include dataset_6 and dataset_7"
            )
        expected = _require_mapping(root.get("expected_counts"), "expected_counts")

        def expected_positive(name: str) -> int:
            value = expected.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise Stage1ScaredCConfigError(f"expected_counts.{name} must be a positive integer")
            return value

        expected_keyframe_count = expected_positive("corrected_keyframes")
        expected_total_samples = expected_positive("total_samples")
        expected_train_samples = expected_positive("train_samples")
        expected_validation_samples = expected_positive("validation_samples")

    raft = _require_mapping(root.get("raft"), "raft")
    encoder_dims_raw = raft.get("encoder_dims")
    hidden_dims_raw = raft.get("hidden_dims")
    if not isinstance(encoder_dims_raw, list) or not all(
        isinstance(v, int) for v in encoder_dims_raw
    ):
        raise Stage1ScaredCConfigError("raft.encoder_dims must be a list of integers")
    if not isinstance(hidden_dims_raw, list) or not all(
        isinstance(v, int) for v in hidden_dims_raw
    ):
        raise Stage1ScaredCConfigError("raft.hidden_dims must be a list of integers")
    data_root = _expand_runtime_path(
        root.get("data_root"),
        name="data_root",
        default=REPOSITORY_ROOT / "data",
    )
    cache_root = _expand_runtime_path(
        root.get("cache_root"), name="cache_root", default=DEFAULT_CACHE_ROOT
    )
    output_dir = _expand_runtime_path(
        root.get("output_dir"),
        name="output_dir",
        default=REPOSITORY_ROOT / "outputs" / "stage1_scared_c",
    )
    frame_split_path = _expand_runtime_path(
        root.get("frame_split_manifest", root.get("frame_split_output")),
        name="frame_split_manifest",
        default=DEFAULT_FULL_FRAME_SPLIT_PATH,
    )
    validation_steps = _tuple_of_positive_ints(
        root.get("validation_steps", list(STAGE1_VALIDATION_STEPS)), "validation_steps"
    )
    gt_quality_policy = str(root.get("gt_quality_policy", "reject"))
    if gt_quality_policy not in ("reject", "exclude_unusable"):
        raise Stage1ScaredCConfigError("gt_quality_policy must be reject or exclude_unusable")
    if gt_quality_policy != "reject" and training_mode != "full_data":
        raise Stage1ScaredCConfigError("GT exclusions require the explicit full-data protocol")
    config = Stage1ScaredCConfig(
        name=str(root.get("name", "stage1-scared-c-mean-v1")),
        dataset=dataset,
        mode=mode,
        data_root=data_root,
        split_path=split_path,
        split_active=split_active,
        cache_root=cache_root,
        train_sequences=train_sequences,
        validation_sequences=validation_sequences,
        initialization=init_mode,
        from_scratch=from_scratch,
        restore_ckpt=restore_ckpt,
        batch_size=int(root.get("batch_size", 1)),
        num_steps=int(root.get("num_steps", 60000)),
        learning_rate=float(root.get("learning_rate", 2e-4)),
        weight_decay=float(root.get("weight_decay", 1e-5)),
        train_iters=int(raft.get("train_iters", 3)),
        val_iters=int(raft.get("val_iters", 3)),
        mixed_precision=bool(raft.get("mixed_precision", False)),
        correlation_implementation=str(raft.get("correlation_implementation", "reg")),
        correlation_levels=int(raft.get("correlation_levels", 4)),
        correlation_radius=int(raft.get("correlation_radius", 4)),
        n_downsample=int(raft.get("n_downsample", 3)),
        n_gru_layers=int(raft.get("n_gru_layers", 1)),
        encoder_dims=tuple(encoder_dims_raw),
        hidden_dims=tuple(hidden_dims_raw),
        gradient_clip=float(root.get("gradient_clip", 1.0)),
        scheduler_pct_start=float(root.get("scheduler_pct_start", 0.01)),
        scheduler_anneal_strategy=str(root.get("scheduler_anneal_strategy", "linear")),
        seed=int(root.get("seed", 1314)),
        output_dir=output_dir,
        training_mode=training_mode,
        train_dataset_ids=train_dataset_ids,
        train_keyframe_entries=train_keyframe_entries,
        train_sample_ids=train_sample_ids,
        frame_split_path=frame_split_path,
        validation_fraction=validation_fraction,
        validation_block_policy=validation_block_policy,
        validation_rounding=validation_rounding,
        min_validation_frames_per_keyframe=min_validation_frames_per_keyframe,
        transfer_keyframe_entries=transfer_keyframe_entries,
        excluded_dataset_ids=excluded_dataset_ids,
        expected_keyframe_count=expected_keyframe_count,
        expected_total_samples=expected_total_samples,
        expected_train_samples=expected_train_samples,
        expected_validation_samples=expected_validation_samples,
        validation_steps=validation_steps,
        gt_quality_policy=gt_quality_policy,
    )
    if config.is_full_data:
        if config.batch_size != 1 or config.num_steps != 60000:
            raise Stage1ScaredCConfigError(
                "full-data Stage1 must retain batch_size=1 and num_steps=60000"
            )
        if config.train_iters != 3 or config.val_iters != 3:
            raise Stage1ScaredCConfigError(
                "full-data Stage1 must retain the existing 3-iteration stereo recipe"
            )
    return config


def _upstream_config(config: Stage1ScaredCConfig) -> SimpleNamespace:
    """Build only the pinned model's existing config fields; no YACS required."""

    raft = SimpleNamespace(
        mixed_precision=config.mixed_precision,
        train_iters=config.train_iters,
        val_iters=config.val_iters,
        corr_implementation=config.correlation_implementation,
        corr_levels=config.correlation_levels,
        corr_radius=config.correlation_radius,
        n_downsample=config.n_downsample,
        n_gru_layers=config.n_gru_layers,
        slow_fast_gru=False,
        encoder_dims=list(config.encoder_dims),
        hidden_dims=list(config.hidden_dims),
    )
    return SimpleNamespace(
        name=config.name,
        restore_ckpt=None,
        stage1_ckpt=None,
        dataset=SimpleNamespace(src_res=1024),
        raft=raft,
        gsnet=SimpleNamespace(encoder_dims=None, decoder_dims=None, parm_head_dim=None),
        record=SimpleNamespace(),
    )


def _load_upstream_network_module() -> ModuleType:
    """Import only ``lib.network``; this avoids train_stage1's TensorFlow path."""

    return import_upstream_module(
        "lib.network",
        UPSTREAM_ROOT,
        require_corr_sampler=False,
        check_python_dependencies=False,
        enforce_clean=False,
    )


def build_stage1_scratch_model(config: Stage1ScaredCConfig) -> torch.nn.Module:
    """Construct the pinned StereoEndoModel with canonical random initialization."""

    config.require_safe_mode(steps=3)
    if config.restore_ckpt is not None:
        raise Stage1ScaredCConfigError("scratch model construction received a checkpoint")
    network = _load_upstream_network_module()
    model = network.StereoEndoModel(_upstream_config(config), with_gs_render=False)
    if not isinstance(model, torch.nn.Module):
        raise Stage1ScaredCConfigError("upstream StereoEndoModel is not a torch module")
    return model


class Stage1ScaredCCachedDataset(Dataset[dict[str, object]]):
    """Real corrected-video RGB plus cached canonical disparity/mask."""

    def __init__(
        self,
        config: Stage1ScaredCConfig,
        *,
        sequences: Sequence[str] | None = None,
        sample_ids: Sequence[str] | None = None,
        cache_expected_sample_ids: Sequence[str] | None = None,
        cache_expected_split_manifest_sha256: str | None = None,
    ) -> None:
        self.config = config
        if (sequences is None) == (sample_ids is None):
            raise Stage1ScaredCConfigError(
                "Stage1 cached dataset requires exactly one of sequences or sample_ids"
            )
        selected_sequences = tuple(sequences or ())
        selected_sample_ids = tuple(sample_ids or ())
        if not selected_sequences and not selected_sample_ids:
            raise Stage1ScaredCConfigError("Stage1 cached dataset selection cannot be empty")
        keyframe_entries = None
        dataset_ids = None
        if sequences is not None:
            for sequence in selected_sequences:
                match = re.fullmatch(r"([1-9][0-9]*)_([1-9][0-9]*)", sequence)
                if match is None:
                    raise Stage1ScaredCConfigError(f"invalid Stage1 sequence selector: {sequence}")
                keyframe_entries = keyframe_entries or []
                keyframe_entries.append(f"dataset_{match.group(1)}/keyframe_{match.group(2)}")
            try:
                keyframe_entries = normalize_keyframe_entries(tuple(keyframe_entries))
            except Stage1FrameSplitError as error:
                raise Stage1ScaredCConfigError(str(error)) from error
            if any(
                entry.split("/", maxsplit=1)[0] not in STAGE1_TRAIN_DATASET_IDS
                for entry in keyframe_entries
            ):
                raise Stage1ScaredCConfigError(
                    "Stage1 cached dataset sequences must stay within dataset_1..3"
                )
        else:
            try:
                selected_sample_ids = normalize_sample_ids(selected_sample_ids)
            except Stage1FrameSplitError as error:
                raise Stage1ScaredCConfigError(str(error)) from error
            sample_dataset_ids = {
                sample_id.split("/", maxsplit=3)[1] for sample_id in selected_sample_ids
            }
            if not sample_dataset_ids <= set(STAGE1_TRAIN_DATASET_IDS):
                raise Stage1ScaredCConfigError(
                    "Stage1 cached dataset sample IDs must stay within dataset_1..3"
                )
            dataset_ids = config.train_dataset_ids if config.is_full_data else None
            if dataset_ids is None:
                dataset_ids = tuple(sorted(sample_dataset_ids))
        index = build_scared_c_index(
            config.dataset_root,
            mode=CORRECTED_VIDEO_MODE,
            manifest_path=config.split_path,
            strict=True,
            archive_validation="lazy",
            dataset_ids=dataset_ids,
            keyframe_entries=keyframe_entries,
        )
        if sequences is not None:
            self.records = select_stage1_records(
                index.records,
                keyframe_entries=keyframe_entries,
            )
        else:
            self.records = select_stage1_records(
                index.records,
                sample_ids=selected_sample_ids,
            )
        self.sequences = tuple(dict.fromkeys(record.sequence_key for record in self.records))
        if not self.records:
            raise Stage1ScaredCConfigError("Stage1 cached dataset has no selected records")
        if any(
            record.final_role
            or record.sequence_id in STAGE1_FORBIDDEN_DATASET_IDS
            or record.dataset_id != "scared_c"
            for record in self.records
        ):
            raise Stage1ScaredCConfigError(
                "Stage1 cached dataset selected forbidden/final-role content"
            )
        self.cache = Stage1GTCache(
            config.cache_root,
            expected_sample_ids=cache_expected_sample_ids,
            expected_split_manifest_sha256=cache_expected_split_manifest_sha256,
            expected_split_manifest_path=config.frame_split_path,
        )
        missing = [
            record.sample_id for record in self.records if not self.cache.contains(record.sample_id)
        ]
        if missing:
            raise Stage1ScaredCConfigError(
                f"Stage1 GT cache is incomplete; first missing sample is {missing[0]}"
            )
        self._rectification_cache = RectificationCache()
        self._video_readers: dict[Path, StackedStereoVideo] = {}

    def __len__(self) -> int:
        return len(self.records)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(record.sample_id for record in self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        rgb = materialize_scared_c_rectified_rgb(
            record,
            rectification_cache=self._rectification_cache,
            video_readers=self._video_readers,
        )
        disparity, valid_mask, disp_const, _entry = self.cache.load(record.sample_id)
        left = np.asarray(rgb["rgb_left_rect"], dtype=np.float32) / 127.5 - 1.0
        right = np.asarray(rgb["rgb_right_rect"], dtype=np.float32) / 127.5 - 1.0
        left = np.ascontiguousarray(left.transpose(2, 0, 1))
        right = np.ascontiguousarray(right.transpose(2, 0, 1))
        clean_disparity = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)
        clean_disparity[~valid_mask] = 0.0
        return {
            "left": torch.from_numpy(left),
            "right": torch.from_numpy(right),
            "disparity": torch.from_numpy(np.ascontiguousarray(clean_disparity[None])),
            "mask": torch.from_numpy(np.ascontiguousarray(valid_mask[None])),
            "disp_const": torch.tensor(disp_const, dtype=torch.float32),
            "sample_id": record.sample_id,
        }

    def close(self) -> None:
        for reader in self._video_readers.values():
            reader.close()
        self._video_readers.clear()
        self._rectification_cache.clear()


def stage1_collate(batch: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Collate the cached dataset into the pinned upstream ``lmain/rmain`` contract."""

    if not batch:
        raise ValueError("cannot collate an empty Stage1 batch")
    return {
        "lmain": {
            "img": torch.stack([item["left"] for item in batch]),
            "disp": torch.stack([item["disparity"] for item in batch]),
            "mask": torch.stack([item["mask"] for item in batch]),
            "disp_const": torch.stack([item["disp_const"] for item in batch]),
        },
        "rmain": {"img": torch.stack([item["right"] for item in batch])},
        "sample_id": [str(item["sample_id"]) for item in batch],
    }


def _move_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {"sample_id": batch["sample_id"]}
    for view in ("lmain", "rmain"):
        source = batch[view]
        assert isinstance(source, Mapping)
        moved[view] = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in source.items()
        }
    return moved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise Stage1ScaredCConfigError(f"unable to hash provenance file {path}: {error}") from error
    return digest.hexdigest()


def _git_head_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Stage1ScaredCConfigError(f"unable to resolve source revision: {error}") from error
    return result.stdout.strip()


def _validate_full_split_contract(
    config: Stage1ScaredCConfig,
    split: Stage1FrameSplit,
) -> Stage1FrameSplit:
    """Validate a persisted full-data split against the active config contract."""

    if tuple(split.dataset_ids) != tuple(config.train_dataset_ids):
        raise Stage1ScaredCConfigError(
            "persisted Stage1 frame split does not match the configured dataset allowlist"
        )
    if tuple(split.transfer_keyframe_entries) != tuple(config.transfer_keyframe_entries):
        raise Stage1ScaredCConfigError(
            "persisted Stage1 frame split does not match transfer keyframe policy"
        )
    if tuple(split.excluded_dataset_ids) != tuple(config.excluded_dataset_ids):
        raise Stage1ScaredCConfigError(
            "persisted Stage1 frame split does not match excluded dataset policy"
        )
    if (
        split.validation_fraction != config.validation_fraction
        or split.block_policy != config.validation_block_policy
        or split.rounding != config.validation_rounding
        or split.min_validation_frames_per_keyframe != config.min_validation_frames_per_keyframe
    ):
        raise Stage1ScaredCConfigError(
            "persisted Stage1 frame split does not match the configured validation policy"
        )
    if (
        config.expected_keyframe_count is not None
        and len(split.keyframe_entries) != config.expected_keyframe_count
    ):
        raise Stage1ScaredCConfigError(
            f"corrected keyframe count {len(split.keyframe_entries)} != "
            f"expected {config.expected_keyframe_count}"
        )
    if (
        config.expected_total_samples is not None
        and len(split.source_sample_ids) != config.expected_total_samples
    ):
        raise Stage1ScaredCConfigError(
            f"full-data sample count {len(split.source_sample_ids)} != "
            f"expected {config.expected_total_samples}"
        )
    if (
        config.expected_train_samples is not None
        and split.train_count != config.expected_train_samples
    ):
        raise Stage1ScaredCConfigError(
            f"train sample count {split.train_count} != expected {config.expected_train_samples}"
        )
    if (
        config.expected_validation_samples is not None
        and split.validation_count != config.expected_validation_samples
    ):
        raise Stage1ScaredCConfigError(
            f"validation sample count {split.validation_count} != "
            f"expected {config.expected_validation_samples}"
        )
    if set(split.train_sample_ids) & set(split.validation_sample_ids):
        raise Stage1ScaredCConfigError("persisted Stage1 train/validation IDs overlap")
    return split


def prepare_stage1_full_split(
    config: Stage1ScaredCConfig,
    *,
    source_code_sha: str | None = None,
) -> Stage1FrameSplit:
    """Resolve, validate, and persist the full-data temporal split.

    This preparation step indexes only the configured dataset allowlist.  It
    reads corrected-video metadata for datasets 1--3, never discovers a
    dataset directory from the root, and does not materialize RGB/GT payloads.
    """

    if not config.is_full_data:
        raise Stage1ScaredCConfigError(
            "full-data split preparation requires training_mode=full_data"
        )
    config.require_safe_mode(steps=3)
    if config.frame_split_path.exists():
        try:
            persisted = load_stage1_frame_split(config.frame_split_path)
        except (OSError, Stage1FrameSplitError) as error:
            raise Stage1ScaredCConfigError(
                f"unable to reuse persisted full Stage1 split: {error}"
            ) from error
        return _validate_full_split_contract(config, persisted)
    try:
        index = build_scared_c_index(
            config.dataset_root,
            mode=CORRECTED_VIDEO_MODE,
            manifest_path=config.split_path,
            strict=True,
            archive_validation="lazy",
            dataset_ids=config.train_dataset_ids,
        )
        records = select_stage1_records(
            index.records,
            dataset_ids=config.train_dataset_ids,
            allowed_dataset_ids=STAGE1_TRAIN_DATASET_IDS,
        )
    except (Stage1FrameSplitError, ValueError) as error:
        raise Stage1ScaredCConfigError(
            f"unable to prepare full Stage1 selection: {error}"
        ) from error
    if index.dataset_selection != tuple(config.train_dataset_ids):
        raise Stage1ScaredCConfigError("Stage1 index did not retain the explicit dataset allowlist")
    if any(
        record.final_role
        or record.sequence_id in STAGE1_FORBIDDEN_DATASET_IDS
        or record.dataset_id != "scared_c"
        for record in records
    ):
        raise Stage1ScaredCConfigError(
            "full-data Stage1 selection contains forbidden/final-role content"
        )
    split = make_stage1_frame_split(
        records,
        dataset_ids=config.train_dataset_ids,
        validation_fraction=config.validation_fraction,
        block_policy=config.validation_block_policy,
        rounding=config.validation_rounding,
        min_validation_frames_per_keyframe=config.min_validation_frames_per_keyframe,
        transfer_keyframe_entries=config.transfer_keyframe_entries,
        excluded_dataset_ids=config.excluded_dataset_ids,
        provenance={
            "role_manifest": str(config.split_path),
            "role_manifest_sha256": _sha256_file(config.split_path),
            "source_code_sha": source_code_sha or _git_head_sha(),
            "selection_contract": "explicit dataset_ids; corrected-video metadata only",
            "dataset6_accessed": False,
            "dataset7_accessed": False,
            "aliases": {
                "run_artifact": "stage1-scared-c-v3",
                "resolved_experiment": "stage1-scared-c-mean-v1",
                "checkpoint_family": "stage1_scared_c_v1",
            },
        },
    )
    _validate_full_split_contract(config, split)
    try:
        write_stage1_frame_split(config.frame_split_path, split)
        loaded = load_stage1_frame_split(config.frame_split_path)
    except (OSError, Stage1FrameSplitError) as error:
        raise Stage1ScaredCConfigError(f"unable to persist full Stage1 split: {error}") from error
    if loaded.manifest_sha256 != split.manifest_sha256:
        raise Stage1ScaredCConfigError(
            "persisted Stage1 frame split hash changed during validation"
        )
    return _validate_full_split_contract(config, loaded)


def prepare_stage1_full_data(
    config: Stage1ScaredCConfig,
    *,
    build_cache: bool = False,
    source_code_sha: str | None = None,
) -> Stage1FrameSplit:
    """Prepare the deterministic split and optionally materialize the GT cache."""

    split = prepare_stage1_full_split(config, source_code_sha=source_code_sha)
    if build_cache:
        build_stage1_gt_cache(
            config.dataset_root,
            config.cache_root,
            split_path=config.split_path,
            source_code_sha=source_code_sha or str(split.provenance.get("source_code_sha", "")),
            sample_ids=split.source_sample_ids,
            split_manifest_path=config.frame_split_path,
            expected_frame_counts=split.source_frame_counts,
        )
        if config.gt_quality_policy == "exclude_unusable":
            print("Auditing Stage1 GT cache before selecting usable frames...", flush=True)

            def report_progress(completed: int, total: int) -> None:
                if completed % 500 == 0 or completed == total:
                    print(f"GT audit: {completed}/{total}", flush=True)

            quality = audit_stage1_gt_cache(
                config.cache_root,
                split,
                split_manifest_path=config.frame_split_path,
                workers=4,
                progress=report_progress,
            )
            write_stage1_gt_quality_manifest(config.gt_quality_manifest_path, quality)
            print(
                f"GT quality: train={len(quality.train_sample_ids)}/{split.train_count} "
                f"validation={len(quality.validation_sample_ids)}/{split.validation_count} "
                f"excluded={len(quality.excluded)} manifest={config.gt_quality_manifest_path}",
                flush=True,
            )
    return split


def _load_gt_quality(
    config: Stage1ScaredCConfig, split: Stage1FrameSplit
) -> Stage1GTQualityManifest | None:
    """Apply only a persisted audit bound to this source split and cache."""

    if config.gt_quality_policy == "reject":
        return None
    return load_stage1_gt_quality_manifest(
        config.gt_quality_manifest_path,
        split=split,
        cache_root=config.cache_root,
        split_manifest_path=config.frame_split_path,
    )


@dataclass(frozen=True, slots=True)
class Stage1SmokeResult:
    """Evidence from exactly the bounded scratch smoke."""

    steps: int
    losses: tuple[float, ...]
    mean_data_seconds: float
    mean_compute_seconds: float
    mean_total_seconds: float
    allocations_gib: tuple[float, ...]
    peak_allocated_gib: float
    checkpoint_loaded: bool


def run_stage1_scratch_smoke(
    config: Stage1ScaredCConfig,
    *,
    device: torch.device,
    steps: int = 3,
) -> Stage1SmokeResult:
    """Run a bounded scratch smoke (full data: 1..100 steps); save no checkpoint."""

    if not config.is_full_data and steps != 3:
        raise Stage1ScaredCConfigError("the readiness smoke must run exactly three optimizer steps")
    if config.is_full_data and not 1 <= steps <= 100:
        raise Stage1ScaredCConfigError("the full-data smoke must run 1..100 optimizer steps")
    config.require_safe_mode(steps=steps)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    if config.is_full_data:
        split = load_stage1_frame_split(config.frame_split_path)
        quality = _load_gt_quality(config, split)
        dataset = Stage1ScaredCCachedDataset(
            config,
            sample_ids=quality.train_sample_ids if quality else split.train_sample_ids,
            cache_expected_sample_ids=split.source_sample_ids,
            cache_expected_split_manifest_sha256=split.manifest_sha256,
        )
    else:
        dataset = Stage1ScaredCCachedDataset(config, sequences=config.train_sequences)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.is_full_data,
        generator=torch.Generator().manual_seed(config.seed) if config.is_full_data else None,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=stage1_collate,
    )
    iterator = iter(loader)
    model = build_stage1_scratch_model(config).to(device)
    model.train()
    model.raft_stereo.freeze_bn()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        config.learning_rate,
        max(config.num_steps, steps),
        pct_start=config.scheduler_pct_start,
        cycle_momentum=False,
        anneal_strategy=config.scheduler_anneal_strategy,
    )
    losses: list[float] = []
    data_times: list[float] = []
    compute_times: list[float] = []
    total_times: list[float] = []
    allocations: list[float] = []
    peak_allocated = 0.0
    for _step in range(steps):
        total_start = time.perf_counter()
        data_start = total_start
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        data_seconds = time.perf_counter() - data_start
        batch_device = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_start = time.perf_counter()
        _output, loss, _metrics = model(batch_device, is_train=True)
        if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss).item():
            raise RuntimeError(f"Stage1 smoke loss is not finite: {loss!r}")
        loss.backward()
        finite_gradients = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        if not finite_gradients:
            raise RuntimeError("Stage1 smoke produced a non-finite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            allocated = float(torch.cuda.memory_allocated(device)) / (1024**3)
            peak_allocated = max(
                peak_allocated, float(torch.cuda.max_memory_allocated(device)) / (1024**3)
            )
        else:
            allocated = 0.0
        compute_seconds = time.perf_counter() - compute_start
        total_seconds = time.perf_counter() - total_start
        losses.append(float(loss.detach().cpu().item()))
        data_times.append(data_seconds)
        compute_times.append(compute_seconds)
        total_times.append(total_seconds)
        allocations.append(allocated)
    dataset.close()
    return Stage1SmokeResult(
        steps=steps,
        losses=tuple(losses),
        mean_data_seconds=float(np.mean(data_times)),
        mean_compute_seconds=float(np.mean(compute_times)),
        mean_total_seconds=float(np.mean(total_times)),
        allocations_gib=tuple(allocations),
        peak_allocated_gib=peak_allocated,
        checkpoint_loaded=False,
    )


def benchmark_stage1_cache_data(
    config: Stage1ScaredCConfig,
    *,
    samples: int = 8,
) -> float:
    """Measure cached RGB/GT preparation for a short sequence-local prefix."""

    if samples != 8:
        raise Stage1ScaredCConfigError("readiness benchmark must use eight consecutive samples")
    if config.is_full_data:
        split = load_stage1_frame_split(config.frame_split_path)
        quality = _load_gt_quality(config, split)
        dataset = Stage1ScaredCCachedDataset(
            config,
            sample_ids=(quality.train_sample_ids if quality else split.train_sample_ids)[:samples],
            cache_expected_sample_ids=split.source_sample_ids,
            cache_expected_split_manifest_sha256=split.manifest_sha256,
        )
    else:
        dataset = Stage1ScaredCCachedDataset(config, sequences=(config.train_sequences[0],))
    count = min(samples, len(dataset))
    durations: list[float] = []
    try:
        for index in range(count):
            start = time.perf_counter()
            dataset[index]
            durations.append(time.perf_counter() - start)
    finally:
        dataset.close()
    if count != samples:
        raise Stage1ScaredCConfigError(f"benchmark sequence has only {count} samples")
    return float(np.mean(durations))


@dataclass(frozen=True, slots=True)
class Stage1TrainingResult:
    """Immutable completion record for an official full-data Stage1 run."""

    steps_completed: int
    best_step: int
    best_validation_epe: float
    best_validation_bias: float
    selected_checkpoint: Path
    selected_checkpoint_sha256: str
    mean_data_seconds: float
    mean_compute_seconds: float
    mean_total_seconds: float


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _upstream_head_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Stage1ScaredCConfigError(
            f"unable to resolve upstream source revision: {error}"
        ) from error
    return result.stdout.strip()


def _resolved_config_payload(config: Stage1ScaredCConfig) -> dict[str, object]:
    """Serialize the resolved recipe and all data-selection contracts."""

    return {
        "name": config.name,
        "dataset": config.dataset,
        "mode": config.mode,
        "training_mode": config.training_mode,
        "data_root": str(config.data_root),
        "role_manifest": str(config.split_path),
        "role_manifest_active": config.split_active,
        "frame_split_manifest": str(config.frame_split_path),
        "cache_root": str(config.cache_root),
        "gt_quality_policy": config.gt_quality_policy,
        "train_sequences": list(config.train_sequences),
        "validation_sequences": list(config.validation_sequences),
        "train_dataset_ids": list(config.train_dataset_ids),
        "train_keyframe_entries": list(config.train_keyframe_entries),
        "train_sample_ids": list(config.train_sample_ids),
        "validation_fraction": config.validation_fraction,
        "validation_block_policy": config.validation_block_policy,
        "validation_rounding": config.validation_rounding,
        "min_validation_frames_per_keyframe": config.min_validation_frames_per_keyframe,
        "transfer_keyframe_entries": list(config.transfer_keyframe_entries),
        "excluded_dataset_ids": list(config.excluded_dataset_ids),
        "expected_counts": {
            "corrected_keyframes": config.expected_keyframe_count,
            "total_samples": config.expected_total_samples,
            "train_samples": config.expected_train_samples,
            "validation_samples": config.expected_validation_samples,
        },
        "initialization": config.initialization,
        "from_scratch": config.from_scratch,
        "restore_ckpt": None if config.restore_ckpt is None else str(config.restore_ckpt),
        "batch_size": config.batch_size,
        "num_steps": config.num_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "train_iters": config.train_iters,
        "val_iters": config.val_iters,
        "mixed_precision": config.mixed_precision,
        "correlation_implementation": config.correlation_implementation,
        "correlation_runtime": "PYTORCH_EQUIVALENT",
        "correlation_levels": config.correlation_levels,
        "correlation_radius": config.correlation_radius,
        "n_downsample": config.n_downsample,
        "n_gru_layers": config.n_gru_layers,
        "encoder_dims": list(config.encoder_dims),
        "hidden_dims": list(config.hidden_dims),
        "gradient_clip": config.gradient_clip,
        "scheduler_pct_start": config.scheduler_pct_start,
        "scheduler_anneal_strategy": config.scheduler_anneal_strategy,
        "seed": config.seed,
        "validation_steps": list(config.validation_steps),
        "output_dir": str(config.output_dir),
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR",
        "loss": "pinned upstream lib.network StereoEndoModel loss",
        "disparity_convention": "x_left - x_right",
        "disparity_sign": "positive",
        "disparity_unit": "pixels",
    }


def _write_effective_config(
    path: Path,
    config: Stage1ScaredCConfig,
    *,
    source_sha: str,
    upstream_sha: str,
) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "source_sha": source_sha,
            "upstream_sha": upstream_sha,
            "resolved": _resolved_config_payload(config),
        },
    )


def _metric_row(
    *,
    step: int,
    sequence: str,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    return {
        "step": step,
        "sequence": sequence,
        "epe": float(metrics["epe"]),
        "median_ae": float(metrics["median_ae"]),
        "bias": float(metrics["bias"]),
        "bad3": float(metrics["bad3"]),
        "bad5": float(metrics["bad5"]),
        "frames": int(metrics["frames"]),
        "valid_pixels": int(metrics["valid_pixels"]),
    }


def _evaluate_sequence(
    model: torch.nn.Module,
    dataset: Stage1ScaredCCachedDataset,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate one sample-ID partition with the canonical Stage1 output."""

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=stage1_collate,
    )
    frame_epe: list[float] = []
    frame_median_ae: list[float] = []
    frame_bias: list[float] = []
    frame_bad3: list[float] = []
    frame_bad5: list[float] = []
    valid_pixels = 0
    for batch in loader:
        batch_device = _move_batch(batch, device)
        with torch.inference_mode():
            output, _loss, _metrics = model(batch_device, is_train=False)
        if not isinstance(output, Mapping):
            raise RuntimeError("Stage1 validation output is not a mapping")
        left_output = output.get("lmain")
        if not isinstance(left_output, Mapping):
            raise RuntimeError("Stage1 validation output is missing lmain")
        prediction = left_output.get("flow_pred")
        target = batch_device["lmain"]["disp"]
        mask = batch_device["lmain"]["mask"].bool()
        if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise RuntimeError("Stage1 validation did not return tensor disparity values")
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise RuntimeError(
                f"Stage1 validation shape mismatch: prediction={tuple(prediction.shape)} "
                f"target={tuple(target.shape)} mask={tuple(mask.shape)}"
            )
        valid = mask & torch.isfinite(prediction) & torch.isfinite(target)
        count = int(valid.sum().item())
        if count == 0:
            raise RuntimeError("Stage1 validation encountered a zero-valid frame")
        signed_error = prediction - target
        absolute_error = signed_error.abs()
        valid_signed = signed_error[valid]
        valid_absolute = absolute_error[valid]
        frame_epe.append(float(valid_absolute.mean().item()))
        frame_median_ae.append(float(valid_absolute.median().item()))
        frame_bias.append(float(valid_signed.mean().item()))
        frame_bad3.append(float((valid_absolute > 3.0).float().mean().item()))
        frame_bad5.append(float((valid_absolute > 5.0).float().mean().item()))
        valid_pixels += count
    if not frame_epe:
        raise RuntimeError("Stage1 validation partition has no frames")
    return {
        "epe": float(np.mean(frame_epe)),
        "median_ae": float(np.mean(frame_median_ae)),
        "bias": float(np.mean(frame_bias)),
        "bad3": float(np.mean(frame_bad3)),
        "bad5": float(np.mean(frame_bad5)),
        "frames": len(frame_epe),
        "valid_pixels": valid_pixels,
    }


def _evaluate_validation(
    model: torch.nn.Module,
    validation_datasets: Mapping[str, Stage1ScaredCCachedDataset],
    *,
    step: int,
    device: torch.device,
) -> list[dict[str, object]]:
    partition_metrics = {
        name: _evaluate_sequence(model, dataset, device=device)
        for name, dataset in validation_datasets.items()
    }
    rows = [
        _metric_row(step=step, sequence=name, metrics=metrics)
        for name, metrics in partition_metrics.items()
    ]
    macro = {
        metric: float(np.mean([float(metrics[metric]) for metrics in partition_metrics.values()]))
        for metric in ("epe", "median_ae", "bias", "bad3", "bad5")
    }
    macro["frames"] = sum(int(metrics["frames"]) for metrics in partition_metrics.values())
    macro["valid_pixels"] = sum(
        int(metrics["valid_pixels"]) for metrics in partition_metrics.values()
    )
    rows.append(_metric_row(step=step, sequence="MACRO", metrics=macro))
    return rows


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    config: Stage1ScaredCConfig,
    source_sha: str,
    split: Stage1FrameSplit,
    validation_row: Mapping[str, object],
    quality: Stage1GTQualityManifest | None = None,
) -> dict[str, object]:
    return {
        "total_steps": step,
        "network": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "stage1_scared_c": {
            "source_sha": source_sha,
            "config": _resolved_config_payload(config),
            "frame_split_manifest": str(config.frame_split_path),
            "frame_split_manifest_sha256": split.manifest_sha256,
            "train_sample_count": len(quality.train_sample_ids) if quality else split.train_count,
            "validation_sample_count": (
                len(quality.validation_sample_ids) if quality else split.validation_count
            ),
            "gt_quality_manifest_sha256": quality.manifest_sha256 if quality else None,
            "excluded_sample_count": len(quality.excluded) if quality else 0,
            "selection_row": dict(validation_row),
            "initialization": "scratch",
            "restore_ckpt": None,
        },
    }


@dataclass(frozen=True, slots=True)
class _Stage1ResumeState:
    """Validated state needed to continue one interrupted full-data run."""

    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_source_sha: str | None
    start_step: int
    best_step: int
    best_validation_epe: float
    best_abs_bias: float
    selection_row: dict[str, object]


def _tensor_is_finite(value: torch.Tensor) -> bool:
    """Return whether a tensor contains only finite values when applicable."""

    if not (value.is_floating_point() or value.is_complex()):
        return True
    return bool(torch.isfinite(value).all().item())


def _load_stage1_resume_checkpoint(
    path: str | Path,
    *,
    config: Stage1ScaredCConfig,
    split: Stage1FrameSplit,
    quality: Stage1GTQualityManifest | None,
) -> tuple[_Stage1ResumeState, Mapping[str, object]]:
    """Load and validate an explicit full-data checkpoint before training."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise Stage1ScaredCConfigError(f"resume checkpoint is missing: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise Stage1ScaredCConfigError(
            f"unable to load resume checkpoint {checkpoint_path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise Stage1ScaredCConfigError("resume checkpoint must contain a mapping")

    raw_step = payload.get("total_steps")
    if isinstance(raw_step, bool) or not isinstance(raw_step, int):
        raise Stage1ScaredCConfigError("resume checkpoint total_steps must be an integer")
    if raw_step < 1 or raw_step >= config.num_steps:
        raise Stage1ScaredCConfigError(
            "resume checkpoint must be an interior step below the configured 60000 steps"
        )
    if raw_step not in config.validation_steps:
        raise Stage1ScaredCConfigError(
            "resume checkpoint must be selected at a configured validation step"
        )

    metadata = payload.get("stage1_scared_c")
    if not isinstance(metadata, Mapping):
        raise Stage1ScaredCConfigError("resume checkpoint metadata is missing")
    checkpoint_config = metadata.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise Stage1ScaredCConfigError("resume checkpoint config metadata is missing")
    for field in (
        "name",
        "dataset",
        "mode",
        "training_mode",
        "num_steps",
        "seed",
        "batch_size",
        "train_iters",
        "val_iters",
    ):
        expected = getattr(config, field)
        actual = checkpoint_config.get(field)
        if actual is not None and actual != expected:
            raise Stage1ScaredCConfigError(
                f"resume checkpoint config mismatch for {field}: {actual!r} != {expected!r}"
            )
    if metadata.get("frame_split_manifest_sha256") != split.manifest_sha256:
        raise Stage1ScaredCConfigError(
            "resume checkpoint frame split manifest does not match the current split"
        )
    expected_quality_sha = quality.manifest_sha256 if quality is not None else None
    if metadata.get("gt_quality_manifest_sha256") != expected_quality_sha:
        raise Stage1ScaredCConfigError(
            "resume checkpoint GT-quality manifest does not match the current audit"
        )
    for field, expected in (
        ("train_sample_count", len(quality.train_sample_ids) if quality else split.train_count),
        (
            "validation_sample_count",
            len(quality.validation_sample_ids) if quality else split.validation_count,
        ),
    ):
        if metadata.get(field) != expected:
            raise Stage1ScaredCConfigError(
                f"resume checkpoint {field} does not match the current split"
            )

    raw_selection = metadata.get("selection_row")
    if not isinstance(raw_selection, Mapping):
        raise Stage1ScaredCConfigError("resume checkpoint selection row is missing")
    if raw_selection.get("step") != raw_step or raw_selection.get("sequence") != "MACRO":
        raise Stage1ScaredCConfigError(
            "resume checkpoint selection row does not identify its saved validation step"
        )
    try:
        selection_row = {
            "step": int(raw_selection["step"]),
            "sequence": "MACRO",
            "epe": float(raw_selection["epe"]),
            "median_ae": float(raw_selection["median_ae"]),
            "bias": float(raw_selection["bias"]),
            "bad3": float(raw_selection["bad3"]),
            "bad5": float(raw_selection["bad5"]),
            "frames": int(raw_selection["frames"]),
            "valid_pixels": int(raw_selection["valid_pixels"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise Stage1ScaredCConfigError(
            "resume checkpoint selection row has invalid metrics"
        ) from error
    if not all(
        math.isfinite(float(selection_row[field]))
        for field in ("epe", "median_ae", "bias", "bad3", "bad5")
    ):
        raise Stage1ScaredCConfigError("resume checkpoint selection metrics must be finite")

    network = payload.get("network")
    optimizer_state = payload.get("optimizer")
    scheduler_state = payload.get("scheduler")
    if not isinstance(network, Mapping):
        raise Stage1ScaredCConfigError("resume checkpoint network state is missing")
    if not isinstance(optimizer_state, Mapping):
        raise Stage1ScaredCConfigError("resume checkpoint optimizer state is missing")
    if not isinstance(scheduler_state, Mapping):
        raise Stage1ScaredCConfigError("resume checkpoint scheduler state is missing")
    for name, state in (("network", network), ("optimizer", optimizer_state)):
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and not _tensor_is_finite(value):
                raise Stage1ScaredCConfigError(
                    f"resume checkpoint {name} contains non-finite tensor {key!r}"
                )
            if name == "optimizer" and isinstance(value, Mapping):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, torch.Tensor) and not _tensor_is_finite(
                        nested_value
                    ):
                        raise Stage1ScaredCConfigError(
                            "resume checkpoint optimizer contains non-finite state "
                            f"{key!r}.{nested_key!r}"
                        )
    if scheduler_state.get("total_steps") != config.num_steps:
        raise Stage1ScaredCConfigError(
            "resume checkpoint scheduler total_steps does not match the 60000-step recipe"
        )
    if scheduler_state.get("last_epoch") != raw_step:
        raise Stage1ScaredCConfigError(
            "resume checkpoint scheduler state does not match total_steps"
        )

    checkpoint_source_sha = metadata.get("source_sha")
    if checkpoint_source_sha is not None and not isinstance(checkpoint_source_sha, str):
        raise Stage1ScaredCConfigError("resume checkpoint source_sha must be a string")
    return (
        _Stage1ResumeState(
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=_sha256_file(checkpoint_path),
            checkpoint_source_sha=checkpoint_source_sha,
            start_step=raw_step,
            best_step=raw_step,
            best_validation_epe=float(selection_row["epe"]),
            best_abs_bias=abs(float(selection_row["bias"])),
            selection_row=selection_row,
        ),
        payload,
    )


def _restore_stage1_resume_state(
    payload: Mapping[str, object],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_step: int,
    device: torch.device,
) -> None:
    """Restore model, optimizer, and scheduler state for an explicit resume."""

    try:
        model.load_state_dict(payload["network"], strict=True)  # type: ignore[arg-type]
        optimizer.load_state_dict(payload["optimizer"])  # type: ignore[arg-type]
        scheduler.load_state_dict(payload["scheduler"])  # type: ignore[arg-type]
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise Stage1ScaredCConfigError(
            f"resume checkpoint state is incompatible with the current model: {error}"
        ) from error
    if scheduler.last_epoch != expected_step:
        raise Stage1ScaredCConfigError("restored scheduler step does not match the checkpoint step")
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _read_csv_prefix(
    path: Path,
    *,
    max_step: int,
    required_fields: Sequence[str],
) -> list[dict[str, str]]:
    """Read a prior run's trace up to the checkpoint step, if available."""

    if not path.is_file():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fieldnames = tuple(reader.fieldnames or ())
            missing = set(required_fields) - set(fieldnames)
            if missing:
                raise Stage1ScaredCConfigError(
                    f"resume trace {path} is missing fields: {sorted(missing)}"
                )
            rows: list[dict[str, str]] = []
            for raw in reader:
                if not raw:
                    continue
                try:
                    step = int(raw.get("step", ""))
                except (TypeError, ValueError) as error:
                    raise Stage1ScaredCConfigError(
                        f"resume trace {path} contains an invalid step"
                    ) from error
                if step <= max_step:
                    rows.append({field: raw.get(field, "") for field in required_fields})
            return rows
    except OSError as error:
        raise Stage1ScaredCConfigError(f"unable to read resume trace {path}: {error}") from error


def run_stage1_scratch_training(
    config: Stage1ScaredCConfig,
    *,
    device: torch.device,
    artifact_root: str | Path,
    resume_ckpt: str | Path | None = None,
) -> Stage1TrainingResult:
    """Run the fixed full-data 60k-step Stage1 protocol from scratch or a checkpoint."""

    if not config.is_full_data:
        raise Stage1ScaredCConfigError(
            "the official full-data entrypoint requires training_mode=full_data"
        )
    steps = config.num_steps
    if steps != 60000:
        raise Stage1ScaredCConfigError(
            "the official Stage1-SCARED-C run must be exactly 60000 steps"
        )
    config.require_safe_mode(steps=steps)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Stage1ScaredCConfigError("the official Stage1 run requires a visible CUDA device")
    if resume_ckpt is None:
        resume_path: Path | None = None
    else:
        resume_path = Path(resume_ckpt).expanduser()
        if not resume_path.is_absolute():
            resume_path = (REPOSITORY_ROOT / resume_path).resolve()

    source_sha = _git_head_sha()
    upstream_sha = _upstream_head_sha()
    split = prepare_stage1_full_data(
        config,
        build_cache=True,
        source_code_sha=source_sha,
    )
    quality = _load_gt_quality(config, split)
    train_sample_ids = quality.train_sample_ids if quality else split.train_sample_ids
    validation_sample_ids = (
        quality.validation_sample_ids if quality else split.validation_sample_ids
    )
    root = Path(artifact_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise Stage1ScaredCConfigError(
            f"refusing to overwrite non-empty Stage1 artifact directory {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    checkpoints_root = root / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    cache_manifest_path = config.cache_root / "manifest.json"
    try:
        cache_payload = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage1ScaredCConfigError(
            f"unable to read prepared Stage1 cache manifest: {error}"
        ) from error
    if not isinstance(cache_payload, Mapping):
        raise Stage1ScaredCConfigError("prepared Stage1 cache manifest must be an object")
    validated_cache = Stage1GTCache(
        config.cache_root,
        expected_sample_ids=split.source_sample_ids,
        expected_split_manifest_sha256=split.manifest_sha256,
        expected_split_manifest_path=config.frame_split_path,
    )
    if cache_payload.get("entry_count") != len(split.source_sample_ids):
        raise Stage1ScaredCConfigError("prepared cache entry count does not match source samples")
    if len(validated_cache) != len(split.source_sample_ids):
        raise Stage1ScaredCConfigError("prepared cache entries do not match source samples")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    train_dataset = Stage1ScaredCCachedDataset(
        config,
        sample_ids=train_sample_ids,
        cache_expected_sample_ids=split.source_sample_ids,
        cache_expected_split_manifest_sha256=split.manifest_sha256,
    )
    validation_dataset = Stage1ScaredCCachedDataset(
        config,
        sample_ids=validation_sample_ids,
        cache_expected_sample_ids=split.source_sample_ids,
        cache_expected_split_manifest_sha256=split.manifest_sha256,
    )
    if set(train_dataset.sample_ids) & set(validation_dataset.sample_ids):
        train_dataset.close()
        validation_dataset.close()
        raise Stage1ScaredCConfigError("full Stage1 train/validation sample IDs overlap")
    if len(train_dataset) != len(train_sample_ids) or len(validation_dataset) != len(
        validation_sample_ids
    ):
        train_dataset.close()
        validation_dataset.close()
        raise Stage1ScaredCConfigError("Stage1 dataset lengths do not match the persisted split")

    run_manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha,
        "upstream_sha": upstream_sha,
        "dataset": "scared_c",
        "dataset_revision": "44baac1187c8729c96db0d1def569bd94c9d9417",
        "dataset_root": str(config.dataset_root),
        "role_manifest": str(config.split_path),
        "role_manifest_sha256": _sha256_file(config.split_path),
        "frame_split_manifest": str(config.frame_split_path),
        "frame_split_manifest_sha256": split.manifest_sha256,
        "cache_root": str(config.cache_root),
        "cache_manifest_sha256": _sha256_file(cache_manifest_path),
        "cache_source_code_sha": cache_payload.get("source_code_sha"),
        "cache_entry_count": cache_payload.get("entry_count"),
        "train_dataset_ids": list(config.train_dataset_ids),
        "train_keyframe_entries": list(split.keyframe_entries),
        "source_train_frames": split.train_count,
        "source_validation_frames": split.validation_count,
        "train_frames": len(train_sample_ids),
        "validation_frames": len(validation_sample_ids),
        "gt_quality_policy": config.gt_quality_policy,
        "gt_quality_manifest": "gt_quality_manifest.json" if quality else None,
        "gt_quality_manifest_sha256": quality.manifest_sha256 if quality else None,
        "excluded_frames": len(quality.excluded) if quality else 0,
        "validation_policy": {
            "fraction": split.validation_fraction,
            "block_policy": split.block_policy,
            "rounding": split.rounding,
            "min_frames_per_keyframe": split.min_validation_frames_per_keyframe,
        },
        "initialization": "resume" if resume_path is not None else "scratch",
        "from_scratch": resume_path is None,
        "restore_ckpt": None if resume_path is None else str(resume_path),
        "resume_checkpoint_sha256": None,
        "resume_checkpoint_source_sha": None,
        "resume_from_step": 0,
        "correlation_runtime": "PYTORCH_EQUIVALENT",
        "validation_steps": list(config.validation_steps),
        "validation_metric_aggregation": "mean frame metrics over the internal validation partition",
        "gpu": {
            "logical_device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "device_count_visible": torch.cuda.device_count(),
        },
        "dataset_allowlist": list(STAGE1_TRAIN_DATASET_IDS),
        "excluded_dataset_ids": list(config.excluded_dataset_ids),
        "transfer_keyframe_entries": list(config.transfer_keyframe_entries),
        "dataset6_content_accessed": False,
        "dataset7_content_accessed": False,
        "aliases": {
            "run_artifact": "stage1-scared-c-v3",
            "resolved_experiment": "stage1-scared-c-mean-v1",
            "checkpoint_family": "stage1_scared_c_v1",
        },
        "effective_config": "effective_config.json",
        "training_log": "training_log.csv",
        "validation_metrics": "validation_metrics.csv",
        "checkpoints": "checkpoints",
    }
    _write_effective_config(
        root / "effective_config.json",
        config,
        source_sha=source_sha,
        upstream_sha=upstream_sha,
    )
    _write_json(root / "run_manifest.json", run_manifest)
    if quality is not None:
        write_stage1_gt_quality_manifest(root / "gt_quality_manifest.json", quality)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=0,
        pin_memory=True,
        collate_fn=stage1_collate,
    )
    train_iterator = iter(train_loader)
    model = build_stage1_scratch_model(config).to(device)
    model.train()
    model.raft_stereo.freeze_bn()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        config.learning_rate,
        steps,
        pct_start=config.scheduler_pct_start,
        cycle_momentum=False,
        anneal_strategy=config.scheduler_anneal_strategy,
    )
    torch.cuda.reset_peak_memory_stats(device)
    resume_state: _Stage1ResumeState | None = None
    resume_payload: Mapping[str, object] | None = None
    start_step = 0
    best_epe: float | None = None
    best_abs_bias: float | None = None
    best_step: int | None = None
    best_checkpoint_tmp = checkpoints_root / "best_current.pth"
    data_times: list[float] = []
    compute_times: list[float] = []
    total_times: list[float] = []
    validation_rows: list[dict[str, object]] = []
    last_step = 0
    training_header = [
        "step",
        "train_loss",
        "learning_rate",
        "data_seconds",
        "compute_seconds",
        "total_seconds",
        "gpu_memory_allocated_gib",
        "gpu_memory_peak_gib",
    ]
    validation_header = [
        "step",
        "sequence",
        "epe",
        "median_ae",
        "bias",
        "bad3",
        "bad5",
        "frames",
        "valid_pixels",
    ]

    try:
        resume_training_rows: list[dict[str, str]] = []
        resume_validation_rows: list[dict[str, str]] = []
        if resume_path is not None:
            resume_state, resume_payload = _load_stage1_resume_checkpoint(
                resume_path,
                config=config,
                split=split,
                quality=quality,
            )
            assert resume_payload is not None
            _restore_stage1_resume_state(
                resume_payload,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_step=resume_state.start_step,
                device=device,
            )
            torch.cuda.reset_peak_memory_stats(device)
            start_step = resume_state.start_step
            best_epe = resume_state.best_validation_epe
            best_abs_bias = resume_state.best_abs_bias
            best_step = resume_state.best_step
            last_step = start_step
            validation_rows.append(dict(resume_state.selection_row))
            source_run_root = resume_state.checkpoint_path.parent.parent
            resume_training_rows = _read_csv_prefix(
                source_run_root / "training_log.csv",
                max_step=start_step,
                required_fields=training_header,
            )
            resume_validation_rows = _read_csv_prefix(
                source_run_root / "validation_metrics.csv",
                max_step=start_step,
                required_fields=validation_header,
            )
            if not any(
                int(row["step"]) == start_step and row["sequence"] == "MACRO"
                for row in resume_validation_rows
            ):
                resume_validation_rows.append(
                    {field: str(resume_state.selection_row[field]) for field in validation_header}
                )
            data_times.extend(
                float(row["data_seconds"]) for row in resume_training_rows if row["data_seconds"]
            )
            compute_times.extend(
                float(row["compute_seconds"])
                for row in resume_training_rows
                if row["compute_seconds"]
            )
            total_times.extend(
                float(row["total_seconds"]) for row in resume_training_rows if row["total_seconds"]
            )
            shutil.copy2(resume_state.checkpoint_path, best_checkpoint_tmp)
            run_manifest.update(
                {
                    "resume_checkpoint_sha256": resume_state.checkpoint_sha256,
                    "resume_checkpoint_source_sha": resume_state.checkpoint_source_sha,
                    "resume_from_step": start_step,
                    "resume_trace_training_rows": len(resume_training_rows),
                    "resume_trace_validation_rows": len(resume_validation_rows),
                    "resume_data_order": "seeded_loader_restart",
                    "resume_bitwise_reproducible": False,
                }
            )
            _write_json(root / "run_manifest.json", run_manifest)
        with (
            (root / "training_log.csv").open("w", newline="", encoding="utf-8") as train_file,
            (root / "validation_metrics.csv").open(
                "w", newline="", encoding="utf-8"
            ) as validation_file,
        ):
            train_writer = csv.DictWriter(train_file, fieldnames=training_header)
            validation_writer = csv.DictWriter(validation_file, fieldnames=validation_header)
            train_writer.writeheader()
            validation_writer.writeheader()
            for row in resume_training_rows:
                train_writer.writerow(row)
            for row in resume_validation_rows:
                validation_writer.writerow(row)
            train_file.flush()
            validation_file.flush()
            for step in range(start_step + 1, steps + 1):
                total_start = time.perf_counter()
                data_start = total_start
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)
                data_seconds = time.perf_counter() - data_start
                batch_device = _move_batch(batch, device)
                valid_count = int(batch_device["lmain"]["mask"].sum().item())
                if valid_count == 0:
                    raise RuntimeError(
                        f"Stage1 training encountered a zero-valid batch at step {step}"
                    )
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                compute_start = time.perf_counter()
                _output, loss, _metrics = model(batch_device, is_train=True)
                if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss).item():
                    raise RuntimeError(
                        f"Stage1 training loss is not finite at step {step}: {loss!r}"
                    )
                loss.backward()
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()
                ):
                    raise RuntimeError(
                        f"Stage1 training produced non-finite gradients at step {step}"
                    )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
                if not torch.isfinite(torch.as_tensor(gradient_norm)).item():
                    raise RuntimeError(f"Stage1 gradient norm is not finite at step {step}")
                optimizer.step()
                scheduler.step()
                torch.cuda.synchronize(device)
                compute_seconds = time.perf_counter() - compute_start
                total_seconds = time.perf_counter() - total_start
                data_times.append(data_seconds)
                compute_times.append(compute_seconds)
                total_times.append(total_seconds)
                last_step = step
                train_writer.writerow(
                    {
                        "step": step,
                        "train_loss": float(loss.detach().cpu().item()),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "data_seconds": data_seconds,
                        "compute_seconds": compute_seconds,
                        "total_seconds": total_seconds,
                        "gpu_memory_allocated_gib": float(torch.cuda.memory_allocated(device))
                        / (1024**3),
                        "gpu_memory_peak_gib": float(torch.cuda.max_memory_allocated(device))
                        / (1024**3),
                    }
                )
                train_file.flush()

                if step in config.validation_steps:
                    model.eval()
                    rows = _evaluate_validation(
                        model,
                        {"INTERNAL_VALIDATION": validation_dataset},
                        step=step,
                        device=device,
                    )
                    model.train()
                    model.raft_stereo.freeze_bn()
                    for row in rows:
                        validation_rows.append(row)
                        validation_writer.writerow(row)
                    validation_file.flush()
                    macro = next(row for row in rows if row["sequence"] == "MACRO")
                    macro_epe = float(macro["epe"])
                    macro_abs_bias = abs(float(macro["bias"]))
                    is_better = (
                        best_epe is None
                        or macro_epe < best_epe
                        or (
                            math.isclose(macro_epe, best_epe, rel_tol=0.0, abs_tol=1e-12)
                            and (best_abs_bias is None or macro_abs_bias < best_abs_bias)
                        )
                    )
                    if is_better:
                        best_epe = macro_epe
                        best_abs_bias = macro_abs_bias
                        best_step = step
                        torch.save(
                            _checkpoint_payload(
                                model,
                                optimizer,
                                scheduler,
                                step=step,
                                config=config,
                                source_sha=source_sha,
                                split=split,
                                validation_row=macro,
                                quality=quality,
                            ),
                            best_checkpoint_tmp,
                        )
                    print(
                        f"VALIDATION step={step} epe={macro_epe:.6f} "
                        f"median_ae={float(macro['median_ae']):.6f} "
                        f"bias={float(macro['bias']):.6f} best_step={best_step}",
                        flush=True,
                    )
                if step % 100 == 0 or step == steps:
                    print(
                        f"PROGRESS step={step}/{steps} loss={float(loss.detach().cpu().item()):.6f} "
                        f"data={data_seconds:.4f}s compute={compute_seconds:.4f}s "
                        f"total={total_seconds:.4f}s",
                        flush=True,
                    )
    except Exception as error:
        run_manifest["status"] = "failed"
        run_manifest["failed_at_step"] = last_step
        run_manifest["error"] = repr(error)
        run_manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(root / "run_manifest.json", run_manifest)
        raise
    finally:
        train_dataset.close()
        validation_dataset.close()

    if (
        best_step is None
        or best_epe is None
        or best_abs_bias is None
        or not best_checkpoint_tmp.is_file()
    ):
        raise RuntimeError("Stage1 completed without a selected validation checkpoint")
    selected_checkpoint = checkpoints_root / (
        f"ReliableEndoGS_stage1_scared_c_v1_best_step{best_step:06d}.pth"
    )
    if selected_checkpoint.exists():
        raise Stage1ScaredCConfigError(
            f"refusing to overwrite selected checkpoint {selected_checkpoint}"
        )
    best_checkpoint_tmp.replace(selected_checkpoint)
    selected_sha256 = _sha256_file(selected_checkpoint)
    best_bias = float(
        next(
            row["bias"]
            for row in validation_rows
            if row["step"] == best_step and row["sequence"] == "MACRO"
        )
    )
    run_manifest.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "steps_completed": last_step,
            "best_step": best_step,
            "best_validation_epe": best_epe,
            "best_validation_bias": best_bias,
            "selected_checkpoint": str(selected_checkpoint),
            "selected_checkpoint_sha256": selected_sha256,
            "mean_data_seconds": float(np.mean(data_times)),
            "mean_compute_seconds": float(np.mean(compute_times)),
            "mean_total_seconds": float(np.mean(total_times)),
        }
    )
    _write_json(root / "run_manifest.json", run_manifest)
    return Stage1TrainingResult(
        steps_completed=last_step,
        best_step=best_step,
        best_validation_epe=best_epe,
        best_validation_bias=best_bias,
        selected_checkpoint=selected_checkpoint,
        selected_checkpoint_sha256=selected_sha256,
        mean_data_seconds=float(np.mean(data_times)),
        mean_compute_seconds=float(np.mean(compute_times)),
        mean_total_seconds=float(np.mean(total_times)),
    )


__all__ = [
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_CONFIG_PATH",
    "Stage1ScaredCConfig",
    "Stage1ScaredCConfigError",
    "Stage1ScaredCCachedDataset",
    "Stage1SmokeResult",
    "Stage1TrainingResult",
    "benchmark_stage1_cache_data",
    "build_stage1_scratch_model",
    "load_stage1_scared_c_config",
    "prepare_stage1_full_data",
    "prepare_stage1_full_split",
    "run_stage1_scratch_smoke",
    "run_stage1_scratch_training",
    "stage1_collate",
]
