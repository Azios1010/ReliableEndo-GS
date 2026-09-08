"""Guarded Stage1 SCARED-C mean-disparity training entry point.

This adapter keeps the pinned upstream stereo architecture and loss while
owning the SCARED-C split, cache, and initialization contracts.  It permits a
three-step smoke while the split manifest is inactive and refuses a longer
run until a separate training task explicitly activates that manifest.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
)
from reliable_endo_gs.data.scared_c_stereo import RectificationCache, StackedStereoVideo

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = REPOSITORY_ROOT / "third_party" / "endo_e2e_gs"
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "training" / "stage1_scared_c_mean_v1.yaml"
DEFAULT_CACHE_ROOT = Path(r"E:\runtime-tmp\reliable-endo-gs\scared-c-stage1-cache")


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

    @property
    def dataset_root(self) -> Path:
        return self.data_root / self.dataset

    def require_safe_mode(self, *, steps: int) -> None:
        if steps < 1:
            raise Stage1ScaredCConfigError("steps must be positive")
        if not self.from_scratch or self.initialization != "scratch":
            raise Stage1ScaredCConfigError("Stage1 SCARED-C requires explicit scratch initialization")
        if self.restore_ckpt is not None:
            raise Stage1ScaredCConfigError(
                f"scratch mode cannot carry restore_ckpt={self.restore_ckpt}"
            )
        if steps != 3 and not _manifest_is_active(self.split_path):
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
    """Load and validate the explicit inactive Stage1 SCARED-C config."""

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (REPOSITORY_ROOT / config_path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise Stage1ScaredCConfigError(f"unable to read Stage1 config {config_path}: {error}") from error
    root = _require_mapping(raw, "Stage1 config")
    dataset = str(root.get("dataset", ""))
    mode = str(root.get("mode", ""))
    if dataset != "scared_c" or mode != CORRECTED_VIDEO_MODE:
        raise Stage1ScaredCConfigError("Stage1 config must select dataset=scared_c and corrected_video")

    initialization = _require_mapping(root.get("initialization"), "initialization")
    init_mode = str(initialization.get("mode", ""))
    from_scratch = initialization.get("from_scratch") is True
    restore_ckpt = _parse_optional_checkpoint(initialization.get("restore_ckpt"))
    if init_mode != "scratch" or not from_scratch or restore_ckpt is not None:
        raise Stage1ScaredCConfigError(
            "initialization must be mode=scratch, from_scratch=true, restore_ckpt=null"
        )

    split_path = _expand_runtime_path(root.get("split_manifest"), name="split_manifest")
    split_payload = _require_mapping(
        json.loads(split_path.read_text(encoding="utf-8")), "split manifest"
    )
    if split_payload.get("active") is not False:
        raise Stage1ScaredCConfigError("the remediation Stage1 split must remain inactive")
    roles = _require_mapping(split_payload.get("roles"), "split roles")
    train_sequences = _tuple_of_strings(root.get("train_sequences"), "train_sequences")
    validation_sequences = _tuple_of_strings(
        root.get("validation_sequences"), "validation_sequences"
    )
    if train_sequences != STAGE1_MEAN_TRAIN or validation_sequences != STAGE1_MEAN_VALIDATION:
        raise Stage1ScaredCConfigError("Stage1 train/validation roles do not match stage1_mean_v1")
    if tuple(roles.get("MEAN_TRAIN", ())) != STAGE1_MEAN_TRAIN:
        raise Stage1ScaredCConfigError("split MEAN_TRAIN does not match the approved sequence list")
    if tuple(roles.get("MEAN_VALIDATION", ())) != STAGE1_MEAN_VALIDATION:
        raise Stage1ScaredCConfigError("split MEAN_VALIDATION does not match the approved sequence list")
    if any(sequence.startswith(("6_", "7_")) for sequence in train_sequences + validation_sequences):
        raise Stage1ScaredCConfigError("dataset_6/dataset_7 cannot enter mean train or validation")

    raft = _require_mapping(root.get("raft"), "raft")
    encoder_dims_raw = raft.get("encoder_dims")
    hidden_dims_raw = raft.get("hidden_dims")
    if not isinstance(encoder_dims_raw, list) or not all(isinstance(v, int) for v in encoder_dims_raw):
        raise Stage1ScaredCConfigError("raft.encoder_dims must be a list of integers")
    if not isinstance(hidden_dims_raw, list) or not all(isinstance(v, int) for v in hidden_dims_raw):
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
        root.get("output_dir"), name="output_dir", default=REPOSITORY_ROOT / "outputs" / "stage1_scared_c"
    )
    return Stage1ScaredCConfig(
        name=str(root.get("name", "stage1-scared-c-mean-v1")),
        dataset=dataset,
        mode=mode,
        data_root=data_root,
        split_path=split_path,
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
    )


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
        sequences: Sequence[str],
    ) -> None:
        self.config = config
        self.sequences = tuple(sequences)
        index = build_scared_c_index(
            config.dataset_root,
            mode=CORRECTED_VIDEO_MODE,
            manifest_path=config.split_path,
            strict=True,
            archive_validation="lazy",
        )
        allowed = set(self.sequences)
        self.records = tuple(
            record
            for record in index.records
            if record.sequence_key in allowed
        )
        if not self.records:
            raise Stage1ScaredCConfigError("Stage1 cached dataset has no selected records")
        if any(record.final_role or record.sequence_key.startswith("6_") for record in self.records):
            raise Stage1ScaredCConfigError("Stage1 cached dataset selected final-role content")
        self.cache = Stage1GTCache(config.cache_root)
        missing = [record.sample_id for record in self.records if not self.cache.contains(record.sample_id)]
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
    """Run exactly three optimizer steps from scratch; never save a checkpoint."""

    if steps != 3:
        raise Stage1ScaredCConfigError("the readiness smoke must run exactly three optimizer steps")
    config.require_safe_mode(steps=steps)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    dataset = Stage1ScaredCCachedDataset(config, sequences=config.train_sequences)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
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
            peak_allocated = max(peak_allocated, float(torch.cuda.max_memory_allocated(device)) / (1024**3))
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


__all__ = [
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_CONFIG_PATH",
    "Stage1ScaredCConfig",
    "Stage1ScaredCConfigError",
    "Stage1ScaredCCachedDataset",
    "Stage1SmokeResult",
    "benchmark_stage1_cache_data",
    "build_stage1_scratch_model",
    "load_stage1_scared_c_config",
    "run_stage1_scratch_smoke",
    "stage1_collate",
]
