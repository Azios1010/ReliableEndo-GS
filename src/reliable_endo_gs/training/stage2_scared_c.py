"""Stage-2 SCARED-C camera-local same-view control runner.

This runner initializes the pinned upstream Gaussian head from the provisional
Stage-1 60k artifact. SCARED-C poses are unresolved, so every frame uses a
camera-local identity extrinsic. The resulting artifact is not valid for
novel-view or cross-view claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

from reliable_endo_gs.data.scared_c_stage1_quality import load_stage1_gt_quality_manifest
from reliable_endo_gs.data.scared_c_stage1_split import load_stage1_frame_split
from reliable_endo_gs.data.scared_c_stage2 import cached_stage1_to_scared_c_stage2_sample
from reliable_endo_gs.data.stage2 import ScaredStage2Sample, stage2_collate_fn
from reliable_endo_gs.training.stage1_scared_c import (
    Stage1ScaredCCachedDataset,
    Stage1ScaredCConfig,
    load_stage1_scared_c_config,
)
from reliable_endo_gs.training.stage2_control import (
    add_upstream_path,
    cpu_state,
    forward_loss,
    health,
    move_batch,
    tensor_finite,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {name}")
    return Path(value).expanduser().resolve()


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


@dataclass(frozen=True, slots=True)
class Stage2ScaredCProtocol:
    repo: Path
    config_path: Path
    artifact_root: Path
    checkpoint_path: Path
    freeze_path: Path
    quality_path: Path
    stage1_config: Stage1ScaredCConfig
    steps: int
    batch_size: int
    seed: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    validation_interval: int
    checkpoint_interval: int


class _ScaredCStage2Dataset(Dataset[ScaredStage2Sample]):
    """Expose quality-audited cached SCARED-C samples through the Stage-2 contract."""

    def __init__(
        self,
        config: Stage1ScaredCConfig,
        *,
        sample_ids: Sequence[str],
        expected_sample_ids: Sequence[str],
        split_sha256: str,
    ) -> None:
        self._cached = Stage1ScaredCCachedDataset(
            config,
            sample_ids=sample_ids,
            cache_expected_sample_ids=expected_sample_ids,
            cache_expected_split_manifest_sha256=split_sha256,
        )
        self.sample_ids = self._cached.sample_ids

    def __len__(self) -> int:
        return len(self._cached)

    def __getitem__(self, index: int) -> ScaredStage2Sample:
        return cached_stage1_to_scared_c_stage2_sample(self._cached[index])

    def close(self) -> None:
        self._cached.close()


def _load_yaml(path: Path) -> Mapping[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Stage-2 config must be a mapping: {path}")
    return value


def resolve_protocol(repo: Path, config_path: Path) -> Stage2ScaredCProtocol:
    raw = _load_yaml(config_path)
    if raw.get("schema_version") != "stage2_scared_c_camera_local.v1":
        raise ValueError("unsupported Stage-2 SCARED-C config schema")
    if raw.get("protocol_status") != "camera_local_same_view_control":
        raise ValueError("Stage-2 config must explicitly declare camera-local same-view scope")
    stage1 = raw.get("stage1")
    data = raw.get("data")
    checkpointing = raw.get("checkpointing")
    if not isinstance(stage1, Mapping) or not isinstance(data, Mapping) or not isinstance(checkpointing, Mapping):
        raise ValueError("Stage-2 config requires stage1, data, and checkpointing mappings")
    artifact_root = _require_env(str(stage1["artifact_env"]))
    freeze_path = artifact_root / str(stage1["freeze_file"])
    checkpoint_path = artifact_root / str(stage1["checkpoint"])
    quality_path = artifact_root / str(stage1["quality_file"])
    stage1_raw_config = load_stage1_scared_c_config(repo / str(stage1["config"]))
    stage1_config = replace(
        stage1_raw_config,
        data_root=_require_env(str(data["root_env"])),
        frame_split_path=_require_env(str(data["frame_split_root_env"])) / "split_manifest.json",
    )
    steps = int(raw["steps"])
    batch_size = int(raw["batch_size"])
    if steps <= 0 or batch_size != 1:
        raise ValueError("Stage-2 SCARED-C requires positive steps and batch_size=1")
    return Stage2ScaredCProtocol(
        repo=repo,
        config_path=config_path,
        artifact_root=artifact_root,
        checkpoint_path=checkpoint_path,
        freeze_path=freeze_path,
        quality_path=quality_path,
        stage1_config=stage1_config,
        steps=steps,
        batch_size=batch_size,
        seed=int(raw["seed"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        gradient_clip=float(raw["gradient_clip"]),
        validation_interval=int(raw["validation_interval"]),
        checkpoint_interval=int(checkpointing["interval"]),
    )


def _frozen_inputs(protocol: Stage2ScaredCProtocol) -> tuple[object, object, dict[str, object]]:
    for path in (protocol.freeze_path, protocol.checkpoint_path, protocol.quality_path, protocol.stage1_config.frame_split_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    freeze = json.loads(protocol.freeze_path.read_text(encoding="utf-8"))
    if not isinstance(freeze, dict) or freeze.get("status") != "provisionally_frozen":
        raise RuntimeError("Stage-1 artifact is not provisionally frozen")
    if freeze.get("scope", {}).get("stage2_input_authorized") is not True:
        raise RuntimeError("Stage-1 freeze record does not authorize Stage-2 input")
    actual_checkpoint_sha = _sha256(protocol.checkpoint_path)
    if freeze.get("selected_checkpoint_sha256") != actual_checkpoint_sha:
        raise RuntimeError("Stage-1 checkpoint SHA-256 differs from freeze record")
    checkpoint = torch.load(protocol.checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("network"), Mapping):
        raise RuntimeError("Stage-1 checkpoint has no network state")
    if int(checkpoint.get("total_steps", -1)) != 60000:
        raise RuntimeError("Stage-1 checkpoint is not the frozen 60000-step artifact")
    if not tensor_finite(checkpoint["network"]):
        raise RuntimeError("Stage-1 checkpoint has non-finite network tensors")
    split = load_stage1_frame_split(protocol.stage1_config.frame_split_path)
    quality = load_stage1_gt_quality_manifest(
        protocol.quality_path,
        split=split,
        cache_root=protocol.stage1_config.cache_root,
        split_manifest_path=protocol.stage1_config.frame_split_path,
    )
    if len(quality.train_sample_ids) != 7749 or len(quality.validation_sample_ids) != 422:
        raise RuntimeError("Stage-2 SCARED-C quality partition differs from the frozen Stage-1 artifact")
    return split, quality, {"checkpoint_sha256": actual_checkpoint_sha, "freeze_sha256": _sha256(protocol.freeze_path)}


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _loader(dataset: Dataset[ScaredStage2Sample], *, shuffle: bool, seed: int) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=True,
        collate_fn=stage2_collate_fn,
    )


def _load_model(protocol: Stage2ScaredCProtocol) -> torch.nn.Module:
    upstream = protocol.repo / "third_party" / "endo_e2e_gs"
    add_upstream_path(upstream)
    from config.stereo_config import ConfigStereo
    from lib.network import StereoEndoModel

    loader = ConfigStereo()
    loader.load(str(upstream / "config" / "stage2.yaml"))
    cfg = loader.get_cfg()
    cfg.defrost()
    cfg.batch_size = protocol.batch_size
    cfg.num_steps = protocol.steps
    cfg.raft.mixed_precision = False
    cfg.freeze()
    model = StereoEndoModel(cfg, with_gs_render=True)
    payload = torch.load(protocol.checkpoint_path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(payload["network"], strict=False)
    missing = list(result.missing_keys)
    if not missing or not all(key.startswith("gs_parm_regresser.") for key in missing) or result.unexpected_keys:
        raise RuntimeError(f"Stage-1 to Stage-2 contract mismatch: missing={missing}, unexpected={result.unexpected_keys}")
    return model


def _validate(model: torch.nn.Module, dataset: Dataset[ScaredStage2Sample], device: torch.device, bg_color: list[float]) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in _loader(dataset, shuffle=False, seed=1314):
            _data, _loss, metrics = forward_loss(model, move_batch(batch, device), bg_color)
            rows.append(metrics)
    model.train()
    model.raft_stereo.freeze_bn()
    return {key: float(np.mean([row[key] for row in rows])) for key in ("total_loss", "disp_loss", "render_l1", "ssim_term", "epe", "one_px", "three_px")}


def smoke(protocol: Stage2ScaredCProtocol, *, steps: int, device: torch.device) -> dict[str, object]:
    if steps < 1 or steps > 10:
        raise ValueError("Stage-2 smoke must run 1..10 steps")
    split, quality, identities = _frozen_inputs(protocol)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Stage-2 SCARED-C requires exactly one visible CUDA GPU")
    _seed(protocol.seed)
    train = _ScaredCStage2Dataset(protocol.stage1_config, sample_ids=quality.train_sample_ids, expected_sample_ids=split.source_sample_ids, split_sha256=split.manifest_sha256)
    try:
        model = _load_model(protocol).to(device).train()
        model.raft_stereo.freeze_bn()
        optimizer = optim.AdamW(model.parameters(), lr=protocol.learning_rate, weight_decay=protocol.weight_decay, eps=1e-8)
        iterator = iter(_loader(train, shuffle=True, seed=protocol.seed))
        losses: list[float] = []
        for _ in range(steps):
            batch = move_batch(next(iterator), device)
            optimizer.zero_grad(set_to_none=True)
            data, loss, _metrics = forward_loss(model, batch, list(model.cfg.dataset.bg_color))
            if not torch.isfinite(loss):
                raise RuntimeError("Stage-2 smoke loss is non-finite")
            health(data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), protocol.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        return {"smoke_steps": steps, "losses": losses, "identities": identities}
    finally:
        train.close()


def run(protocol: Stage2ScaredCProtocol, *, output_dir: Path, device: torch.device) -> None:
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite Stage-2 output: {output_dir}")
    if _git_output(protocol.repo, "status", "--short", "--untracked-files=all"):
        raise RuntimeError("commit the Stage-2 runner before starting an official artifact")
    split, quality, identities = _frozen_inputs(protocol)
    _seed(protocol.seed)
    train = _ScaredCStage2Dataset(protocol.stage1_config, sample_ids=quality.train_sample_ids, expected_sample_ids=split.source_sample_ids, split_sha256=split.manifest_sha256)
    validation = _ScaredCStage2Dataset(protocol.stage1_config, sample_ids=quality.validation_sample_ids, expected_sample_ids=split.source_sample_ids, split_sha256=split.manifest_sha256)
    try:
        model = _load_model(protocol).to(device).train()
        model.raft_stereo.freeze_bn()
        optimizer = optim.AdamW(model.parameters(), lr=protocol.learning_rate, weight_decay=protocol.weight_decay, eps=1e-8)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, protocol.learning_rate, total_steps=protocol.steps, pct_start=0.01, cycle_momentum=False, anneal_strategy="linear")
        output_dir.mkdir(parents=True)
        provenance = {
            "protocol": "stage2_scared_c_camera_local_same_view_control",
            "stage1_checkpoint": str(protocol.checkpoint_path),
            "stage1_step": 60000,
            "camera_extrinsics": "identity_camera_local",
            "novel_view_claims": "prohibited",
            "train_samples": len(train),
            "validation_samples": len(validation),
            "split_sha256": split.manifest_sha256,
            "quality_sha256": quality.manifest_sha256,
            "source_sha": _git_output(protocol.repo, "rev-parse", "HEAD"),
            **identities,
        }
        (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        best = float("inf")
        iterator = iter(_loader(train, shuffle=True, seed=protocol.seed))
        with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as train_file, (output_dir / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as validation_file:
            train_writer = csv.DictWriter(train_file, fieldnames=("step", "total_loss", "epe", "render_l1", "seconds"))
            validation_writer = csv.DictWriter(validation_file, fieldnames=("step", "total_loss", "disp_loss", "render_l1", "ssim_term", "epe", "one_px", "three_px"))
            train_writer.writeheader()
            validation_writer.writeheader()
            for step in range(1, protocol.steps + 1):
                started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(_loader(train, shuffle=True, seed=protocol.seed))
                    batch = next(iterator)
                optimizer.zero_grad(set_to_none=True)
                data, loss, metrics = forward_loss(model, move_batch(batch, device), list(model.cfg.dataset.bg_color))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite Stage-2 loss at step {step}")
                health(data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), protocol.gradient_clip)
                optimizer.step()
                scheduler.step()
                train_writer.writerow({"step": step, "total_loss": float(loss.detach().cpu().item()), "epe": metrics["epe"], "render_l1": metrics["render_l1"], "seconds": time.perf_counter() - started})
                if step % protocol.validation_interval == 0:
                    metrics = _validate(model, validation, device, list(model.cfg.dataset.bg_color))
                    validation_writer.writerow({"step": step, **metrics})
                    if metrics["total_loss"] < best:
                        best = metrics["total_loss"]
                        torch.save({"network": cpu_state(model.state_dict()), "optimizer": cpu_state(optimizer.state_dict()), "scheduler": cpu_state(scheduler.state_dict()), "step": step, "provenance": provenance | {"best_validation_total_loss": best}}, output_dir / "best_validation_total_loss.pth")
                if step % protocol.checkpoint_interval == 0:
                    torch.save({"network": cpu_state(model.state_dict()), "optimizer": cpu_state(optimizer.state_dict()), "scheduler": cpu_state(scheduler.state_dict()), "step": step, "provenance": provenance}, output_dir / "latest.pth")
    finally:
        train.close()
        validation.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    protocol = resolve_protocol(args.repo.resolve(), args.config.resolve())
    device = torch.device(args.device)
    if args.smoke_steps is not None:
        print(json.dumps(smoke(protocol, steps=args.smoke_steps, device=device), indent=2, sort_keys=True))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required for the full Stage-2 run")
    run(protocol, output_dir=args.output_dir.resolve(), device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
