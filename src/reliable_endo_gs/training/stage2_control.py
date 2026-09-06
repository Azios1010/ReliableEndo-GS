"""Production vanilla Stage-2 multi-sequence v3 control runner.

This module owns only orchestration around the pinned Endo-E2E-GS model and the
validated v3 adapter.  It intentionally contains no uncertainty or repair logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
import yaml

EXPECTED_TRAIN = ("dataset_1/keyframe_1", "dataset_2/keyframe_1", "dataset_3/keyframe_1")
EXPECTED_VAL = ("dataset_7/keyframe_2", "dataset_4/keyframe_4")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def git_output(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


@dataclass(frozen=True)
class Protocol:
    repo: Path
    upstream: Path
    config_path: Path
    run_dir: Path
    data_root: Path
    manifest_path: Path
    split_path: Path
    stage1_path: Path
    source_sha: str
    upstream_sha: str
    manifest_sha: str
    split_sha: str
    data_config_sha: str
    stage1_sha: str
    steps: int = 10000
    scheduler_steps: int = 10100
    seed: int = 1314
    batch_size: int = 1
    val_interval: int = 1000


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def resolve_protocol(repo: Path, config_path: Path, run_dir: Path) -> Protocol:
    raw = load_yaml(config_path)
    data = raw["data"]
    identity = raw["identity"]
    stage1 = raw["stage1"]
    data_root = Path(os.environ[data["root_env"]])
    manifest_path = Path(os.environ[data["manifest_env"]])
    stage1_path = Path(os.environ[stage1["checkpoint_env"]])
    split_path = repo / data["split"]
    data_config_path = repo / data["config"]
    upstream = repo / "third_party" / "endo_e2e_gs"
    source_sha_env = str(identity.get("source_sha_env", ""))
    source_sha = os.environ.get(source_sha_env, "") if source_sha_env else str(identity.get("source_sha", ""))
    if not source_sha:
        raise RuntimeError("RELIABLE_ENDO_SOURCE_SHA must identify the exact pushed source commit")
    return Protocol(
        repo=repo,
        upstream=upstream,
        config_path=config_path,
        run_dir=run_dir,
        data_root=data_root,
        manifest_path=manifest_path,
        split_path=split_path,
        stage1_path=stage1_path,
        source_sha=source_sha,
        upstream_sha=str(identity["upstream_sha"]),
        manifest_sha=str(identity["manifest_sha256"]),
        split_sha=str(identity["split_sha256"]),
        data_config_sha=str(identity["config_sha256"]),
        stage1_sha=str(stage1["sha256"]),
        steps=int(raw["steps"]),
        scheduler_steps=int(raw["scheduler_total_steps"]),
        seed=int(raw["seed"]),
        batch_size=int(raw["batch_size"]),
        val_interval=int(raw["validation"]["interval"]),
    )


def verify_identities(protocol: Protocol, *, require_remote: bool = True) -> dict[str, Any]:
    if protocol.run_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run directory: {protocol.run_dir}")
    if not protocol.stage1_path.is_file():
        raise FileNotFoundError(protocol.stage1_path)
    for path in (protocol.manifest_path, protocol.split_path, protocol.config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_sha = git_output("rev-parse", "HEAD", cwd=protocol.repo)
    remote_sha = git_output("ls-remote", "origin", "refs/heads/main", cwd=protocol.repo).split()[0]
    source_status = git_output("status", "--short", "--untracked-files=all", cwd=protocol.repo)
    upstream_sha = git_output("rev-parse", "HEAD", cwd=protocol.upstream)
    upstream_status = git_output("status", "--short", "--untracked-files=all", cwd=protocol.upstream)
    hashes = {
        "stage1_sha256": sha256_file(protocol.stage1_path),
        "manifest_sha256": sha256_file(protocol.manifest_path),
        "split_sha256": sha256_file(protocol.split_path),
        "config_sha256": sha256_file(protocol.repo / "configs" / "data" / "scared_five_keyframe_v3.yaml"),
    }
    expected = {
        "stage1_sha256": protocol.stage1_sha,
        "manifest_sha256": protocol.manifest_sha,
        "split_sha256": protocol.split_sha,
        "config_sha256": protocol.data_config_sha,
    }
    if hashes != expected:
        raise RuntimeError(f"frozen hash mismatch: actual={hashes}, expected={expected}")
    if source_sha != protocol.source_sha or (require_remote and remote_sha != protocol.source_sha) or source_status:
        raise RuntimeError(f"source identity/status failure: {source_sha}, {remote_sha}, {source_status!r}")
    if upstream_sha != protocol.upstream_sha or upstream_status:
        raise RuntimeError(f"third_party identity/status failure: {upstream_sha}, {upstream_status!r}")
    return {
        "source_sha": source_sha,
        "remote_sha": remote_sha,
        "upstream_sha": upstream_sha,
        "source_status": source_status,
        "third_party_status": upstream_status,
        **hashes,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def add_upstream_path(upstream: Path) -> None:
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))


def load_upstream_config(protocol: Protocol) -> Any:
    add_upstream_path(protocol.upstream)
    from config.stereo_config import ConfigStereo

    loader = ConfigStereo()
    loader.load(str(protocol.upstream / "config" / "stage2.yaml"))
    cfg = loader.get_cfg()
    cfg.defrost()
    cfg.batch_size = protocol.batch_size
    cfg.num_steps = protocol.steps
    cfg.raft.mixed_precision = False
    cfg.freeze()
    if int(cfg.raft.train_iters) != 3 or int(cfg.raft.val_iters) != 3:
        raise RuntimeError("pinned RAFT train/validation iterations are not 3/3")
    return cfg


def scheduler_probe(protocol: Protocol) -> dict[str, Any]:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = optim.AdamW([parameter], lr=2e-4, weight_decay=1e-5, eps=1e-8)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-4,
        total_steps=protocol.scheduler_steps,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )
    samples = [float(optimizer.param_groups[0]["lr"])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(protocol.steps):
            scheduler.step()
            samples.append(float(optimizer.param_groups[0]["lr"]))
    if not all(math.isfinite(value) for value in samples) or min(samples) < 0:
        raise RuntimeError("final OneCycleLR probe produced invalid learning rates")
    return {
        "class": "OneCycleLR",
        "total_steps": protocol.scheduler_steps,
        "pct_start": 0.01,
        "anneal_strategy": "linear",
        "initial_lr": samples[0],
        "peak_lr": max(samples),
        "final_sampled_lr": samples[-1],
        "negative_lr_samples": sum(value < 0 for value in samples),
        "scheduler_only_steps": protocol.steps,
    }


def load_stage1_contract(protocol: Protocol, cfg: Any) -> dict[str, Any]:
    from lib.network import StereoEndoModel

    model = StereoEndoModel(cfg, with_gs_render=True)
    checkpoint = torch.load(protocol.stage1_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "network" not in checkpoint:
        raise RuntimeError("Stage-1 checkpoint has no network state")
    if int(checkpoint.get("step", checkpoint.get("global_step", 7000))) != 7000:
        raise RuntimeError("Stage-1 checkpoint step is not 7000")
    if not tensor_finite(checkpoint["network"]):
        raise RuntimeError("Stage-1 checkpoint contains non-finite network tensors")
    result = model.load_state_dict(checkpoint["network"], strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if len(missing) != 144 or not all(key.startswith("gs_parm_regresser.") for key in missing) or unexpected:
        raise RuntimeError(f"Stage-1 load contract mismatch: missing={len(missing)}, unexpected={unexpected}")
    return {"missing_keys": missing, "unexpected_keys": unexpected, "model": model}


def dry_run(protocol: Protocol) -> dict[str, Any]:
    identities = verify_identities(protocol)
    cfg = load_upstream_config(protocol)
    scheduler = scheduler_probe(protocol)
    model_contract = load_stage1_contract(protocol, cfg)
    del model_contract["model"]
    cuda = {
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if not cuda["available"] or cuda["device_count"] != 1:
        raise RuntimeError(f"GPU1 visibility gate failed: {cuda}")
    import wandb

    if not getattr(wandb.api, "api_key", None):
        raise RuntimeError("W&B API credentials are not available")
    return {
        "dry_run": True,
        "identities": identities,
        "stage1_contract": model_contract,
        "scheduler": scheduler,
        "gpu": cuda,
        "wandb": {"entity": "azios1010-hanoi-university-of-science-and-technology", "authenticated": True},
    }


def tensor_finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(tensor_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensor_finite(item) for item in value)
    return True


def scalar(value: torch.Tensor | float) -> float:
    return float(value.detach().item() if isinstance(value, torch.Tensor) else value)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for view in ("lmain", "rmain"):
        for key, value in batch[view].items():
            if isinstance(value, torch.Tensor):
                batch[view][key] = value.to(device, non_blocking=True)
    return batch


def forward_loss(model: torch.nn.Module, batch: dict[str, Any], bg_color: list[float], *, render: bool = True) -> tuple[dict[str, Any], torch.Tensor, dict[str, float]]:
    from lib.GaussianRender import pts2render
    from lib.loss import l1_loss, ssim

    data, disp_loss, _ = model(batch, is_train=True)
    if disp_loss is None:
        raise RuntimeError("Stage-2 forward did not return disparity loss")
    if render:
        data, _ = pts2render(data, bg_color=bg_color)
        rendered = data["lmain"]["img_pred"]
        target = (data["lmain"]["img"] + 1.0) / 2.0
        render_l1 = l1_loss(rendered, target)
        ssim_term = 1.0 - ssim(rendered, target)
    else:
        render_l1 = torch.zeros_like(disp_loss)
        ssim_term = torch.zeros_like(disp_loss)
    total = disp_loss + 0.8 * render_l1 + 0.2 * ssim_term
    disparity = data["lmain"]["disp"]
    flow = data["lmain"]["flow_pred"]
    valid = data["lmain"]["mask"] >= 0.5
    epe_map = torch.sum((disparity - flow) ** 2, dim=1).sqrt()
    epe = epe_map.view(-1)[valid.view(-1)]
    if epe.numel() == 0:
        raise RuntimeError("validation/training sample has no valid disparity")
    metrics = {
        "total_loss": scalar(total),
        "disp_loss": scalar(disp_loss),
        "render_l1": scalar(render_l1),
        "ssim_term": scalar(ssim_term),
        "epe": scalar(epe.mean()),
        "one_px": scalar((epe < 1).float().mean()),
        "three_px": scalar((epe < 3).float().mean()),
        "valid_gaussian_count": int(data["lmain"]["pts_valid"].sum().item()),
    }
    return data, total, metrics


def health(data: dict[str, Any]) -> dict[str, Any]:
    lmain = data["lmain"]
    valid = lmain["pts_valid"].bool()
    if not valid.any():
        raise RuntimeError("all Gaussian centers are invalid")
    opacity_map = lmain["opacity_maps"].permute(0, 2, 3, 1)
    scale_map = lmain["scale_maps"].permute(0, 2, 3, 1)
    opacity = opacity_map.reshape(-1, opacity_map.shape[-1])[valid.reshape(-1)]
    scale = scale_map.reshape(-1, scale_map.shape[-1])[valid.reshape(-1)]
    xyz = lmain["xyz"].reshape(-1, lmain["xyz"].shape[-1])[valid.reshape(-1)]
    w2v = lmain["world_view_transform"]
    xyz_h = torch.cat((xyz, torch.ones_like(xyz[..., :1])), dim=-1).unsqueeze(0)
    camera_xyz = torch.bmm(xyz_h, w2v).squeeze(0)[..., :3]
    values = {
        "valid_gaussian_count": int(valid.sum().item()),
        "opacity_mean": scalar(opacity.mean()),
        "opacity_min": scalar(opacity.min()),
        "opacity_max": scalar(opacity.max()),
        "scale_mean": scalar(scale.mean()),
        "scale_min": scalar(scale.min()),
        "scale_max": scalar(scale.max()),
        "camera_z_min": scalar(camera_xyz[..., 2].min()),
        "camera_z_max": scalar(camera_xyz[..., 2].max()),
        "render_min": scalar(lmain["img_pred"].min()),
        "render_max": scalar(lmain["img_pred"].max()),
        "negative_valid_camera_z": int((camera_xyz[..., 2] < 0).sum().item()),
    }
    if not tensor_finite(lmain) or values["negative_valid_camera_z"]:
        raise RuntimeError(f"invalid Gaussian health: {values}")
    return values


def cpu_state(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def make_loader(dataset: Any, shuffle: bool, seed: int) -> Any:
    from torch.utils.data import DataLoader
    from reliable_endo_gs.data.stage2 import stage2_collate_fn

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, generator=generator, num_workers=0, pin_memory=True, collate_fn=stage2_collate_fn)


def sequence_from_name(name: str) -> str:
    return "/".join(str(name).split("/")[:2])


def evaluate(model: torch.nn.Module, dataset: Any, device: torch.device, bg_color: list[float]) -> dict[str, Any]:
    from statistics import fmean

    loader = make_loader(dataset, False, 1314)
    was_training = model.training
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            sequence = sequence_from_name(batch["name"][0])
            batch = move_batch(batch, device)
            _, _, metrics = forward_loss(model, batch, bg_color)
            records.append({"sequence": sequence, **metrics})
    if was_training:
        model.train()
        model.raft_stereo.freeze_bn()
    keys = ("total_loss", "disp_loss", "render_l1", "ssim_term", "epe", "one_px", "three_px")
    per_sequence = {}
    for sequence in EXPECTED_VAL:
        rows = [row for row in records if row["sequence"] == sequence]
        if not rows:
            raise RuntimeError(f"validation sequence missing: {sequence}")
        per_sequence[sequence] = {"samples": len(rows), **{key: fmean(row[key] for row in rows) for key in keys}}
    aggregate = {key: fmean(row[key] for row in records) for key in keys}
    macro = {key: fmean(per_sequence[sequence][key] for sequence in EXPECTED_VAL) for key in keys}
    return {"samples": len(records), "aggregate": aggregate, "per_sequence": per_sequence, "macro": macro}


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: Any, scheduler: Any, step: int, protocol: Protocol, identities: dict[str, Any], metadata: dict[str, Any]) -> str:
    payload = {
        "network": cpu_state(model.state_dict()),
        "optimizer": cpu_state(optimizer.state_dict()),
        "scheduler": cpu_state(scheduler.state_dict()),
        "step": step,
        "protocol": metadata,
        "provenance": {**identities, "stage1_checkpoint": protocol.stage1_path.name, "stage1_step": 7000, "stage1_sha256": protocol.stage1_sha},
    }
    torch.save(payload, path)
    return sha256_file(path)


def run(protocol: Protocol) -> None:
    identities = verify_identities(protocol)
    seed_everything(protocol.seed)
    add_upstream_path(protocol.upstream)
    from lib.network import StereoEndoModel
    from reliable_endo_gs.data.stage2 import ScaredStage2Dataset
    import wandb

    cfg = load_upstream_config(protocol)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("physical GPU1 must be the only visible GPU")
    train = ScaredStage2Dataset(scared_root=protocol.data_root, split_manifest=protocol.split_path, split="train", phase="train", test_every=8)
    validation = ScaredStage2Dataset(scared_root=protocol.data_root, split_manifest=protocol.split_path, split="validation", phase="val", test_every=8)
    if len(train) != 235 or len(validation) != 32 or tuple(train.keyframe_entries) != EXPECTED_TRAIN or tuple(validation.keyframe_entries) != EXPECTED_VAL:
        raise RuntimeError("v3 dataset contract mismatch")
    protocol.run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "protocol_status": "final_vanilla_multisequence_stage2_control",
        "steps": protocol.steps,
        "scheduler_total_steps": protocol.scheduler_steps,
        "batch_size": 1,
        "precision": "FP32",
        "seed": protocol.seed,
        "optimizer": "AdamW",
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "eps": 1e-8,
        "gradient_clip_norm": 1.0,
        "loss_weights": {"disparity": 1.0, "render_l1": 0.8, "ssim": 0.2},
        "train_sequences": list(EXPECTED_TRAIN),
        "validation_sequences": list(EXPECTED_VAL),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "stage1_checkpoint_filename": protocol.stage1_path.name,
        "stage1_checkpoint_sha256": protocol.stage1_sha,
        "stage1_checkpoint_step": 7000,
        "reliable_endo_gs_sha": identities["source_sha"],
        "endo_e2e_gs_sha": identities["upstream_sha"],
        "manifest_sha256": identities["manifest_sha256"],
        "split_sha256": identities["split_sha256"],
        "config_sha256": identities["config_sha256"],
    }
    (protocol.run_dir / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    run = wandb.init(entity="azios1010-hanoi-university-of-science-and-technology", project="Reliable-GS", group="control-backbone", job_type="stage2-final", name="stage2-control-multiseq-v3-final-10k", config=metadata, dir=os.environ.get("WANDB_DIR"))
    model = StereoEndoModel(cfg, with_gs_render=True)
    checkpoint = torch.load(protocol.stage1_path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(checkpoint["network"], strict=False)
    if len(result.missing_keys) != 144 or result.unexpected_keys:
        raise RuntimeError("Stage-1 compatibility contract failed at launch")
    model.to(device).train()
    model.raft_stereo.freeze_bn()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5, eps=1e-8)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, total_steps=protocol.scheduler_steps, pct_start=0.01, cycle_momentum=False, anneal_strategy="linear")
    loader = make_loader(train, True, protocol.seed)
    iterator = iter(loader)
    counts: defaultdict[str, int] = defaultdict(int)
    best = float("inf")
    bg_color = list(cfg.dataset.bg_color)
    try:
        for step in range(1, protocol.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            sequence = sequence_from_name(batch["name"][0])
            counts[sequence] += 1
            batch = move_batch(batch, device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            data, loss, metrics = forward_loss(model, batch, bg_color)
            if not torch.isfinite(loss) or not tensor_finite(data):
                raise RuntimeError(f"non-finite forward at step {step}")
            health_stats = health(data)
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise RuntimeError(f"non-finite gradient at step {step}")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if step % protocol.val_interval == 0:
                validation_metrics = evaluate(model, validation, device, bg_color)
                val_log = {f"val/{key}": value for key, value in validation_metrics["aggregate"].items()}
                val_log.update({f"val_macro/{key}": value for key, value in validation_metrics["macro"].items()})
                run.log(val_log, step=step)
                if validation_metrics["macro"]["total_loss"] < best:
                    best = validation_metrics["macro"]["total_loss"]
                    save_checkpoint(protocol.run_dir / "best_val_total_loss.pth", model, optimizer, scheduler, step, protocol, identities, metadata | {"best_macro_total_loss": best, "validation": validation_metrics})
            record = {"step": step, **metrics, **health_stats, "lr": float(optimizer.param_groups[0]["lr"]), "gradient_norm": scalar(grad_norm), "step_time_sec": time.perf_counter() - start}
            run.log({f"train/{key}": value for key, value in record.items() if key != "step"}, step=step)
            if step % 1000 == 0:
                save_checkpoint(protocol.run_dir / f"stage2_multiseq_v3_step_{step:06d}.pth", model, optimizer, scheduler, step, protocol, identities, metadata)
                save_checkpoint(protocol.run_dir / "latest.pth", model, optimizer, scheduler, step, protocol, identities, metadata)
        if set(counts) != set(EXPECTED_TRAIN):
            raise RuntimeError(f"training sequence coverage failure: {dict(counts)}")
        save_checkpoint(protocol.run_dir / "GPS-GS_stage2_multiseq_v3_final_10k.pth", model, optimizer, scheduler, protocol.steps, protocol, identities, metadata | {"sequence_counts": dict(counts)})
        run.summary.update({"sequence_counts": dict(counts), "source_identity": "PASS", "protocol_status": metadata["protocol_status"]})
    finally:
        run.finish()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = resolve_protocol(args.repo.resolve(), args.config.resolve(), args.run_dir.resolve())
    result = dry_run(protocol) if args.dry_run else run(protocol)
    if result is not None:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
