"""Resumable fine-tuning loop for a posed feed-forward Gaussian baseline (photometric loss on both input cameras).

The caller owns the ordered sample list (hashed, train split only), the loader and the forward function. The loop never
evaluates, never ranks checkpoints and never reads ground truth: the only comparison artifact is ``final.pt`` after exactly
``config.exposures`` exposures. ``resume.pt`` exists for crash recovery only.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch

ALLOWED_DATASETS = frozenset({"dataset_1", "dataset_2", "dataset_3", "dataset_6"})


@dataclass(frozen=True, slots=True)
class FinetuneConfig:
    """Fixed settings, written before any validation or test result exists."""

    kind: str
    exposures: int
    seed: int
    lr_main: float
    lr_pretrained: float | None
    weight_decay: float
    grad_clip: float
    checkpoint_every: int = 500
    log_every: int = 50
    max_consecutive_skips: int = 20
    max_skip_fraction: float = 0.005

    def __post_init__(self) -> None:
        if self.exposures < 1 or self.checkpoint_every < 1 or self.log_every < 1:
            raise ValueError("exposures, checkpoint_every and log_every must be positive")
        if not 0 < self.lr_main < 1 or self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def order_digest(ordered_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ordered_ids).encode()).hexdigest()


def replay_order(manifest_ids: Sequence[str], epochs: int) -> list[str]:
    """Replay the manifest order ``epochs`` times, with no shuffle (the historical student arms used manifest order)."""
    if epochs < 1 or not manifest_ids:
        raise ValueError("need at least one epoch and one sample")
    return [sid for _ in range(epochs) for sid in manifest_ids]


def validate_train_order(ordered_ids: Sequence[str], config: FinetuneConfig) -> None:
    """Refuse any id outside datasets 1/2/3/6 before a file is opened."""
    if len(ordered_ids) != config.exposures:
        raise ValueError(
            f"ordered ids ({len(ordered_ids)}) must equal exposures ({config.exposures})"
        )
    bad = sorted({s.split("/")[0] for s in ordered_ids} - ALLOWED_DATASETS)
    if bad:
        raise ValueError(f"training order contains forbidden datasets: {bad}")


def parameter_groups(predictor: nn.Module, config: FinetuneConfig) -> list[dict[str, Any]]:
    named = [(n, p) for n, p in predictor.named_parameters() if p.requires_grad]
    main = [p for n, p in named if "pretrained" not in n]
    groups: list[dict[str, Any]] = [{"params": main, "lr": config.lr_main}]
    pretrained = [p for n, p in named if "pretrained" in n]
    if pretrained:
        if config.lr_pretrained is None:
            raise ValueError("a pretrained parameter group exists but lr_pretrained is None")
        groups.append({"params": pretrained, "lr": config.lr_pretrained})
    return groups


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_finetune(
    predictor: nn.Module,
    config: FinetuneConfig,
    ordered_ids: Sequence[str],
    load: Callable[[str], DeploymentStereoBatch],
    forward_rgb: Callable[[nn.Module, DeploymentStereoBatch], torch.Tensor],
    loss_fn: Callable[[torch.Tensor, DeploymentStereoBatch], torch.Tensor],
    out_dir: Path,
    *,
    device: torch.device,
    provenance: dict[str, Any],
    prefetch: int = 2,
) -> dict[str, Any]:
    """Train ``predictor.encoder`` for ``config.exposures`` exposures; resume from ``resume.pt`` when present.

    ``load`` returns a CPU batch (RGB [1,3,H,W], support [1,1,H,W]); ``forward_rgb`` returns [1,2,3,H,W] and
    ``loss_fn`` a finite scalar. A non-finite loss raises ``FloatingPointError``. An optimizer step skipped by the grad
    scaler (non-finite gradients) is counted; too many consecutive or total skips raise ``RuntimeError``.
    """
    validate_train_order(ordered_ids, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path, resume_path, log_path = (
        out_dir / "final.pt",
        out_dir / "resume.pt",
        out_dir / "train_log.jsonl",
    )
    if final_path.exists():
        raise FileExistsError(f"{final_path} exists; refusing to overwrite a finished run")
    order_sha = order_digest(ordered_ids)
    order_file = out_dir / "order_ids.txt"
    if not order_file.exists():
        order_file.write_text("\n".join(ordered_ids) + "\n")
    elif order_file.read_text().split() != list(ordered_ids):
        raise RuntimeError(
            "order_ids.txt of this output directory differs from the requested order"
        )
    predictor.to(
        device
    ).train()  # before the optimizer exists, so its state is created and restored on the device
    optimizer = torch.optim.AdamW(
        parameter_groups(predictor, config), weight_decay=config.weight_decay, foreach=False
    )
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_cuda)
    start, skipped, consecutive = 0, 0, 0
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["config_digest"] != config.digest() or state["order_sha256"] != order_sha:
            raise RuntimeError("resume state does not match this configuration or sample order")
        predictor.encoder.load_state_dict(state["encoder"])  # type: ignore[attr-defined]
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, skipped = int(state["step"]), int(state["skipped"])
        _restore_rng(state["rng"])
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=max(1, prefetch))
    pending: dict[int, Future[DeploymentStereoBatch]] = {}

    def schedule(index: int) -> None:
        if index < config.exposures and index not in pending:
            pending[index] = pool.submit(load, ordered_ids[index])

    for ahead in range(start, min(start + prefetch + 1, config.exposures)):
        schedule(ahead)
    losses_window: list[float] = []
    try:
        with log_path.open("a") as log:
            for step in range(start, config.exposures):
                cpu_batch = pending.pop(step).result()
                schedule(step + prefetch + 1)
                batch = _to_device(cpu_batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                    rgb = forward_rgb(predictor, batch)
                loss = loss_fn(rgb, batch)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite loss at exposure {step} ({ordered_ids[step]})"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), config.grad_clip)
                )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                did_skip = use_cuda and scaler.get_scale() < scale_before
                if did_skip:
                    skipped += 1
                    consecutive += 1
                    if (
                        consecutive > config.max_consecutive_skips
                        or skipped > config.max_skip_fraction * config.exposures
                    ):
                        raise RuntimeError(
                            f"too many skipped optimizer steps ({skipped}, {consecutive} in a row)"
                        )
                else:
                    consecutive = 0
                losses_window.append(float(loss.detach()))
                done = step + 1
                if done % config.log_every == 0 or step == start:
                    log.write(
                        json.dumps(
                            {
                                "exposure": done,
                                "loss": losses_window[-1],
                                "loss_window_mean": sum(losses_window) / len(losses_window),
                                "grad_norm": grad_norm,
                                "skipped_total": skipped,
                                "seconds": time.monotonic() - started,
                                "sample_id": ordered_ids[step],
                                "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                                / 2**30
                                if use_cuda
                                else None,
                            }
                        )
                        + "\n"
                    )
                    log.flush()
                    losses_window.clear()
                if done % config.checkpoint_every == 0 and done < config.exposures:
                    _atomic_save(
                        {
                            "encoder": predictor.encoder.state_dict(),  # type: ignore[attr-defined]
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "step": done,
                            "skipped": skipped,
                            "rng": _rng_state(),
                            "config_digest": config.digest(),
                            "order_sha256": order_sha,
                        },
                        resume_path,
                    )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    summary = {
        "kind": config.kind,
        "exposures": config.exposures,
        "optimizer_steps_skipped": skipped,
        "config": asdict(config),
        "config_digest": config.digest(),
        "order_sha256": order_sha,
        "wall_seconds_this_process": time.monotonic() - started,
        "resumed_from": start,
        "provenance": provenance,
    }
    _atomic_save(
        {"encoder": predictor.encoder.state_dict(), "summary": summary},  # type: ignore[attr-defined]
        final_path,
    )
    summary["final_sha256"] = _file_sha256(final_path)
    (out_dir / "FINAL.json").write_text(json.dumps(summary, indent=2, default=str))
    if resume_path.exists():
        resume_path.unlink()
    return summary


def _to_device(batch: DeploymentStereoBatch, device: torch.device) -> DeploymentStereoBatch:
    from dataclasses import replace

    from reliable_endo_gs.contracts.deployment import RectifiedStereoCamera

    camera = batch.camera
    return replace(
        batch,
        left=batch.left.to(device),
        right=batch.right.to(device),
        support=batch.support.to(device),
        camera=RectifiedStereoCamera(
            camera.K_L.to(device), camera.K_R.to(device), camera.T_R_from_L.to(device), True
        ),
    )
