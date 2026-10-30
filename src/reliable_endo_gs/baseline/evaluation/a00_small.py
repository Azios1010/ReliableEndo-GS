"""Read-only output evaluation and immutable student reference comparison."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.baseline.a00_model import A00EndoModel, render_a00_endo
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample, ScaredSupervision
from reliable_endo_gs.evaluation.benchmark_geometry import evaluate_geometry
from reliable_endo_gs.evaluation.rendering import psnr, windowed_ssim
from reliable_endo_gs.rendering.production import ProductionFieldRenderer


def _state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate_a00_small(
    model: A00EndoModel,
    samples: list[ScaredA00Sample],
    truths: dict[str, ScaredSupervision],
    renderer: ProductionFieldRenderer,
    output: Path,
    *,
    stage: str,
    stage_updates: int,
    total_updates: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate the fixed fourteen samples without changing weights, buffers or RNG."""
    before = _state_sha(model)
    rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    flags = [(m, m.training) for m in model.modules()]
    rows = []
    evaluation_started = time.monotonic()
    try:
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                warm = model(
                    samples[0].batch,
                    with_gaussians=stage == "stage2",
                    mixed_precision=stage == "stage2",
                )
                if stage == "stage2":
                    for view in ("left", "right"):
                        render_a00_endo(warm, samples[0].batch, renderer, view=view)
            torch.cuda.synchronize(device)
            for sample in samples:
                batch = sample.batch
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                prediction = model(
                    batch,
                    with_gaussians=stage == "stage2",
                    mixed_precision=stage == "stage2",
                )
                renders = {}
                if stage == "stage2":
                    renders = {
                        v: render_a00_endo(prediction, batch, renderer, view=v)
                        for v in ("left", "right")
                    }
                torch.cuda.synchronize(device)
                elapsed_ms = 1000 * (time.perf_counter() - start)
                gt = truths[sample.sample_id]
                field = prediction.field
                geometry, _ = evaluate_geometry(
                    prediction.disparity,
                    gt.disparity,
                    gt.valid_mask,
                    batch.support,
                    batch.camera,
                    cloud_max_points=256,
                    cloud_chunk_size=128,
                    gaussian_centres=field.means3d if field is not None else None,
                    gaussian_valid=field.valid_mask if field is not None else None,
                    gaussian_source_index=torch.arange(512 * 640, device=device)[None]
                    if field is not None
                    else None,
                )
                metrics = {k: v.to_dict() for k, v in geometry.items()}
                for view, render in renders.items():
                    image = getattr(batch, view)
                    metrics[f"{view}_psnr"] = psnr(render, image).to_dict()
                    metrics[f"{view}_ssim"] = windowed_ssim(render.image, image).to_dict()
                    if total_updates == 112:
                        from PIL import Image

                        pixels = (
                            (render.image[0].clamp(0, 1) * 255)
                            .round()
                            .byte()
                            .permute(1, 2, 0)
                            .cpu()
                            .numpy()
                        )
                        Image.fromarray(pixels).save(
                            output / f"{sample.sample_id.replace('/', '_')}_{view}.png"
                        )
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "stage": stage,
                        "stage_updates": stage_updates,
                        "total_updates": total_updates,
                        "metrics": metrics,
                        "forward_ms": elapsed_ms,
                    }
                )
    finally:
        for m, flag in flags:
            m.training = flag
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1], device)
    if _state_sha(model) != before:
        raise ValueError("evaluation mutated model weights/buffers")
    path = output / f"validation-{stage}-{stage_updates:03d}.jsonl"
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = {}
    for key in rows[0]["metrics"]:
        values = [
            r["metrics"][key]["value"] for r in rows if r["metrics"][key]["value"] is not None
        ]
        summary[key] = {
            "mean": statistics.mean(values) if values else None,
            "available": len(values),
        }
    entry = {
        "stage": stage,
        "stage_updates": stage_updates,
        "total_updates": total_updates,
        "metrics": summary,
        "evaluation_seconds": time.monotonic() - evaluation_started,
        "mean_forward_ms": statistics.mean(r["forward_ms"] for r in rows),
        "weights_unchanged": True,
    }
    return entry
