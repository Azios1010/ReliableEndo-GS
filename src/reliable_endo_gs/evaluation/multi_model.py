"""Resumable per-frame evaluation of one model with the frozen metric functions (one model per process).

The loop is model-agnostic: ``predict(sample)`` returns a :class:`Prediction` (disparity always; renders and Gaussians when the
model has them). Geometry uses ``evaluate_geometry``, rendering uses ``psnr``/``windowed_ssim``/LPIPS, exactly as the paired
development benchmark does. Ground truth is loaded after the prediction through the same path for every model. Rows are
appended to ``rows.jsonl`` (``{"sample_id", "metrics": {name: MetricResult.to_dict()}}``), so an interrupted run continues
and ``summarize_metrics`` / the paired statistics read the same files.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.contracts import GaussianField, RenderOutput
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, ScaredA00Sample
from reliable_endo_gs.evaluation.a04_benchmark import _safe
from reliable_endo_gs.evaluation.benchmark_geometry import evaluate_geometry
from reliable_endo_gs.evaluation.perceptual import LPIPSAlexNet
from reliable_endo_gs.evaluation.rendering import psnr, windowed_ssim


@dataclass(frozen=True)
class Prediction:
    """Disparity [1,1,H,W] (d = u_left - u_right, input pixels); renders and Gaussians are optional."""

    disparity: torch.Tensor
    left: RenderOutput | None = None
    right: RenderOutput | None = None
    field: GaussianField | None = None
    source_index: torch.Tensor | None = None


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def finished_ids(rows_path: Path) -> set[str]:
    if not rows_path.exists():
        return set()
    return {json.loads(line)["sample_id"] for line in rows_path.read_text().splitlines() if line}


def run_rows(
    sample_ids: Sequence[str],
    rows_path: Path,
    make_row: Callable[[str], dict[str, Any]],
    *,
    progress_every: int = 200,
    progress: Callable[[dict[str, Any]], None] = lambda event: None,
) -> int:
    """Append one row per id that has none yet; returns the number of rows written in this call."""
    done = finished_ids(rows_path)
    written = 0
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("a") as out:
        for index, sample_id in enumerate(sample_ids):
            if sample_id in done:
                continue
            row = make_row(sample_id)
            if row["sample_id"] != sample_id:
                raise ValueError("row id differs from the requested id")
            out.write(json.dumps(_safe(row), allow_nan=False) + "\n")
            out.flush()
            written += 1
            if written == 1 or (index + 1) % progress_every == 0:
                progress({"done_in_file": len(done) + written, "total": len(sample_ids)})
    return written


def frame_row_builder(
    predict: Callable[[ScaredA00Sample], Prediction],
    adapter: ScaredA00Adapter,
    *,
    data_role: str,
    device: torch.device,
    perceptual: LPIPSAlexNet | None,
    output_hw: tuple[int, int] = (512, 640),
) -> Callable[[str], dict[str, Any]]:
    """Closure for :func:`run_rows`: load, predict, load GT, evaluate."""

    def make_row(sample_id: str) -> dict[str, Any]:
        sample = adapter.load(
            sample_id,
            data_role=data_role,
            include_supervision=False,
            output_hw=output_hw,
            device=device,
        )
        if sample.supervision is not None:
            raise ValueError("model input must not contain ground truth")
        with torch.no_grad():
            out = predict(sample)
            gt = adapter.load_supervision(sample)
            field = out.field if out.source_index is not None else None
            metrics, _ = evaluate_geometry(
                out.disparity,
                gt.disparity.to(device),
                gt.valid_mask.to(device),
                sample.batch.support & torch.isfinite(out.disparity),
                sample.batch.camera,
                cloud_max_points=1024,
                cloud_chunk_size=256,
                gaussian_centres=None if field is None else field.means3d,
                gaussian_valid=None if field is None else field.valid_mask,
                gaussian_source_index=out.source_index,
                rendered_depth=None if out.left is None else out.left.depth,
            )
            for view, render, target in (
                ("left", out.left, sample.batch.left),
                ("right", out.right, sample.batch.right),
            ):
                if render is None:
                    continue
                metrics[view + "_psnr"] = psnr(render, target)
                metrics[view + "_ssim"] = windowed_ssim(render.image, target)
                if perceptual is not None:
                    metrics[view + "_lpips_alex"] = perceptual.measure(render.image, target)
        return {"sample_id": sample_id, "metrics": {k: v.to_dict() for k, v in metrics.items()}}

    return make_row
