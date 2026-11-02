"""Zero-shot rendering evaluation of frozen models on an external, test-only dataset (EndoNeRF sample data).

One model and one calibration variant per process. The models are exactly the entries frozen for the SCARED comparison; nothing
is trained, tuned or selected here. The external data have no independent ground truth, so only image-space metrics enter the
headline: PSNR, SSIM and LPIPS of both rendered input views against the captured images (full frame), and the left-view PSNR
outside the dataset's tool masks. One supplementary, scale-free number per frame compares the predicted disparity with the
dataset's own stereo-derived depth after a per-frame affine fit in inverse depth (``disp_affine_*``); it measures agreement with
that reference, not accuracy, and does not depend on the assumed baseline.

Rows are appended to ``rows.jsonl`` as in the SCARED runner, so an interrupted run continues. A non-smoke run refuses to start
unless the external freeze file exists, its SHA-256 matches the stored one, the running source snapshot is the frozen one, the
sidecar manifest is the frozen one and the spec entry of the model equals the frozen entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from reliable_endo_gs.data.endonerf import (
    EXTERNAL_TEST_ROLE,
    EndoNerfAdapter,
    EndoNerfConfig,
    OffsetMode,
)
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.evaluation.comparison_runner import build_predictor, sha256_file
from reliable_endo_gs.evaluation.multi_model import Prediction, finished_ids, run_rows, state_hash
from reliable_endo_gs.evaluation.perceptual import LPIPSAlexNet
from reliable_endo_gs.evaluation.rendering import psnr, windowed_ssim
from reliable_endo_gs.evaluation.stereo import MetricResult

_UNSUPPORTED_KINDS = (
    "endo_released",
    "stereo_matching",
    "teacher",
)  # need full-resolution frames or have no renders
_MIN_AFFINE_PIXELS = 1000


def disparity_affine_agreement(
    disparity: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor
) -> dict[str, MetricResult]:
    """Pearson r and RMSE (px) of ``disparity ~ a / depth + b`` over the valid pixels; ``a`` and ``b`` are fitted per frame.

    Shapes ``[1, 1, H, W]``. Valid pixels have positive depth, finite disparity and ``valid`` set. The fit removes the unknown depth
    unit, the baseline and the principal-point offset, so the numbers do not depend on the calibration assumptions.
    """
    keep = valid & (depth > 0) & torch.isfinite(disparity)
    count, total = int(keep.sum()), int(keep.numel())
    definition = "per-frame affine fit of predicted disparity on 1/(dataset depth), non-tool pixels; agreement with the dataset's own stereo-derived depth, not accuracy"
    if count < _MIN_AFFINE_PIXELS:
        return {
            "disp_affine_r": MetricResult("disp_affine_r", None, count, total, definition),
            "disp_affine_rmse_px": MetricResult(
                "disp_affine_rmse_px", None, count, total, definition
            ),
        }
    x = (1.0 / depth[keep]).double()
    y = disparity[keep].double()
    design = torch.stack([x, torch.ones_like(x)], dim=1)
    coefficients = torch.linalg.lstsq(design, y.unsqueeze(1)).solution.squeeze(1)
    residual = y - design @ coefficients
    rmse = float(residual.square().mean().sqrt())
    spread = float(x.std(unbiased=False) * y.std(unbiased=False))
    correlation = float(((x - x.mean()) * (y - y.mean())).mean() / spread) if spread > 0 else None
    return {
        "disp_affine_r": MetricResult("disp_affine_r", correlation, count, total, definition),
        "disp_affine_rmse_px": MetricResult("disp_affine_rmse_px", rmse, count, total, definition),
    }


def render_row_builder(
    predict: Callable[[ScaredA00Sample], Prediction],
    adapter: EndoNerfAdapter,
    *,
    device: torch.device,
    perceptual: LPIPSAlexNet | None,
) -> Callable[[str], dict[str, Any]]:
    """Closure for :func:`run_rows`: load the frame, predict, then score the renders against the captured images."""

    def make_row(sample_id: str) -> dict[str, Any]:
        sample = adapter.load(sample_id, device=device)
        if sample.supervision is not None:
            raise ValueError("model input must not contain ground truth")
        with torch.no_grad():
            out = predict(sample)
            tool = adapter.tool_mask(sample_id, device=device)
            metrics: dict[str, MetricResult] = {}
            for view, render, target in (
                ("left", out.left, sample.batch.left),
                ("right", out.right, sample.batch.right),
            ):
                if render is None:
                    raise ValueError("an external-evaluation model must render both input views")
                metrics[view + "_psnr"] = psnr(render, target)
                metrics[view + "_ssim"] = windowed_ssim(render.image, target)
                if perceptual is not None:
                    metrics[view + "_lpips_alex"] = perceptual.measure(render.image, target)
            assert out.left is not None
            metrics["left_psnr_no_tool"] = psnr(out.left, sample.batch.left, ~tool)
            metrics.update(
                disparity_affine_agreement(
                    out.disparity, adapter.reference_depth(sample_id, device=device), ~tool
                )
            )
        return {"sample_id": sample_id, "metrics": {k: v.to_dict() for k, v in metrics.items()}}

    return make_row


def _external_guard(spec: dict[str, Any], name: str, adapter: EndoNerfAdapter) -> None:
    freeze_path, sha_path = Path(spec["freeze"]["json"]), Path(spec["freeze"]["sha256_file"])
    if not freeze_path.exists() or not sha_path.exists():
        raise SystemExit("REFUSED: the external freeze file (or its hash file) does not exist")
    if sha256_file(freeze_path) != sha_path.read_text().split()[0]:
        raise SystemExit("REFUSED: the external freeze file changed since it was hashed")
    frozen = json.loads(freeze_path.read_text())
    if os.environ.get("SNAPSHOT_SHA256") != frozen.get("snapshot_tree_sha256"):
        raise SystemExit("REFUSED: the running source snapshot is not the frozen one")
    if frozen.get("sidecar_manifest_sha256") != adapter.manifest_sha256:
        raise SystemExit("REFUSED: the sidecar manifest is not the frozen one")
    if frozen["models"].get(name) != spec["models"][name]:
        raise SystemExit(f"REFUSED: the spec entry of {name} differs from the frozen one")


def run_external_model(
    spec: dict[str, Any],
    name: str,
    out_dir: Path,
    *,
    baseline_scale: float = 1.0,
    offset_mode: OffsetMode = "zero",
    every: int | None = None,
    limit: int | None = None,
    smoke: bool = False,
    device: torch.device,
) -> dict[str, Any]:
    entry = spec["models"][name]
    if entry["kind"] in _UNSUPPORTED_KINDS:
        raise ValueError(f"{name}: kind {entry['kind']!r} is not evaluated on the external data")
    external = spec["data"]["external"]
    adapter = EndoNerfAdapter(
        EndoNerfConfig(
            Path(external["dataset_root"]),
            Path(external["sidecar_dir"]),
            baseline_scale=baseline_scale,
            offset_mode=offset_mode,
        )
    )
    if not smoke:
        _external_guard(spec, name, adapter)
    ids = adapter.sample_ids
    if every:
        ids = ids[::every]
    ids = ids[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        (out_dir / "SMOKE.txt").write_text(
            "smoke run: not a result; analysis must ignore this folder\n"
        )
    predictor = build_predictor(entry, None, EXTERNAL_TEST_ROLE, device)  # type: ignore[arg-type]
    before = state_hash(predictor) if isinstance(predictor, nn.Module) else None
    perceptual = (
        LPIPSAlexNet(Path(spec["lpips"]["backbone"]), Path(spec["lpips"]["calibration"]))
        .to(device)
        .eval()
    )
    started = time.monotonic()

    def log(event: dict[str, Any]) -> None:
        print(
            json.dumps(
                {"model": name, "external": True, **event, "t": round(time.monotonic() - started)}
            ),
            flush=True,
        )

    rows_path = out_dir / "rows.jsonl"
    written = run_rows(
        ids,
        rows_path,
        render_row_builder(predictor, adapter, device=device, perceptual=perceptual),
        progress_every=20,
        progress=log,
    )
    after = state_hash(predictor) if before is not None else None
    if before != after:
        raise RuntimeError("evaluation mutated the model state")
    provenance = {
        "model": name,
        "dataset": "endonerf_sample_data (test only)",
        "smoke": smoke,
        "baseline_scale": baseline_scale,
        "offset_mode": offset_mode,
        "every": every,
        "frames_requested": len(ids),
        "frames_in_file": len(finished_ids(rows_path)),
        "frames_written_in_this_process": written,
        "entry": entry,
        "model_state_sha256": before,
        "sidecar_manifest_sha256": adapter.manifest_sha256,
        "project_head": subprocess.check_output(
            ["git", "-C", spec["repo"], "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_snapshot_sha256": os.environ.get("SNAPSHOT_SHA256"),
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "load_average": os.getloadavg(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda"
        else None,
        "wall_seconds_this_process": time.monotonic() - started,
    }
    (out_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, default=str))
    return provenance
