"""Right-view metrics on the part of the right image that the left image can see, one frozen model per process.

A model whose Gaussians are anchored in the left image only (ours, Endo-E2E-GS) has no Gaussian for the strip at the right edge of the right
view: a left pixel at the last column lands ``disparity`` pixels to the left in the right view, so the columns beyond that are not covered.
The strip width is taken from the predicted disparity of the frozen student at the border (``strip_width``): the 95th percentile over the last
``BORDER_COLUMNS`` columns, rounded up. The width of every frame is computed once by the reference model (the unrefined student) and read by the
other models, so all models are cropped identically. The right view and the captured right image are cropped to the visible columns and PSNR,
windowed SSIM and LPIPS are computed on the crops; the full-frame right PSNR is recomputed in the same run as a check against the main comparison.
This is an additional, post hoc metric: the headline metrics of the comparison are unchanged.
The run refuses to start unless the freeze file exists with its recorded SHA-256, the running source snapshot is the frozen one and the spec
entry of the model equals the frozen entry.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, load_a01_manifest
from reliable_endo_gs.evaluation.comparison_runner import build_predictor, sha256_file
from reliable_endo_gs.evaluation.multi_model import finished_ids, run_rows, state_hash
from reliable_endo_gs.evaluation.perceptual import LPIPSAlexNet
from reliable_endo_gs.evaluation.rendering import psnr, windowed_ssim
from reliable_endo_gs.teachers import load_a01_roles

BORDER_COLUMNS = 8
STRIP_QUANTILE = 0.95
REFERENCE_MODEL = "ours_base"


def strip_width(disparity: torch.Tensor) -> int:
    """Width in pixels of the right-view strip not visible from the left image; ``disparity`` is [1,1,H,W] in pixels (d = u_left - u_right)."""
    if disparity.ndim != 4 or disparity.shape[0] != 1 or disparity.shape[1] != 1:
        raise ValueError("disparity must have shape [1,1,H,W]")
    border = disparity[..., -BORDER_COLUMNS:]
    values = border[torch.isfinite(border)]
    if values.numel() == 0:
        raise ValueError("no finite disparity at the right border")
    return int(math.ceil(float(torch.quantile(values.float().flatten(), STRIP_QUANTILE))))


def crop_columns(image: torch.Tensor, strip_px: int) -> torch.Tensor:
    """Drop the last ``strip_px`` columns of an image [..., H, W]; at least one column must remain."""
    width = image.shape[-1]
    if strip_px < 0 or strip_px >= width:
        raise ValueError(f"strip of {strip_px} px does not fit an image of width {width}")
    return image[..., : width - strip_px]


def _guard(spec: dict[str, Any], name: str) -> None:
    freeze_path, sha_path = Path(spec["freeze"]["json"]), Path(spec["freeze"]["sha256_file"])
    if not freeze_path.exists() or not sha_path.exists():
        raise SystemExit("REFUSED: the freeze file (or its hash file) does not exist")
    if sha256_file(freeze_path) != sha_path.read_text().split()[0]:
        raise SystemExit("REFUSED: the freeze file changed since it was hashed")
    frozen = json.loads(freeze_path.read_text())
    if os.environ.get("SNAPSHOT_SHA256") != frozen.get("snapshot_tree_sha256"):
        raise SystemExit("REFUSED: the running source snapshot is not the frozen one")
    if frozen["models"].get(name) != spec["models"][name]:
        raise SystemExit(f"REFUSED: the spec entry of {name} differs from the frozen one")


def _reference_strips(path: Path) -> dict[str, int]:
    if not path.exists():
        raise SystemExit(
            f"REFUSED: the reference rows {path} do not exist (run {REFERENCE_MODEL} first)"
        )
    return {
        json.loads(line)["sample_id"]: int(json.loads(line)["metrics"]["strip_px"]["value"])
        for line in path.read_text().splitlines()
        if line
    }


def run_visible(
    spec: dict[str, Any],
    name: str,
    out_dir: Path,
    *,
    reference_rows: Path | None,
    every: int = 20,
    limit: int | None = None,
    smoke: bool = False,
    device: torch.device,
) -> dict[str, Any]:
    if not smoke:
        _guard(spec, name)
    if (name == REFERENCE_MODEL) != (reference_rows is None):
        raise SystemExit(
            f"the reference model {REFERENCE_MODEL} computes the strips; every other model needs --reference"
        )
    data = spec["data"]["test"]
    roles = load_a01_roles(Path(spec["data"]["roles_config"]))
    manifest = load_a01_manifest(Path(data["manifest"]), roles=roles)
    adapter = ScaredA00Adapter(Path(spec["data"]["root"]), manifest, roles=roles)
    ids = [e.sample_id for e in manifest.entries if e.data_role == data["role"]][::every][:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        (out_dir / "SMOKE.txt").write_text(
            "smoke run: not a result; analysis must ignore this folder\n"
        )
    strips = _reference_strips(reference_rows) if reference_rows is not None else {}
    predictor = build_predictor(spec["models"][name], adapter, data["role"], device)
    perceptual = (
        LPIPSAlexNet(Path(spec["lpips"]["backbone"]), Path(spec["lpips"]["calibration"]))
        .to(device)
        .eval()
    )
    before = state_hash(predictor) if isinstance(predictor, nn.Module) else None
    started = time.monotonic()

    def make_row(sample_id: str) -> dict[str, Any]:
        sample = adapter.load(
            sample_id,
            data_role=data["role"],
            include_supervision=False,
            output_hw=(512, 640),
            device=device,
        )
        with torch.no_grad():
            out = predictor(sample)
            if out.right is None:
                raise ValueError(f"{name} returns no right-view render")
            strip = strip_width(out.disparity) if reference_rows is None else strips[sample_id]
            target = sample.batch.right
            render = replace(out.right, image=crop_columns(out.right.image, strip))
            cropped_target = crop_columns(target, strip)
            metrics = {
                "right_psnr_full": psnr(out.right, target).to_dict(),
                "right_psnr_visible": psnr(render, cropped_target).to_dict(),
                "right_ssim_visible": windowed_ssim(render.image, cropped_target).to_dict(),
                "right_lpips_visible": perceptual.measure(render.image, cropped_target).to_dict(),
                "strip_px": {"value": float(strip)},
            }
        return {"sample_id": sample_id, "metrics": metrics}

    rows_path = out_dir / "rows.jsonl"
    written = run_rows(
        ids,
        rows_path,
        make_row,
        progress_every=20,
        progress=lambda event: print(
            json.dumps(
                {
                    "model": name,
                    "visible_region": True,
                    **event,
                    "t": round(time.monotonic() - started),
                }
            ),
            flush=True,
        ),
    )
    after = state_hash(predictor) if before is not None else None
    if before != after:
        raise RuntimeError("evaluation mutated the model state")
    provenance = {
        "model": name,
        "metric": "visible_region",
        "smoke": smoke,
        "every": every,
        "frames_requested": len(ids),
        "frames_in_file": len(finished_ids(rows_path)),
        "frames_written_in_this_process": written,
        "entry": spec["models"][name],
        "model_state_sha256": before,
        "source_snapshot_sha256": os.environ.get("SNAPSHOT_SHA256"),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "load_average": os.getloadavg(),
        "wall_seconds_this_process": time.monotonic() - started,
    }
    (out_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, default=str))
    return provenance
