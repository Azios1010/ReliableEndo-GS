"""Right-view renderings of one frozen model on a fixed list of frames (qualitative figure; no metric of the comparison changes).

The frames are chosen elsewhere by a rule written and hashed before any image exists (``QUALITATIVE_PROTOCOL.txt``). For each frame the run
stores the captured left and right images, the predicted disparity, the right-view rendering, its PSNR (same function as the comparison) and the
width of the right-edge strip that the left image cannot see. The reference model computes the strip width with
:func:`reliable_endo_gs.evaluation.visible_region.strip_width`; the other models read it, so every tile of a frame uses the same strip.
The run refuses to start unless the freeze file exists with its recorded SHA-256, the frame list matches the frozen hash, the running source
snapshot is the frozen one and the spec entry of the model equals the frozen entry.

Array shapes in the stored ``.npz``: ``left``, ``right``, ``right_render`` ``[3,H,W]`` in [0,1]; ``disparity`` ``[H,W]`` pixels.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, load_a01_manifest
from reliable_endo_gs.evaluation.comparison_runner import build_predictor, sha256_file
from reliable_endo_gs.evaluation.multi_model import state_hash
from reliable_endo_gs.evaluation.rendering import psnr
from reliable_endo_gs.evaluation.visible_region import strip_width
from reliable_endo_gs.teachers import load_a01_roles

REFERENCE_MODEL = "ours_refined"


def file_stem(sample_id: str) -> str:
    """``dataset_8/keyframe_0/frame_data000307`` -> ``dataset_8__keyframe_0__frame_data000307``."""
    if sample_id.count("/") != 2 or ".." in sample_id:
        raise ValueError(f"unexpected sample id {sample_id!r}")
    return sample_id.replace("/", "__")


def _guard(spec: dict[str, Any], name: str, frames_file: Path) -> None:
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
    if sha256_file(frames_file) != frozen.get("frames_sha256"):
        raise SystemExit("REFUSED: the frame list differs from the frozen one")


def run_qualitative(
    spec: dict[str, Any],
    name: str,
    frames_file: Path,
    out_dir: Path,
    *,
    reference_dir: Path | None,
    smoke: bool = False,
    device: torch.device,
) -> dict[str, Any]:
    if not smoke:
        _guard(spec, name, frames_file)
    if (name == REFERENCE_MODEL) != (reference_dir is None):
        raise SystemExit(
            f"the reference model {REFERENCE_MODEL} computes the strips; every other model needs --reference"
        )
    frames: dict[str, str] = json.loads(frames_file.read_text())["frames"]
    data = spec["data"]["test"]
    roles = load_a01_roles(Path(spec["data"]["roles_config"]))
    manifest = load_a01_manifest(Path(data["manifest"]), roles=roles)
    known = {e.sample_id for e in manifest.entries if e.data_role == data["role"]}
    unknown = sorted(set(frames.values()) - known)
    if unknown:
        raise SystemExit(f"frames not in the test split: {unknown}")
    adapter = ScaredA00Adapter(Path(spec["data"]["root"]), manifest, roles=roles)
    out_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        (out_dir / "SMOKE.txt").write_text("smoke run: not a result\n")
    predictor = build_predictor(spec["models"][name], adapter, data["role"], device)
    before = state_hash(predictor) if isinstance(predictor, nn.Module) else None
    started = time.monotonic()
    stored: dict[str, float] = {}
    for kind, sample_id in frames.items():
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
            stem = file_stem(sample_id)
            if reference_dir is None:
                strip = strip_width(out.disparity)
            else:
                with np.load(reference_dir / f"{stem}.npz") as ref:
                    strip = int(ref["strip_px"])
            right_psnr = float(psnr(out.right, sample.batch.right).to_dict()["value"])
        np.savez_compressed(
            out_dir / f"{stem}.npz",
            sample_id=sample_id,
            kind=kind,
            left=sample.batch.left[0].float().cpu().numpy(),
            right=sample.batch.right[0].float().cpu().numpy(),
            disparity=out.disparity[0, 0].float().cpu().numpy(),
            right_render=out.right.image[0].float().clamp(0, 1).cpu().numpy(),
            psnr_right=right_psnr,
            strip_px=strip,
        )
        stored[sample_id] = right_psnr
        print(json.dumps({"model": name, "frame": sample_id, "psnr_right": right_psnr}), flush=True)
    after = state_hash(predictor) if before is not None else None
    if before != after:
        raise RuntimeError("evaluation mutated the model state")
    provenance = {
        "model": name,
        "smoke": smoke,
        "frames": frames,
        "right_psnr": stored,
        "entry": spec["models"][name],
        "model_state_sha256": before,
        "source_snapshot_sha256": os.environ.get("SNAPSHOT_SHA256"),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "wall_seconds_this_process": time.monotonic() - started,
    }
    (out_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, default=str))
    return provenance
