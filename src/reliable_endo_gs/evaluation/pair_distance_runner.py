"""Pairwise 3D distance error of a frozen model on the SCARED test frames (every ``every``-th frame), one model per process.

The models are the entries frozen for the SCARED comparison; nothing is trained or tuned. The disparity a model returns for a test
frame is lifted with the canonical rectified geometry (as in the geometry benchmark), and the distances between pairs of surface
points are compared with the ground-truth distances (see ``pair_distance``). The run refuses to start unless the P1 freeze file
exists with its recorded SHA-256, the running source snapshot is the frozen one and the spec entry equals the frozen entry.
Rows are appended per frame, so an interrupted run continues.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, load_a01_manifest
from reliable_endo_gs.evaluation.comparison_runner import build_predictor, sha256_file
from reliable_endo_gs.evaluation.multi_model import finished_ids, run_rows, state_hash
from reliable_endo_gs.evaluation.pair_distance import pair_distance_metrics, seed_from_sample_id
from reliable_endo_gs.geometry.rectified import reconstruct_rectified
from reliable_endo_gs.teachers import load_a01_roles


def _guard(spec: dict[str, Any], name: str) -> None:
    freeze_path, sha_path = Path(spec["freeze"]["json"]), Path(spec["freeze"]["sha256_file"])
    if not freeze_path.exists() or not sha_path.exists():
        raise SystemExit("REFUSED: the P1 freeze file (or its hash file) does not exist")
    if sha256_file(freeze_path) != sha_path.read_text().split()[0]:
        raise SystemExit("REFUSED: the P1 freeze file changed since it was hashed")
    frozen = json.loads(freeze_path.read_text())
    if os.environ.get("SNAPSHOT_SHA256") != frozen.get("snapshot_tree_sha256"):
        raise SystemExit("REFUSED: the running source snapshot is not the frozen one")
    if frozen["models"].get(name) != spec["models"][name]:
        raise SystemExit(f"REFUSED: the spec entry of {name} differs from the frozen one")


def run_pair_distance(
    spec: dict[str, Any],
    name: str,
    out_dir: Path,
    *,
    every: int = 20,
    limit: int | None = None,
    smoke: bool = False,
    device: torch.device,
) -> dict[str, Any]:
    if not smoke:
        _guard(spec, name)
    data = spec["data"]["test"]
    roles = load_a01_roles(Path(spec["data"]["roles_config"]))
    manifest = load_a01_manifest(Path(data["manifest"]), roles=roles)
    adapter = ScaredA00Adapter(Path(spec["data"]["root"]), manifest, roles=roles)
    ids = [entry.sample_id for entry in manifest.entries if entry.data_role == data["role"]][
        ::every
    ][:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        (out_dir / "SMOKE.txt").write_text(
            "smoke run: not a result; analysis must ignore this folder\n"
        )
    predictor = build_predictor(spec["models"][name], adapter, data["role"], device)
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
            disparity = predictor(sample).disparity
            gt = adapter.load_supervision(sample)
            gt_valid = gt.valid_mask.to(device)
            predicted_valid = sample.batch.support & torch.isfinite(disparity)
            camera = sample.batch.camera
            gt_points = reconstruct_rectified(
                gt.disparity.to(device), camera, gt_valid, min_effective_disparity=1e-6
            )
            predicted_points = reconstruct_rectified(
                disparity, camera, predicted_valid, min_effective_disparity=1e-6
            )
            metrics = pair_distance_metrics(
                predicted_points.points.centers,
                gt_points.points.centers,
                gt_points.depth.valid_mask,
                predicted_points.depth.valid_mask,
                seed=seed_from_sample_id(sample_id),
            )
        return {"sample_id": sample_id, "metrics": {k: v.to_dict() for k, v in metrics.items()}}

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
                    "pair_distance": True,
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
        "metric": "pair_distance",
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
