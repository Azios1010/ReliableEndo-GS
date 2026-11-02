"""One model, one split, one process: builds the predictor from a spec entry and writes resumable per-frame rows (and shift rows).

SPLIT ``val`` evaluates the 56 benchmark validation frames (role final_train) and is the parity gate. SPLIT ``test`` opens
final_test (datasets 8, 9) and refuses to start unless the freeze file exists and its SHA-256 equals the stored one, and the spec
entry of the model equals the entry recorded in the freeze. Geometry-only models yield rows without render metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, ScaredA00Sample, load_a01_manifest
from reliable_endo_gs.evaluation.disparity_shift import SHIFTS, shift_stats, shifted_batch
from reliable_endo_gs.evaluation.multi_model import (
    Prediction,
    finished_ids,
    frame_row_builder,
    run_rows,
    state_hash,
)
from reliable_endo_gs.evaluation.perceptual import LPIPSAlexNet
from reliable_endo_gs.teachers import load_a01_roles


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class TeacherPredictor(nn.Module):
    """Frozen FoundationStereo (geometry only). Run in the teacher environment."""

    def __init__(self) -> None:
        super().__init__()
        from reliable_endo_gs.reconstruction.model import _freeze_injected_teacher
        from reliable_endo_gs.teachers.workflow import load_teacher_factory

        self.teacher = load_teacher_factory(
            "reliable_endo_gs.teachers.external:build_foundation_stereo"
        )
        _freeze_injected_teacher(self.teacher)

    def forward(self, sample: ScaredA00Sample) -> Prediction:
        with torch.inference_mode():
            disparity = self.teacher.infer(sample.batch.left, sample.batch.right).disparity.clone()
        return Prediction(disparity)


def build_predictor(
    entry: dict[str, Any], adapter: ScaredA00Adapter, data_role: str, device: torch.device
) -> nn.Module:
    kind = entry["kind"]
    if kind == "student":
        from reliable_endo_gs.evaluation.ours import StudentPredictor, build_student, load_refiner

        model, epsilon = build_student(
            Path(entry["stage_a_checkpoint"]),
            entry["stage_a_sha256"],
            device,
            weights_from=Path(entry["weights_from"]) if entry.get("weights_from") else None,
            weights_from_sha256=entry.get("weights_from_sha256"),
            grid_stride=entry.get("grid_stride"),
            grid_cap=entry.get("grid_cap"),
        )
        refiner = None
        if entry.get("refiner_dir"):
            refiner_dir = Path(entry["refiner_dir"])
            if sha256_file(refiner_dir / "refiner.pt") != entry["refiner_sha256"]:
                raise ValueError("refiner hash differs from the frozen one")
            refiner = load_refiner(refiner_dir, model, device)
        return StudentPredictor(model, refiner, epsilon).eval()
    if kind == "endo_matched":
        from reliable_endo_gs.baseline.comparison.endo_e2e import build_matched

        return build_matched(
            Path(entry["checkpoint"]), entry["sha256"], Path(entry["upstream"]), device
        )
    if kind == "endo_released":
        from reliable_endo_gs.baseline.comparison.endo_e2e import build_released

        return build_released(
            Path(entry["checkpoint"]),
            entry["sha256"],
            Path(entry["upstream"]),
            adapter,
            data_role,
            device,
        )
    if kind == "posed_ft":
        from reliable_endo_gs.baseline.comparison.posed_ft import build_finetuned

        return build_finetuned(
            entry["model"],
            Path(entry["upstream"]),
            entry["upstream_commit"],
            Path(entry["official_checkpoint"]),
            Path(entry["finetuned_final"]) if entry.get("finetuned_final") else None,
            entry["official_checkpoint_sha256"],
            entry.get("finetuned_final_sha256"),
            entry["near_m"],
            entry["far_m"],
            device,
            left_source_only=bool(entry.get("left_source_only", False)),
        )
    if kind == "teacher":
        return TeacherPredictor().to(device).eval()
    if kind == "stereo_matching":
        from reliable_endo_gs.baseline.stereo.predictors import build_stereo_predictor

        return build_stereo_predictor(entry, adapter, data_role, device)
    raise ValueError(f"unknown model kind {kind!r}")


def _freeze_guard(spec: dict[str, Any], name: str) -> dict[str, Any]:
    freeze_path, sha_path = Path(spec["freeze"]["json"]), Path(spec["freeze"]["sha256_file"])
    if not freeze_path.exists() or not sha_path.exists():
        raise SystemExit("REFUSED: FREEZE_COMPARISON.json (or its hash file) does not exist")
    if sha256_file(freeze_path) != sha_path.read_text().split()[0]:
        raise SystemExit("REFUSED: FREEZE_COMPARISON.json changed since it was hashed")
    frozen = json.loads(freeze_path.read_text())
    if os.environ.get("SNAPSHOT_SHA256") != frozen.get("snapshot_tree_sha256"):
        raise SystemExit("REFUSED: the running source snapshot is not the frozen one")
    if frozen["models"].get(name) != spec["models"][name]:
        raise SystemExit(f"REFUSED: the spec entry of {name} differs from the frozen one")
    return frozen


def run_model(
    spec: dict[str, Any],
    name: str,
    split: str,
    out_dir: Path,
    *,
    shift: bool = False,
    every: int | None = None,
    limit: int | None = None,
    device: torch.device,
) -> dict[str, Any]:
    if split not in ("val", "test"):
        raise ValueError("split must be val or test")
    if split == "test":
        _freeze_guard(spec, name)
    data = spec["data"][split]
    roles = load_a01_roles(Path(spec["data"]["roles_config"]))
    manifest = load_a01_manifest(Path(data["manifest"]), roles=roles)
    adapter = ScaredA00Adapter(Path(spec["data"]["root"]), manifest, roles=roles)
    if split == "val":
        listing = json.loads(Path(data["ids_file"]).read_text())
        ids = list(listing["ids"] if "ids" in listing else listing["sample_ids"])
    else:
        ids = [entry.sample_id for entry in manifest.entries if entry.data_role == data["role"]]
    if every is None and shift:
        every = 20
    if every:
        ids = ids[::every]
    ids = ids[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_predictor(spec["models"][name], adapter, data["role"], device)
    before = state_hash(predictor) if isinstance(predictor, nn.Module) else None
    perceptual = None
    if spec.get("lpips") and not shift:
        perceptual = (
            LPIPSAlexNet(Path(spec["lpips"]["backbone"]), Path(spec["lpips"]["calibration"]))
            .to(device)
            .eval()
        )
    started = time.monotonic()

    def log(event: dict[str, Any]) -> None:
        print(
            json.dumps(
                {"model": name, "split": split, **event, "t": round(time.monotonic() - started)}
            ),
            flush=True,
        )

    if shift:
        rows_path = out_dir / "shift_rows.jsonl"

        def make_shift_row(sample_id: str) -> dict[str, Any]:
            sample = adapter.load(
                sample_id,
                data_role=data["role"],
                include_supervision=False,
                output_hw=(512, 640),
                device=device,
            )
            gt = adapter.load_supervision(sample)
            stats = []
            with torch.no_grad():
                for delta in SHIFTS:
                    batch, gt_disparity, gt_valid = shifted_batch(
                        sample, gt.disparity.to(device), gt.valid_mask.to(device), delta
                    )
                    prediction = predictor(replace(sample, batch=batch)).disparity
                    stats.append(
                        {
                            "shift": delta,
                            **(
                                shift_stats(prediction, gt_disparity, gt_valid, batch.support)
                                or {"n": 0}
                            ),
                        }
                    )
            return {"sample_id": sample_id, "shifts": stats}

        written = run_rows(ids, rows_path, make_shift_row, progress_every=20, progress=log)
    else:
        rows_path = out_dir / "rows.jsonl"
        make_row = frame_row_builder(
            predictor, adapter, data_role=data["role"], device=device, perceptual=perceptual
        )
        written = run_rows(ids, rows_path, make_row, progress=log)
    after = state_hash(predictor) if before is not None else None
    if before != after:
        raise RuntimeError("evaluation mutated the model state")
    provenance = {
        "model": name,
        "split": split,
        "shift": shift,
        "every": every,
        "frames_requested": len(ids),
        "frames_in_file": len(finished_ids(rows_path)),
        "frames_written_in_this_process": written,
        "entry": spec["models"][name],
        "model_state_sha256": before,
        "project_head": subprocess.check_output(
            ["git", "-C", spec["repo"], "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_snapshot_sha256": os.environ.get("SNAPSHOT_SHA256"),
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "torch": torch.__version__,
        "load_average": os.getloadavg(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda"
        else None,
        "wall_seconds_this_process": time.monotonic() - started,
    }
    (out_dir / ("PROVENANCE_shift.json" if shift else "PROVENANCE.json")).write_text(
        json.dumps(provenance, indent=2, default=str)
    )
    return provenance
