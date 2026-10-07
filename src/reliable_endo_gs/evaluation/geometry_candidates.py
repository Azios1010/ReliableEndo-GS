"""Bounded CPU geometry-only oracle screen; GT is loaded after interventions."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.actions.geometry_repair import GeometryRepair, load_geometry_repair_config
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, load_a01_manifest
from reliable_endo_gs.evaluation.depth import depth_mae, depth_rmse
from reliable_endo_gs.evaluation.stereo import endpoint_error
from reliable_endo_gs.geometry.rectified import reconstruct_rectified
from reliable_endo_gs.geometry.stereo_matching import correspondence_error
from reliable_endo_gs.reconstruction.student import StereoStudent
from reliable_endo_gs.regions.grid import grid_regions
from reliable_endo_gs.runtime.identity import source_sha256
from reliable_endo_gs.teachers import load_a01_roles
from reliable_endo_gs.training.student_checkpoint import read_student_checkpoint
from reliable_endo_gs.utils.io import write_json_atomic


def _metrics(
    disparity: torch.Tensor,
    gt_disparity: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    batch: DeploymentStereoBatch,
) -> dict[str, float | None]:
    geometry = reconstruct_rectified(
        disparity, batch.camera, batch.support, min_effective_disparity=1e-6
    )
    if bool((mask & ~geometry.depth.valid_mask).any()):
        raise ValueError("candidate lost valid metric geometry; denominator may not shrink")
    return {
        "depth_mae_mm": depth_mae(geometry.depth.depth * 1000, gt_depth * 1000, mask).value,
        "depth_rmse_mm": depth_rmse(geometry.depth.depth * 1000, gt_depth * 1000, mask).value,
        "disparity_epe_px": endpoint_error(disparity, gt_disparity, mask).value,
    }


def screen_geometry_candidates(
    *,
    checkpoint: Path,
    data_root: Path,
    manifest_path: Path,
    training_manifest: Path,
    roles_path: Path,
    candidate_configs: tuple[Path, ...],
    output_dir: Path,
    per_dataset: int = 2,
    minimum_local_gain_mm: float = 0.1,
) -> dict[str, object]:
    """Frozen stereo prediction, four local cores, fixed GT denominators.

    This screens disparity actions only. No Gaussian reconstruction, rendering,
    teacher or router runs, so it cannot establish full RGB/geometry utility.
    """
    if (
        type(per_dataset) is not int
        or not 1 <= per_dataset <= 4
        or not math.isfinite(minimum_local_gain_mm)
        or minimum_local_gain_mm <= 0
    ):
        raise ValueError("bounded screen needs 1..4 samples per dataset and positive gain margin")
    repo = Path(__file__).resolve().parents[3]
    if output_dir.resolve() == repo or repo in output_dir.resolve().parents:
        raise ValueError("geometry probe outputs must be outside repository")
    if output_dir.exists():
        raise FileExistsError("geometry probe output exists")
    configs = tuple(load_geometry_repair_config(path) for path in candidate_configs)
    if not configs or len({c.mode for c in configs}) != len(configs):
        raise ValueError("need unique geometry candidate modes")
    roles = load_a01_roles(roles_path)
    train = load_a01_manifest(training_manifest, roles=roles)
    development = load_a01_manifest(manifest_path, roles=roles)
    train_ids = {entry.sample_id for entry in train.entries}
    ids: list[str] = []
    for fold in range(1, 8):
        pool = [
            e.sample_id
            for e in development.entries
            if e.data_role == "final_train" and e.sample_id.startswith(f"dataset_{fold}/")
        ]
        if len(pool) < per_dataset:
            raise ValueError(f"insufficient development samples in dataset_{fold}")
        ids.extend(pool[i * len(pool) // per_dataset] for i in range(per_dataset))
    if train_ids.intersection(ids):
        raise ValueError("geometry development IDs overlap training")
    raw = read_student_checkpoint(checkpoint)
    if (
        raw["identity"]["manifest_hash"] != train.manifest_hash
        or raw["identity"]["role_hash"] != roles.role_hash
    ):
        raise ValueError("checkpoint training/role identity mismatch")
    cfg = raw["student_config"]
    model = StereoStudent(cfg["width"], cfg["bins"], cfg["minimum_effective_px"])
    model.load_state_dict(
        {
            k.removeprefix("stereo_student."): v
            for k, v in raw["model"].items()
            if k.startswith("stereo_student.")
        },
        strict=True,
    )
    model.eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    adapter = ScaredA00Adapter(data_root, development, roles=roles)
    hw = tuple(raw["construction_config"]["runtime"]["resolution_hw"])
    actions = tuple(GeometryRepair(c) for c in configs)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    with torch.no_grad(), (output_dir / "samples.jsonl").open("x") as stream:
        for sid in ids:
            sample = adapter.load(
                sid, data_role="final_train", include_supervision=False, output_hw=hw
            )
            assert sample.supervision is None
            batch = sample.batch
            disparity = model(batch)
            base_error, _ = correspondence_error(
                batch.left, batch.right, disparity, right_support=batch.support
            )
            regions = grid_regions(hw, ((hw[0] + 1) // 2, (hw[1] + 1) // 2), 0)
            executed = []
            # GT is unavailable to each action and to every acceptance decision.
            for region in regions:
                core = region.mask(batch.left)
                for action in actions:
                    t0 = time.perf_counter()
                    result = action.apply(batch, disparity, core)
                    elapsed = time.perf_counter() - t0
                    error, _ = correspondence_error(
                        batch.left, batch.right, result.disparity, right_support=batch.support
                    )
                    executed.append(
                        (
                            action.config.mode,
                            region.index,
                            core,
                            result,
                            elapsed,
                            float((base_error - error)[batch.support].mean()),
                        )
                    )
            gt = adapter.load(
                sid, data_role="final_train", include_supervision=True, output_hw=hw
            ).supervision
            assert gt is not None
            target = reconstruct_rectified(
                gt.disparity, batch.camera, gt.valid_mask, min_effective_disparity=1e-6
            )
            base_geometry = reconstruct_rectified(
                disparity, batch.camera, batch.support, min_effective_disparity=1e-6
            )
            fixed = target.depth.valid_mask & base_geometry.depth.valid_mask & batch.support
            base = _metrics(disparity, gt.disparity, target.depth.depth, fixed, batch)
            outcomes: list[dict[str, Any]] = []
            for mode, ri, core, result, seconds, warp_gain in executed:
                local_mask = fixed & core
                local_base = _metrics(
                    disparity, gt.disparity, target.depth.depth, local_mask, batch
                )
                global_after = _metrics(
                    result.disparity, gt.disparity, target.depth.depth, fixed, batch
                )
                local_after = _metrics(
                    result.disparity, gt.disparity, target.depth.depth, local_mask, batch
                )
                a, b = local_base["depth_mae_mm"], local_after["depth_mae_mm"]
                local_gain = a - b if a is not None and b is not None else None
                outcomes.append(
                    {
                        "mode": mode,
                        "region_id": ri,
                        "accepted_pixels": int(result.accepted.sum()),
                        "accepted_GT_pixels": int((result.accepted & fixed).sum()),
                        "cpu_action_seconds": seconds,
                        "global_warp_gain": warp_gain,
                        "global_after": global_after,
                        "local_before": local_base,
                        "local_after": local_after,
                        "local_depth_gain_mm": local_gain,
                    }
                )
            record = {
                "sample_id": sid,
                "GT_valid_pixels": int(fixed.sum()),
                "base": base,
                "outcomes": outcomes,
            }
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "event": "geometry_probe_progress",
                        "sample_id": sid,
                        "completed": len(records),
                    }
                ),
                flush=True,
            )
    summaries = {}
    for config in configs:
        rows = [r for case in records for r in case["outcomes"] if r["mode"] == config.mode]
        gains = [r["local_depth_gain_mm"] for r in rows if r["local_depth_gain_mm"] is not None]
        summaries[config.mode] = {
            "regions": len(rows),
            "GT_valid_regions": len(gains),
            "mean_local_depth_gain_mm": statistics.mean(gains) if gains else None,
            "regions_improved_at_least_margin": sum(g >= minimum_local_gain_mm for g in gains),
            "regions_worsened_at_least_margin": sum(g <= -minimum_local_gain_mm for g in gains),
            "accepted_pixels": sum(r["accepted_pixels"] for r in rows),
            "mean_cpu_action_seconds": statistics.mean(r["cpu_action_seconds"] for r in rows),
        }
    oracle: Counter[str] = Counter()
    for case in records:
        for ri in range(4):
            rows = [
                r
                for r in case["outcomes"]
                if r["region_id"] == ri and r["local_depth_gain_mm"] is not None
            ]
            best = max(rows, key=lambda r: r["local_depth_gain_mm"], default=None)
            oracle[
                best["mode"]
                if best and best["local_depth_gain_mm"] >= minimum_local_gain_mm
                else "STOP"
            ] += 1
    if not all(torch.equal(v, before[k]) for k, v in model.state_dict().items()):
        raise ValueError("geometry screen mutated frozen model")
    result_report: dict[str, object] = {
        "status": "cpu_geometry_candidate_screen_complete",
        "samples": len(records),
        "sample_ids": ids,
        "development_manifest_hash": development.manifest_hash,
        "training_manifest_hash": train.manifest_hash,
        "role_hash": roles.role_hash,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_source_sha256": raw["identity"]["source_sha256"],
        "probe_source_sha256": source_sha256(repo),
        "configs": [{"resolved": asdict(c), "hash": c.config_hash} for c in configs],
        "minimum_local_gain_mm": minimum_local_gain_mm,
        "summary": summaries,
        "metric_depth_oracle_region_choices": dict(oracle),
        "elapsed_seconds": time.perf_counter() - started,
        "device": "cpu",
        "torch": str(torch.__version__),
        "training_executed": False,
        "native_renderer_executed": False,
        "teacher_inference_executed": False,
        "model_weights_unchanged": True,
        "limitations": [
            "Two samples per fold by default; same-keyframe development, not final-test evidence.",
            "Offline GT depth oracle only; no RGB render, Gaussian-action comparison or selector learning.",
            "CPU stereo outputs may differ slightly from prior native GPU outputs.",
            "Candidate thresholds were declared before this screen; no calibrated confidence claim.",
        ],
    }
    write_json_atomic(output_dir / "result.json", result_report)
    return result_report
