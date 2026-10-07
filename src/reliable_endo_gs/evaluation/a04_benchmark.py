"""Frozen A04 benchmark orchestration; no optimizer, training or GT model input."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, load_a01_manifest
from reliable_endo_gs.evaluation.benchmark_geometry import evaluate_geometry
from reliable_endo_gs.evaluation.benchmark_protocol import BenchmarkProtocol
from reliable_endo_gs.evaluation.perceptual import LPIPSAlexNet, file_sha256
from reliable_endo_gs.evaluation.rendering import psnr, windowed_ssim
from reliable_endo_gs.evaluation.stereo import MetricResult
from reliable_endo_gs.profiling.runtime import profile_callable
from reliable_endo_gs.reconstruction import (
    ReferenceConfig,
    ReferenceReconstructionModel,
    load_reference_checkpoint,
)
from reliable_endo_gs.rendering import ProductionFieldRenderer
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.teachers.external import build_foundation_stereo
from reliable_endo_gs.teachers.roles import load_a01_roles
from reliable_endo_gs.utils.seed import set_seed


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("NaN benchmark result")
        return "+inf" if value > 0 else "-inf"
    if isinstance(value, Mapping):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    return value


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False, encoding="utf-8"
        ) as handle:
            temporary = Path(handle.name)
            json.dump(_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_table(path: Path, summary: Mapping[str, Any]) -> None:
    """Export generated scalars and denominators; never edit metric values."""
    with io.StringIO(newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scope",
                "metric",
                "frame_mean",
                "valid_weighted",
                "available_frames",
                "unavailable_frames",
                "valid_count",
                "total_count",
                "definition",
            ),
        )
        writer.writeheader()
        groups = {
            "overall": summary["overall"],
            **summary["per_dataset"],
            **summary["per_sequence"],
        }
        for scope, metrics in groups.items():
            for metric, value in metrics.items():
                writer.writerow(_safe({"scope": scope, "metric": metric, **value}))
        text = handle.getvalue()
    if path.exists():
        if path.read_bytes() == text.encode():
            return
        raise ValueError("existing benchmark CSV differs from the canonical frame records")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False, encoding="utf-8", newline=""
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def select_samples(
    adapter: ScaredA00Adapter, *, sample_ids: Sequence[str] = (), max_samples: int | None = None
) -> tuple[str, ...]:
    """Select only final_test; bounded selection spans every available keyframe."""
    pool = tuple(e.sample_id for e in adapter.manifest.entries if e.data_role == "final_test")
    if not pool:
        raise ValueError("manifest contains no final_test samples")
    if sample_ids and max_samples is not None:
        raise ValueError("choose explicit sample IDs or --max-samples, not both")
    if sample_ids:
        if len(set(sample_ids)) != len(sample_ids) or not set(sample_ids) <= set(pool):
            raise ValueError("sample IDs must be unique manifest final_test entries")
        return tuple(sample_ids)
    if max_samples is None:
        return pool
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    if max_samples >= len(pool):
        return pool
    # Deterministic round-robin equispaced positions within each keyframe.
    groups: dict[str, list[str]] = {}
    for sample in pool:
        groups.setdefault(sample.rsplit("/", 1)[0], []).append(sample)
    counts = {key: 0 for key in groups}
    for _ in range(max_samples):
        key = min((k for k in groups if counts[k] < len(groups[k])), key=lambda k: (counts[k], k))
        counts[key] += 1
    selected = set()
    for key, group in groups.items():
        n = counts[key]
        if n == 1:
            selected.add(group[0])
        elif n > 1:
            selected.update(group[round((len(group) - 1) * j / (n - 1))] for j in range(n))
    return tuple(i for i in pool if i in selected)


def summarize_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Retain frame means, valid-count weighting and missing metric counts.

    Frame means match the upstream validation's mean of per-frame scalars.
    RMSE in the valid-weighted column pools squared errors before taking sqrt.
    This function accepts records independent of the inference backend.
    """
    names = sorted({name for row in rows for name in row["metrics"]})
    summaries = {}
    for name in names:
        values = [row["metrics"][name] for row in rows if name in row["metrics"]]
        present = [v for v in values if v["value"] is not None]
        numeric = [(float(v["value"]), int(v["valid_count"])) for v in present]
        definitions = {v["definition"] for v in values}
        if len(definitions) != 1:
            raise ValueError(f"incompatible definitions for {name}")
        count = sum(n for _, n in numeric)
        if count:
            pooled = sum((v * v if "rmse" in name else v) * n for v, n in numeric) / count
            weighted = math.sqrt(pooled) if "rmse" in name else pooled
        else:
            weighted = None
        summaries[name] = {
            "frame_mean": statistics.fmean(v for v, _ in numeric) if numeric else None,
            "valid_weighted": weighted,
            "available_frames": len(present),
            "unavailable_frames": len(rows) - len(present),
            "valid_count": count,
            "total_count": sum(int(v["total_count"]) for v in values),
            "definition": next(iter(definitions)),
        }
    return summaries


def _deployment_to_device(
    batch: DeploymentStereoBatch, device: torch.device
) -> DeploymentStereoBatch:
    c = batch.camera
    camera = RectifiedStereoCamera(
        c.K_L.to(device), c.K_R.to(device), c.T_R_from_L.to(device), True
    )
    return DeploymentStereoBatch(
        batch.left.to(device), batch.right.to(device), camera, batch.support.to(device)
    )


def run_a04_benchmark(
    *,
    checkpoint: Path,
    data_root: Path,
    manifest: Path,
    roles_path: Path,
    protocol_path: Path,
    output_dir: Path,
    lpips_backbone: Path,
    lpips_calibration: Path,
    sample_ids: Sequence[str] = (),
    max_samples: int | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Strict-load original training config; evaluate final_test without changing it.

    Output directory must be outside source and data. Per-frame JSONL is flushed
    and can be resumed with identical code/artifact/protocol/sample identities.
    Failed frames stop the run with an error artifact, never disappear from a
    supposedly complete report. All finite zero-visibility RGB renders are scored.
    """
    repo = Path(__file__).resolve().parents[3]
    out = output_dir.expanduser().resolve()
    data_root = data_root.expanduser().resolve(strict=True)
    if out.is_relative_to(repo) or out.is_relative_to(data_root):
        raise ValueError("benchmark outputs must be external to source and dataset")
    if out.exists() and not resume:
        raise ValueError("refusing to overwrite an existing benchmark directory")
    protocol = BenchmarkProtocol.load(protocol_path)
    roles = load_a01_roles(roles_path)
    adapter = ScaredA00Adapter(data_root, load_a01_manifest(manifest, roles=roles), roles=roles)
    ids = select_samples(adapter, sample_ids=sample_ids, max_samples=max_samples)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = ReferenceConfig.from_dict(payload["config"])
    if payload["config_hash"] != config.config_hash:
        raise ValueError("saved checkpoint configuration hash mismatch")
    device = torch.device(config.runtime.device)
    if device.type == "cuda":
        if device.index not in (None, 0) or os.environ.get("CUDA_VISIBLE_DEVICES", "").count(","):
            raise ValueError(
                "mask one physical GPU with CUDA_VISIBLE_DEVICES; native renderer requires cuda:0"
            )
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    resolution = protocol.resolution_hw or config.runtime.resolution_hw
    if resolution is None:
        raise ValueError("declare benchmark resolution explicitly")
    set_seed(config.seed)
    # Validate local perceptual assets before any teacher inference. No downloads.
    perceptual = LPIPSAlexNet(lpips_backbone, lpips_calibration)
    renderer = ProductionFieldRenderer()
    model = ReferenceReconstructionModel(
        config=config, teacher=build_foundation_stereo(), renderer=renderer
    ).to(device)
    provenance = load_reference_checkpoint(checkpoint, model)
    if provenance.renderer_identity != renderer.identity:
        raise ValueError("benchmark renderer differs from the checkpoint renderer")
    model.eval()
    driver = None
    if device.type == "cuda":
        try:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,driver_version,memory.total",
                    "--format=csv",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            driver = {
                "returncode": query.returncode,
                "inventory_csv": query.stdout,
                "error": query.stderr,
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            driver = {"unavailable_reason": str(error)}
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    source_root = repo / "src/reliable_endo_gs"
    digest = hashlib.sha256()
    for p in sorted(source_root.rglob("*.py")):
        digest.update(p.relative_to(source_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(p.read_bytes())
    identity = {
        "schema": "a04_benchmark_identity.v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "training_config": config.to_dict(),
        "training_config_hash": config.config_hash,
        "protocol": protocol.to_dict(),
        "protocol_hash": protocol.identity_hash,
        "resolution_hw": resolution,
        "manifest_hash": adapter.manifest.manifest_hash,
        "role_hash": roles.role_hash,
        "data_root": str(data_root),
        "sample_ids": ids,
        "dataset_counts": dict(Counter(i.split("/")[0] for i in ids)),
        "teacher_identity": provenance.to_dict()["teacher_identity"],
        "renderer": renderer.identity,
        "lpips": perceptual.identity,
        "git_commit": get_git_commit(repo),
        "git_dirty": is_git_dirty(repo),
        "source_sha256": digest.hexdigest(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "driver_inventory": driver,
        "geometry_units": "metres internally; depth/CD errors millimetres; disparity output-grid pixels",
        "rgb_protocol": "full image [0,1], left and right separate; no occlusion/tool exclusion",
        "paper_protocol_equivalence": False,
        "paper_protocol_unresolved": [
            "exact paper split/frame list/crop/masks/LPIPS backbone/aggregation",
            "CD/NC point sampling and distance definition",
        ],
        "timing_scope": "host deployment tensors to device + live teacher + A03 + MLP + both-view native render; excludes disk/GT/metrics",
        "timing_isolated": protocol.dedicated_gpu,
        "optimizer_updates": 0,
    }
    rows: list[dict[str, Any]] = []
    if resume:
        saved = json.loads((out / "identity.json").read_text())
        if saved != json.loads(json.dumps(identity)) or (out / "summary.json").exists():
            raise ValueError("resume needs identical identities and an incomplete benchmark")
        with (out / "samples.jsonl").open() as handle:
            rows = [json.loads(line) for line in handle]
        if [r["sample_id"] for r in rows] != list(ids[: len(rows)]):
            raise ValueError("resume records are not the selected manifest prefix")
    else:
        out.mkdir(parents=True, exist_ok=False)
        _write_new(out / "identity.json", identity)
    dtype = torch.float16 if config.runtime.precision == "fp16" else torch.bfloat16

    def forward(sample: Any) -> Any:
        batch = _deployment_to_device(sample.batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=config.runtime.precision in ("fp16", "bf16"),
        ):
            return model(
                batch,
                data_role="final_test",
                sample_ids=(sample.sample_id,),
                calibration_metadata=sample.calibration_metadata,
            )

    started = time.monotonic()
    profile_path = out / "profile.json"
    run_lock = (out / "run.lock").open("a+b")
    try:
        if sys.platform == "win32":
            import msvcrt

            run_lock.seek(0)
            msvcrt.locking(run_lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        run_lock.close()
        raise ValueError("another process owns this benchmark output directory") from error
    try:
        with torch.inference_mode():
            if not profile_path.exists():
                first = adapter.load(
                    ids[0], data_role="final_test", include_supervision=False, output_hw=resolution
                )

                def timed_forward() -> None:
                    forward(first)

                profile, _ = profile_callable(
                    timed_forward,
                    warmup=protocol.warmup,
                    repetitions=protocol.profile_repetitions,
                    device=device,
                    reset_peak_memory=(lambda: torch.cuda.reset_peak_memory_stats(device))
                    if device.type == "cuda"
                    else None,
                    peak_memory=(lambda: torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else None,
                )
                _write_new(
                    profile_path,
                    {
                        **profile.to_dict(),
                        "sample_id": ids[0],
                        "scope": identity["timing_scope"],
                        "isolated": protocol.dedicated_gpu,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                        if device.type == "cuda"
                        else None,
                    },
                )
                del first
            perceptual = perceptual.to(device)
            print(
                json.dumps(
                    {
                        "event": "benchmark_start",
                        "selected_samples": len(ids),
                        "completed_samples": len(rows),
                        "output_dir": str(out),
                    }
                ),
                flush=True,
            )
            with (out / "samples.jsonl").open("a" if resume else "x") as handle:
                for index in range(len(rows), len(ids)):
                    sample_id = ids[index]
                    # The inference sample has NO GT. Load supervision only AFTER prediction.
                    sample = adapter.load(
                        sample_id,
                        data_role="final_test",
                        include_supervision=False,
                        output_hw=resolution,
                    )
                    assert sample.supervision is None
                    result = forward(sample)
                    gt = adapter.load(
                        sample_id,
                        data_role="final_test",
                        include_supervision=True,
                        output_hw=resolution,
                    )
                    assert gt.supervision is not None
                    if sample.calibration_metadata != gt.calibration_metadata:
                        raise ValueError(
                            "GT/deployment calibration or image correspondence changed"
                        )
                    batch = _deployment_to_device(sample.batch, device)
                    metrics: dict[str, MetricResult] = {}
                    for view, render, target in (
                        ("left", result.render_left, batch.left),
                        ("right", result.render_right, batch.right),
                    ):
                        if render is None:
                            raise ValueError(f"benchmark requires {view} render")
                        for metric in (
                            psnr(render.image, target),
                            windowed_ssim(render.image, target),
                            perceptual.measure(render.image, target),
                        ):
                            metrics[f"{view}_{metric.name}"] = metric
                    geometry, details = evaluate_geometry(
                        result.target.disparity,
                        gt.supervision.disparity.to(device),
                        gt.supervision.valid_mask.to(device),
                        result.target.validity & result.target.support & batch.support,
                        batch.camera,
                        cloud_max_points=protocol.cloud_max_points,
                        cloud_chunk_size=protocol.cloud_chunk_size,
                        gaussian_centres=result.gaussians.means3d,
                        gaussian_valid=result.gaussians.valid_mask,
                        gaussian_source_index=result.prior.source_index,
                        rendered_depth=result.render_left.depth,
                    )
                    metrics.update(geometry)
                    dataset, keyframe, frame = sample_id.split("/")
                    directory = data_root / dataset / keyframe
                    directory = directory / "data" if (directory / "data").is_dir() else directory
                    gt_sha = file_sha256(directory / "disparity" / f"{frame}.tiff")
                    data_provenance = directory / "a00_provenance" / f"{frame}.json"
                    preprocessing_evidence = None
                    if data_provenance.is_file():
                        evidence = json.loads(data_provenance.read_text())
                        if evidence.get("sample_id") != sample_id:
                            raise ValueError("preprocessing provenance has a different sample ID")
                        expected_gt = evidence.get("output_files_sha256", {}).get(
                            f"disparity/{frame}.tiff"
                        )
                        if expected_gt is not None and expected_gt != gt_sha:
                            raise ValueError("GT disparity differs from preprocessing provenance")
                        preprocessing_evidence = {
                            "sha256": file_sha256(data_provenance),
                            "frame_correspondence": evidence.get("frame_correspondence"),
                            "calibration_evidence_sha256": evidence.get(
                                "calibration_evidence_sha256"
                            ),
                        }
                    row = {
                        "sample_id": sample_id,
                        "sample_index": index,
                        "sequence_id": dataset + "/" + keyframe,
                        "elapsed_session_seconds": time.monotonic() - started,
                        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else None,
                        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(device)
                        if device.type == "cuda"
                        else None,
                        "geometry": details,
                        "metrics": {k: replace(v, name=k).to_dict() for k, v in metrics.items()},
                        "gt_disparity_sha256": gt_sha,
                        "preprocessing_evidence": preprocessing_evidence,
                        "calibration": sample.calibration_metadata,
                        "model_provenance_hash": result.provenance_record.provenance_hash,
                        "zero_visible_left": bool(
                            result.render_left.visibility is not None
                            and not bool((result.render_left.visibility > 0).any())
                        ),
                    }
                    handle.write(json.dumps(_safe(row), allow_nan=False) + "\n")
                    handle.flush()
                    rows.append(row)
                    if index == 0 or (index + 1) % protocol.log_every == 0 or index + 1 == len(ids):
                        print(
                            json.dumps(
                                {
                                    "event": "benchmark_progress",
                                    "completed": index + 1,
                                    "total": len(ids),
                                    "sample_id": sample_id,
                                    "elapsed_session_seconds": row["elapsed_session_seconds"],
                                }
                            ),
                            flush=True,
                        )
                    del sample, gt, result, batch, metrics, geometry
        if any(not torch.equal(v.detach().cpu(), before[k]) for k, v in model.state_dict().items()):
            raise ValueError("benchmark mutated model state")
        if file_sha256(checkpoint) != identity["checkpoint_sha256"]:
            raise ValueError("checkpoint changed during benchmark")
        sequences = sorted({r["sequence_id"] for r in rows})
        per_sequence = {
            s: summarize_metrics([r for r in rows if r["sequence_id"] == s]) for s in sequences
        }
        macro = {
            name: statistics.fmean(
                float(v[name]["frame_mean"])
                for v in per_sequence.values()
                if v[name]["frame_mean"] is not None
            )
            if any(v[name]["frame_mean"] is not None for v in per_sequence.values())
            else None
            for name in per_sequence[sequences[0]]
        }
        summary = {
            "status": "PASS",
            "sample_count": len(rows),
            "dataset_counts": identity["dataset_counts"],
            "complete_final_test": len(ids)
            == sum(e.data_role == "final_test" for e in adapter.manifest.entries),
            "paper_protocol_equivalence": False,
            "overall": summarize_metrics(rows),
            "per_sequence": per_sequence,
            "sequence_macro_frame_mean": macro,
            "per_dataset": {
                d: summarize_metrics([r for r in rows if r["sample_id"].split("/")[0] == d])
                for d in identity["dataset_counts"]
            },
            "checkpoint_unchanged": True,
            "model_state_unchanged": True,
            "optimizer_updates": 0,
            "zero_visible_left_frames": sum(r["zero_visible_left"] for r in rows),
            "peak_torch_allocated_bytes": max(
                [torch.cuda.max_memory_allocated(device)]
                + [
                    r["peak_torch_allocated_bytes"]
                    for r in rows
                    if r["peak_torch_allocated_bytes"] is not None
                ]
            )
            if device.type == "cuda"
            else None,
            "peak_torch_reserved_bytes": max(
                [torch.cuda.max_memory_reserved(device)]
                + [
                    r["peak_torch_reserved_bytes"]
                    for r in rows
                    if r["peak_torch_reserved_bytes"] is not None
                ]
            )
            if device.type == "cuda"
            else None,
            "memory_scope": "entire harness including LPIPS and geometry metrics; model-only peak in profile.json",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_table(out / "metrics.csv", summary)
        _write_new(out / "summary.json", summary)
        return summary
    except BaseException as error:
        _write_new(
            out / f"error-{time.time_ns()}.json",
            {
                "type": type(error).__name__,
                "error": str(error),
                "completed_samples": len(rows),
                "next_sample_index": len(rows),
                "complete": False,
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        run_lock.close()
