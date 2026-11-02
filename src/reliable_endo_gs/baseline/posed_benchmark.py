"""Development-only operational harness for the native posed-Gaussian adapters.

Metric implementations remain owned by evaluation/training. This harness admits
only the exact benchmark val IDs and the fixed train control rule; no other cohort
or dataset arguments are exposed. One model/process, sequential frames, no timing
benchmark on shared hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reliable_endo_gs.baseline.posed_gaussians_native import build_posed_predictor
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, load_a01_manifest
from reliable_endo_gs.evaluation.a04_benchmark import _safe, summarize_metrics
from reliable_endo_gs.evaluation.benchmark_geometry import evaluate_geometry
from reliable_endo_gs.evaluation.paired_models import _state_hash
from reliable_endo_gs.evaluation.perceptual import LPIPSAlexNet
from reliable_endo_gs.evaluation.stereo import MetricResult
from reliable_endo_gs.runtime.identity import source_sha256
from reliable_endo_gs.teachers import load_a01_roles
from reliable_endo_gs.training.field_refinement import frame_metrics


def digest(path: Path) -> str:
    """SHA-256 a file without loading a checkpoint into memory."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def development_cohorts(cfg: dict) -> dict[str, list[str]]:
    """Read only allowed manifests; return val_7=56 and train_1236=4×14 IDs.

    Train control follows kdattr_common.train_ids exactly: remove recorded
    unusable samples, sort within each dataset, take midpoint quantiles.
    No forbidden worklist is opened, even for an overlap check.
    """
    baseline = Path(cfg["benchmark_config"])
    original = json.loads(baseline.read_text())
    for key in ("samples", "training_manifest", "validation_manifest"):
        pin = next(p for p in original["pinned_files"] if p["path"] == original[key])
        if digest(Path(original[key])) != pin["sha256"]:
            raise ValueError("frozen input identity changed: " + key)
    val = list(json.loads(Path(original["samples"]).read_text())["sample_ids"])
    entries = json.loads(Path(original["training_manifest"]).read_text())["entries"]
    excluded = {
        sid for sid, _ in json.loads((baseline.parent / "usable_ids.json").read_text())["excluded"]
    }
    grouped: dict[str, list[str]] = {}
    for entry in entries:
        sid = entry["sample_id"]
        dataset = sid.split("/")[0]
        if dataset not in {"dataset_1", "dataset_2", "dataset_3", "dataset_6"}:
            raise ValueError("training manifest includes unauthorized dataset")
        if sid not in excluded:
            grouped.setdefault(dataset, []).append(sid)
    control = []
    for dataset in sorted(grouped):
        ids = sorted(grouped[dataset])
        control.extend(ids[int((k + 0.5) * len(ids) / 14)] for k in range(14))
    if (
        len(val) != 56
        or len(set(val)) != 56
        or any(s.split("/")[0] != "dataset_7" for s in val)
        or len(control) != 56
        or len(set(control)) != 56
        or set(val) & set(control)
    ):
        raise ValueError("only the exact 56 val frames and 56 train controls are permitted")
    return {"val_7": val, "train_1236": control}


def main() -> None:
    """Execute quality and the existing seven-shift protocol; no model selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg: dict[str, Any] = json.loads(args.config.read_text())
    if cfg.get("near_m") is None or cfg.get("far_m") is None:
        raise ValueError("fixed near_m/far_m must be resolved explicitly before inference")
    cohorts = development_cohorts(cfg)
    if cohorts != cfg["cohorts"]:
        raise ValueError("cohort differs from frozen development selection")
    root = Path(cfg["artifact_root"]).resolve()
    if not args.output.resolve().is_relative_to(root) or args.output.exists():
        raise ValueError("output must be a new directory inside the D: artifact root")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["gpu_uuid"]:
        raise ValueError("pin CUDA_VISIBLE_DEVICES to the configured physical GPU UUID")
    # Do not rely on yesterday's device state; inspect free memory immediately.
    check = subprocess.run(
        [
            "timeout",
            "30",
            "nvidia-smi",
            "--id=" + cfg["gpu_uuid"],
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if int(check.stdout.strip()) < 4096:
        raise RuntimeError("less than 4 GiB free; no GPU work started")
    torch.cuda.set_per_process_memory_fraction(0.2, 0)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if str(props.uuid).removeprefix("GPU-") != cfg["gpu_uuid"].removeprefix("GPU-"):
        raise ValueError("physical GPU UUID mismatch")
    torch.set_num_threads(1)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    checkpoint = Path(cfg["checkpoint"])
    if digest(checkpoint) != cfg["checkpoint_sha256"]:
        raise ValueError("checkpoint identity changed")
    source = Path(cfg["source"])
    before = source_sha256(source)
    if before != cfg["source_sha256"]:
        raise ValueError("source snapshot identity changed")
    for path, expected in cfg.get("extra_asset_sha256", {}).items():
        if digest(Path(path)) != expected:
            raise ValueError("native dependency identity changed: " + path)
    import reliable_endo_gs

    if not Path(reliable_endo_gs.__file__).resolve().is_relative_to(source):
        raise ValueError("must import the prepared source snapshot")
    predictor, native = build_posed_predictor(
        cfg["kind"],
        Path(cfg["upstream"]),
        cfg["upstream_commit"],
        checkpoint,
        near_m=cfg["near_m"],
        far_m=cfg["far_m"],
    )
    predictor.to(device).eval().requires_grad_(False)
    state_before = _state_hash(predictor)
    original = json.loads(Path(cfg["benchmark_config"]).read_text())
    for key in ("lpips_backbone", "lpips_calibration"):
        pin = next(p for p in original["pinned_files"] if p["path"] == original[key])
        if digest(Path(original[key])) != pin["sha256"]:
            raise ValueError("LPIPS identity changed")
    perceptual = (
        LPIPSAlexNet(Path(original["lpips_backbone"]), Path(original["lpips_calibration"]))
        .to(device)
        .eval()
    )
    roles = load_a01_roles(source / "configs/teachers/a01_scared.yaml")
    shift_file = Path(cfg["shift_common"])
    if digest(shift_file) != cfg["shift_common_sha256"]:
        raise ValueError("shift protocol changed")
    spec = importlib.util.spec_from_file_location("frozen_kdattr_common", shift_file)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen shift helpers")
    shifts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shifts)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = {
        **cfg,
        "resolved_native": native,
        "config_sha256": digest(args.config),
        "hardware": {
            "gpu": props.name,
            "gpu_uuid": str(props.uuid),
            "total_memory_bytes": props.total_memory,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
            "load_average_start": os.getloadavg(),
            "memory_fraction": 0.2,
        },
        "latency": "not measured: shared GPU",
        "lpips_identity": perceptual.identity,
        "model_state_sha256": state_before,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    try:
        for split, ids in cohorts.items():
            manifest_key = "validation_manifest" if split == "val_7" else "training_manifest"
            manifest = load_a01_manifest(Path(original[manifest_key]), roles=roles)
            adapter = ScaredA00Adapter(Path(original["data_root"]), manifest, roles=roles)
            directory = args.output / split
            directory.mkdir()
            rows = []
            shift_rows = []
            with (directory / "samples.jsonl").open("x") as log, torch.inference_mode():
                for ordinal, sid in enumerate(ids):
                    sample = adapter.load(
                        sid,
                        data_role="final_train",
                        include_supervision=False,
                        output_hw=(512, 640),
                        device=device,
                    )
                    out = predictor(sample)
                    gt = adapter.load_supervision(sample)
                    gtd, gtv = gt.disparity.to(device), gt.valid_mask.to(device)
                    metrics, _ = evaluate_geometry(
                        out.disparity,
                        gtd,
                        gtv,
                        sample.batch.support & torch.isfinite(out.disparity),
                        sample.batch.camera,
                        cloud_max_points=1024,
                        cloud_chunk_size=256,
                        gaussian_centres=out.field.means3d,
                        gaussian_valid=out.field.valid_mask,
                        gaussian_source_index=out.source_index,
                        rendered_depth=None,
                    )
                    image_values = frame_metrics(out.left.image, out.right.image, sample.batch)
                    count = sample.batch.left.numel()
                    for name, value in image_values.items():
                        definition = (
                            "10*log10(data_range^2/MSE)"
                            if name.endswith("psnr")
                            else "RGB Gaussian11 sigma1.5 zero_padding5 C1=0.01^2 C2=0.03^2 full_image"
                        )
                        metrics[name] = MetricResult(name, value, count, count, definition)
                    for view, render, target in (
                        ("left", out.left, sample.batch.left),
                        ("right", out.right, sample.batch.right),
                    ):
                        metrics[view + "_lpips_alex"] = perceptual.measure(render.image, target)
                    row = {
                        "sample_id": sid,
                        "metrics": {k: v.to_dict() for k, v in metrics.items()},
                        "quality_pass_forward_ms": None,
                    }
                    log.write(json.dumps(_safe(row), allow_nan=False) + "\n")
                    log.flush()
                    rows.append(row)
                    with (directory / f"prediction-{ordinal:03d}.pt").open("xb") as handle:
                        torch.save(
                            {
                                "sample_id": sid,
                                "disparity": out.disparity.cpu(),
                                "render_left": out.left.image.cpu(),
                                "render_right": out.right.image.cpu(),
                            },
                            handle,
                        )
                    for shift in (-24, -16, -8, 0, 8, 16, 24):
                        batch, target, valid = shifts.shifted_batch(sample, gtd, gtv, shift)
                        stats = shifts.shift_stats(
                            predictor.predict_disparity(batch), target, valid, batch.support
                        )
                        shift_rows.append(
                            {
                                "model": cfg["kind"],
                                "split": split,
                                "sample_id": sid,
                                "shift": shift,
                                **(stats or {"n": 0}),
                            }
                        )
                    print(json.dumps({"split": split, "done": ordinal + 1}), flush=True)
            (directory / "summary.json").write_text(
                json.dumps(_safe(summarize_metrics(rows)), indent=2)
            )
            (directory / "shift.json").write_text(json.dumps(shift_rows, indent=2))
        if digest(checkpoint) != cfg["checkpoint_sha256"] or source_sha256(source) != before:
            raise ValueError("checkpoint or source changed during evaluation")
        if _state_hash(predictor) != state_before:
            raise ValueError("evaluation mutated model parameters or buffers")
        for path, expected in cfg.get("extra_asset_sha256", {}).items():
            if digest(Path(path)) != expected:
                raise ValueError("native dependency changed during evaluation: " + path)
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"type": type(error).__name__, "error": str(error)})
        )
        raise
    finally:
        (args.output / "resources.json").write_text(
            json.dumps(
                {
                    "wall_seconds": time.perf_counter() - started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "load_average_end": os.getloadavg(),
                    "latency": "not measured: shared GPU",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
