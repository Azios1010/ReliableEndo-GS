"""Shared read-only development benchmark for two frozen reconstruction models."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.contracts import GaussianField, RenderOutput
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter
from reliable_endo_gs.evaluation.a04_benchmark import _safe, summarize_metrics
from reliable_endo_gs.evaluation.benchmark_geometry import evaluate_geometry
from reliable_endo_gs.evaluation.perceptual import LPIPSAlexNet
from reliable_endo_gs.evaluation.rendering import psnr, windowed_ssim
from reliable_endo_gs.profiling.runtime import profile_callable
from reliable_endo_gs.training.harness import capture_rng, restore_rng
from reliable_endo_gs.utils.io import write_json_atomic


@dataclass(frozen=True)
class FrozenModelPrediction:
    """Disparity [1,1,H,W], field [1,N,*], source pixels [1,N], two RGB renders."""

    disparity: torch.Tensor
    field: GaussianField
    source_index: torch.Tensor
    left: RenderOutput
    right: RenderOutput


def validate_development_worklist(
    ids: tuple[str, ...], training_ids: set[str], development_ids: set[str]
) -> None:
    """The first comparison uses only fixed, disjoint dataset_7 IDs; no final_test."""
    if (
        not ids
        or len(set(ids)) != len(ids)
        or set(ids) & training_ids
        or not set(ids) <= development_ids
        or any(s.split("/")[0] != "dataset_7" for s in ids)
    ):
        raise ValueError("matched development IDs must be unique disjoint val_7 members")


def _state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def benchmark_frozen_predictor(
    *,
    predictor: torch.nn.Module,
    adapter: ScaredA00Adapter,
    sample_ids: tuple[str, ...],
    training_ids: set[str],
    output_dir: Path,
    output_hw: tuple[int, int],
    device: torch.device,
    provenance: dict[str, Any],
    lpips_backbone: Path | None = None,
    lpips_calibration: Path | None = None,
    warmup: int = 10,
    repetitions: int = 200,
) -> dict[str, Any]:
    """One model per process; shared metrics, GT after prediction, frozen weights/RNG.

    Timings include model, geometry, Gaussian construction and BOTH RGB renders.
    The fixed input batches are preloaded to GPU; decode/H2D/I/O/metrics are excluded
    explicitly. Repetitions cycle across the fixed worklist, not 200 unique frames.
    Peak PyTorch allocation includes the input cache and the model, without LPIPS.
    """
    known = {e.sample_id for e in adapter.manifest.entries if e.data_role == "final_train"}
    validate_development_worklist(sample_ids, training_ids, known)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite benchmark output")
    if (lpips_backbone is None) != (lpips_calibration is None):
        raise ValueError("both pinned LPIPS files are required")
    output_dir.mkdir(parents=True, exist_ok=False)
    before = _state_hash(predictor)
    rng = capture_rng(device)
    flags = [(m, m.training) for m in predictor.modules()]
    rows = []
    perceptual_identity: dict[str, Any] = {"status": "unavailable"}
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    try:
        predictor.eval()
        perceptual = None
        if lpips_backbone is not None and lpips_calibration is not None:
            perceptual = LPIPSAlexNet(lpips_backbone, lpips_calibration).to(device).eval()
            perceptual_identity = {"status": "measured", **perceptual.identity}
        with torch.no_grad(), (output_dir / "samples.jsonl").open("x") as log:
            for ordinal, sid in enumerate(sample_ids):
                sample = adapter.load(
                    sid,
                    data_role="final_train",
                    include_supervision=False,
                    output_hw=output_hw,
                    device=device,
                )
                if sample.supervision is not None:
                    raise ValueError("benchmark model input contains GT")
                sync()
                started = time.perf_counter()
                out: FrozenModelPrediction = predictor(sample)
                sync()
                forward_ms = (time.perf_counter() - started) * 1000
                # One identical separate GT loading/resampling path for both actors.
                gt = adapter.load_supervision(sample)
                metrics, _ = evaluate_geometry(
                    out.disparity,
                    gt.disparity.to(device),
                    gt.valid_mask.to(device),
                    sample.batch.support & torch.isfinite(out.disparity),
                    sample.batch.camera,
                    cloud_max_points=1024,
                    cloud_chunk_size=256,
                    gaussian_centres=out.field.means3d,
                    gaussian_valid=out.field.valid_mask,
                    gaussian_source_index=out.source_index,
                    rendered_depth=out.left.depth,
                )
                for view, render, target in (
                    ("left", out.left, sample.batch.left),
                    ("right", out.right, sample.batch.right),
                ):
                    metrics[view + "_psnr"] = psnr(render, target)
                    metrics[view + "_ssim"] = windowed_ssim(render.image, target)
                    if perceptual is not None:
                        metrics[view + "_lpips_alex"] = perceptual.measure(render.image, target)
                row = {
                    "sample_id": sid,
                    "metrics": {k: v.to_dict() for k, v in metrics.items()},
                    "quality_pass_forward_ms": forward_ms,
                }
                rows.append(row)
                log.write(json.dumps(_safe(row), allow_nan=False) + "\n")
                log.flush()
                if ordinal == 0 or (ordinal + 1) % 8 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "paired_model_quality_progress",
                                "samples": ordinal + 1,
                                "total": len(sample_ids),
                            }
                        ),
                        flush=True,
                    )
        del perceptual
        if device.type == "cuda":
            torch.cuda.empty_cache()
        inputs = [
            adapter.load(
                sid,
                data_role="final_train",
                include_supervision=False,
                output_hw=output_hw,
                device=device,
            )
            for sid in sample_ids
        ]
        position = 0

        def forward() -> FrozenModelPrediction:
            nonlocal position
            sample = inputs[position % len(inputs)]
            position += 1
            return predictor(sample)

        with torch.no_grad():
            profile, last = profile_callable(
                forward,
                warmup=warmup,
                repetitions=repetitions,
                batch_size=1,
                device=device,
                reset_peak_memory=(lambda: torch.cuda.reset_peak_memory_stats(device))
                if device.type == "cuda"
                else None,
                peak_memory=(lambda: torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None,
            )
        inputs.clear()
        del last, out, sample
    finally:
        for module, flag in flags:
            module.training = flag
        restore_rng(rng, device)
    if before != _state_hash(predictor):
        raise ValueError("read-only benchmark mutated model weights/buffers")
    by_keyframe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_keyframe[row["sample_id"].rsplit("/", 1)[0]].append(row)
    result = {
        **provenance,
        "status": "development_benchmark_complete",
        "sample_ids": list(sample_ids),
        "manifest_hash": adapter.manifest.manifest_hash,
        "samples_sha256": hashlib.sha256((output_dir / "samples.jsonl").read_bytes()).hexdigest(),
        "overall": summarize_metrics(rows),
        "per_keyframe": {k: summarize_metrics(v) for k, v in by_keyframe.items()},
        "timing": {
            **profile.to_dict(),
            "scope": "model_geometry_Gaussians_both_RGB_renders",
            "input_protocol": "preloaded_fixed_worklist_batches; decode_H2D_IO_metrics_excluded",
            "unique_frames": len(sample_ids),
            "parameter_count": sum(p.numel() for p in predictor.parameters()),
        },
        "perceptual": perceptual_identity,
        "weights_unchanged": True,
        "rng_restored": True,
        "model_state_sha256": before,
        "final_test_executed": False,
    }
    write_json_atomic(output_dir / "result.json", _safe(result))
    return result


def compare_frozen_benchmarks(
    baseline_dir: Path, kd_dir: Path, output_dir: Path, *, repetitions: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Paired fixed-frame deltas and whole-keyframe bootstrap, never checkpoint selection."""
    if output_dir.exists():
        raise FileExistsError("comparison requires a new directory")
    summaries, frames = [], []
    for directory in (baseline_dir, kd_dir):
        summary = json.loads((directory / "result.json").read_text())
        content = (directory / "samples.jsonl").read_bytes()
        if (
            summary.get("weights_unchanged") is not True
            or summary.get("samples_sha256") != hashlib.sha256(content).hexdigest()
        ):
            raise ValueError("benchmark evidence changed or model was mutable")
        records = [json.loads(line) for line in content.splitlines()]
        if [row["sample_id"] for row in records] != summary["sample_ids"]:
            raise ValueError("benchmark frame order differs from its sealed worklist")
        summaries.append(summary)
        frames.append(records)
    a, b = summaries
    if (
        a["sample_ids"] != b["sample_ids"]
        or a["manifest_hash"] != b["manifest_hash"]
        or a["timing"]["scope"] != b["timing"]["scope"]
        or a["timing"]["input_protocol"] != b["timing"]["input_protocol"]
    ):
        raise ValueError("benchmark models must share cohort/metrics/timing scope")
    table = {}
    for metric in sorted(set(a["overall"]) & set(b["overall"])):
        groups: dict[str, list[float]] = defaultdict(list)
        for left, right in zip(*frames, strict=True):
            x, y = (
                left["metrics"].get(metric, {}).get("value"),
                right["metrics"].get(metric, {}).get("value"),
            )
            if (
                x is not None
                and y is not None
                and math.isfinite(float(x))
                and math.isfinite(float(y))
            ):
                groups[left["sample_id"].rsplit("/", 1)[0]].append(float(y) - float(x))
        keys = sorted(groups)
        delta = [v for values in groups.values() for v in values]
        rng = random.Random(seed)
        boot = (
            [
                statistics.mean(v for k in rng.choices(keys, k=len(keys)) for v in groups[k])
                for _ in range(repetitions)
            ]
            if len(keys) > 1
            else []
        )
        boot.sort()
        ci = (
            [boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))]]
            if boot
            else None
        )
        table[metric] = {
            "baseline": a["overall"][metric],
            "kd": b["overall"][metric],
            "KD_minus_baseline": statistics.mean(delta) if delta else None,
            "paired_known_frames": len(delta),
            "ci95_cluster_keyframe": ci,
            "keyframes": len(keys),
            "KD_greater_fraction": sum(v > 0 for v in delta) / len(delta) if delta else None,
        }
    result = {
        "status": "paired_final_models_development_comparison_complete",
        "metrics": table,
        "baseline_checkpoint": a.get("checkpoint"),
        "baseline_checkpoint_sha256": a.get("checkpoint_sha256"),
        "kd_checkpoint": b.get("checkpoint"),
        "kd_checkpoint_sha256": b.get("checkpoint_sha256"),
        "sample_ids": a["sample_ids"],
        "baseline_timing": a["timing"],
        "kd_timing": b["timing"],
        "bootstrap_unit": "whole_keyframe_preserving_frame_mean",
        "bootstrap_repetitions": repetitions,
        "scope": "fixed_val7_development_only; one_seed_per_model; no_checkpoint_selection",
        "final_test_executed": False,
        "superiority_claim": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "comparison.json", result)
    lines = [
        "# Final baseline vs final KD: fixed val_7 development comparison",
        "",
        "| Metric | Baseline | KD | KD − baseline | Paired frames |",
        "|---|---:|---:|---:|---:|",
    ]

    def fmt(x):
        return "unavailable" if x is None else f"{float(x):.6g}"

    for name, row in table.items():
        lines.append(
            f"| {name} | {fmt(row['baseline']['frame_mean'])} | {fmt(row['kd']['frame_mean'])} | {fmt(row['KD_minus_baseline'])} | {row['paired_known_frames']} |"
        )
    lines += ["", "| Inference (both views) | Baseline | KD |", "|---|---:|---:|"]
    for name in ("median_ms", "p90_ms", "fps", "peak_memory_bytes"):
        lines.append(f"| {name} | {fmt(a['timing'].get(name))} | {fmt(b['timing'].get(name))} |")
    lines += [
        "",
        "Timings include model, geometry, Gaussian construction and both renders with batch size 1; inputs are preloaded. I/O, decoding, H2D and metrics are excluded.",
        "This is a preliminary fixed-val_7 comparison, not final_test or a superiority/non-inferiority claim. Both models have one training seed. Baseline uses its disclosed exact-zero FP32 scale recovery.",
    ]
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    return result
