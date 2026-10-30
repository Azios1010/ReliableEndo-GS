"""Bounded, provenance-bound A00 Endo-E2E-GS adaptation and student comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

from reliable_endo_gs.baseline.a00_model import build_a00_endo_model, render_a00_endo
from reliable_endo_gs.baseline.evaluation.a00_small import evaluate_a00_small
from reliable_endo_gs.baseline.training.a00_optimization import finite_update
from reliable_endo_gs.baseline.upstream import import_upstream_module, require_pinned_upstream
from reliable_endo_gs.data.scared_a00 import (
    ScaredA00Adapter,
    ScaredA00Sample,
    ScaredSupervision,
    load_a01_manifest,
)
from reliable_endo_gs.rendering.production import ProductionFieldRenderer
from reliable_endo_gs.runtime.identity import source_sha256
from reliable_endo_gs.teachers import load_a01_roles
from reliable_endo_gs.utils.seed import set_seed


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def validate_spec(config: dict[str, Any], repo: Path, manifest_root: Path) -> None:
    """Fail closed on scope, exact manifests and the explicitly adapted variant."""
    if config["schema"] != "endo_student_small.v1":
        raise ValueError("unsupported small-comparison schema")
    execution = config["execution"]
    if not execution["training_authorized"] or not execution["inference_authorized"]:
        raise ValueError("explicit training/inference authorization is absent")
    if any(
        execution[k]
        for k in ("phase_ii", "repairs", "selector", "automatic_full_data_continuation")
    ):
        raise ValueError("small baseline comparison forbids later stages")
    arm = config["arms"]["endo_e2e_gs"]
    if arm.get("variant") != "a00_metric_geometry":
        raise ValueError("runner only implements the documented A00 metric adaptation")
    if arm["stage1"]["updates"] + arm["stage2"]["updates"] != 112:
        raise ValueError("bounded total must remain 112 updates")
    if config["experiment"]["batch_size"] != 1 or config["data"]["resolution_hw"] != [512, 640]:
        raise ValueError("unexpected batch/grid")
    roles = load_a01_roles(repo / config["data"]["roles"])
    groups = {}
    for label, size in (("fit", 7), ("validation", 14)):
        spec = config["data"][label]
        path = manifest_root / spec["manifest"]
        if _sha(path) != spec["file_sha256"]:
            raise ValueError(f"{label} manifest bytes changed")
        manifest = load_a01_manifest(path, roles=roles)
        ids = [e.sample_id for e in manifest.entries]
        if len(ids) != size or len(set(ids)) != size:
            raise ValueError("unexpected small split size/duplicates")
        if {s.split("/")[0] for s in ids} != {f"dataset_{i}" for i in range(1, 8)}:
            raise ValueError("only original TRAIN1-7 may enter this pilot")
        if any(e.data_role != "final_train" for e in manifest.entries):
            raise ValueError("wrong manifest role")
        groups[label] = {s.rsplit("/", 1)[0] for s in ids}
    if not groups["fit"].isdisjoint(groups["validation"]):
        raise ValueError("FIT/validation keyframes overlap")
    for relative, expected in (
        (config["data"]["roles"], config["data"]["roles_file_sha256"]),
        (arm["stage1"]["upstream_config"], arm["stage1"]["upstream_config_sha256"]),
        (arm["stage2"]["upstream_config"], arm["stage2"]["upstream_config_sha256"]),
    ):
        if _sha(repo / relative) != expected:
            raise ValueError(f"pinned input changed: {relative}")
    state = require_pinned_upstream(repo / arm["upstream_root"])
    if state.commit != arm["upstream_commit"]:
        raise ValueError("wrong upstream revision")


def run_small(
    *, repo: Path, config_path: Path, output: Path, stage1_checkpoint: Path | None = None
) -> None:
    """Run only baseline 56+56 updates and compare with immutable student evidence."""
    started = time.monotonic()
    config = yaml.safe_load(config_path.read_text())
    manifest_root = Path(os.environ[config["data"]["manifest_root_env"]])
    validate_spec(config, repo, manifest_root)
    output.mkdir(exist_ok=False)

    def emit(event: str, **values: Any) -> None:
        record = {"event": event, "elapsed_seconds": time.monotonic() - started, **values}
        (output / "status.json").write_text(json.dumps(record, indent=2) + "\n")
        with (output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != config["runtime"]["cuda_visible_devices"]:
            raise ValueError("GPU0 UUID mask required")
        if torch.cuda.device_count() != 1 or torch.get_num_threads() != 1:
            raise ValueError("one visible GPU and OMP_NUM_THREADS=1 required")
        device = torch.device("cuda:0")
        torch.cuda.init()
        torch.cuda.set_device(device)
        set_seed(config["experiment"]["seed"])
        roles = load_a01_roles(repo / config["data"]["roles"])
        data_root = Path(os.environ[config["data"]["root_env"]])
        samples, truths, adapters, manifests = {}, {}, {}, {}
        for label in ("fit", "validation"):
            manifest = load_a01_manifest(
                manifest_root / config["data"][label]["manifest"], roles=roles
            )
            manifests[label] = manifest
            adapters[label] = ScaredA00Adapter(data_root, manifest, roles=roles)
            samples[label] = []
            for entry in manifest.entries:
                sample = adapters[label].load(
                    entry.sample_id,
                    data_role="final_train",
                    include_supervision=False,
                    output_hw=(512, 640),
                    device=device,
                )
                assert sample.supervision is None
                samples[label].append(sample)
                # GT remains outside the deployment batch/model interface.
                truths[sample.sample_id] = adapters[label].load_supervision(sample)
        upstream = repo / config["arms"]["endo_e2e_gs"]["upstream_root"]
        model = build_a00_endo_model(upstream).to(device)
        renderer = ProductionFieldRenderer()
        losses = import_upstream_module("lib.loss", upstream)
        import corr_sampler
        import diff_gaussian_rasterization

        identity = {
            "variant": "endo_e2e_gs_a00_metric_geometry",
            "adaptations": config["arms"]["endo_e2e_gs"]["protocol_bridge"],
            "config": config,
            "config_file_sha256": _sha(config_path),
            "source_sha256": source_sha256(repo),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "upstream_commit": require_pinned_upstream(upstream).commit,
            "seed": config["experiment"]["seed"],
            "manifest_hashes": {k: v.manifest_hash for k, v in manifests.items()},
            "renderer_identity": renderer.identity,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unknown")),
            "corr_sampler_file": corr_sampler.__file__,
            "corr_sampler_sha256": _sha(Path(corr_sampler.__file__)),
            "rasterizer_file": diff_gaussian_rasterization.__file__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "student_reference": config["comparison"]["student_reference"],
            "teacher_or_repairs_or_selector_loaded": False,
        }
        _write(output / "run.json", identity)
        emit(
            "inputs_ready",
            fit=7,
            validation=14,
            parameters=identity["parameters"],
            gpu=identity["gpu"],
        )
        histories = []
        total_updates = 0
        stage_costs = {}
        if stage1_checkpoint is not None:
            recovery = config["recovery"]
            if _sha(stage1_checkpoint) != recovery["stage1_checkpoint_sha256"]:
                raise ValueError("stage-one recovery checkpoint hash mismatch")
            previous = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
            if (
                previous["stage"] != "stage1"
                or previous["stage_updates"] != 56
                or previous["total_updates"] != 56
            ):
                raise ValueError("recovery requires exactly the completed stage-one artifact")
            for key in (
                "variant",
                "upstream_commit",
                "seed",
                "manifest_hashes",
                "renderer_identity",
            ):
                if previous["identity"][key] != identity[key]:
                    raise ValueError(f"stage-one recovery identity mismatch: {key}")
            for relative, expected in recovery["unchanged_scientific_files_sha256"].items():
                if _sha(repo / relative) != expected:
                    raise ValueError(f"scientific source changed during recovery: {relative}")
            model.native.load_state_dict(previous["network"], strict=True)
            torch.set_rng_state(previous["rng_cpu"])
            torch.cuda.set_rng_state(previous["rng_cuda"], device)
            total_updates = 56
            _write(
                output / "recovery.json",
                {
                    "checkpoint": str(stage1_checkpoint),
                    "sha256": _sha(stage1_checkpoint),
                    "old_source_sha256": previous["identity"]["source_sha256"],
                    "new_source_sha256": identity["source_sha256"],
                    "reason": recovery["reason"],
                },
            )
            old_events = [
                json.loads(line)
                for line in (stage1_checkpoint.parent / "events.jsonl").read_text().splitlines()
            ]
            histories.extend(
                {k: v for k, v in row.items() if k not in ("event", "elapsed_seconds")}
                for row in old_events
                if row["event"] == "validation_complete" and row["stage"] == "stage1"
            )
            old_steps = [
                json.loads(line)
                for line in (stage1_checkpoint.parent / "train.jsonl").read_text().splitlines()
            ]
            old_times = [row["step_seconds"] for row in old_steps if row["stage"] == "stage1"]
            stage_costs["stage1"] = {
                "reused_completed_stage1": True,
                "source_attempt": str(stage1_checkpoint.parent),
                "model_update_seconds": sum(old_times),
                "median_step_seconds_after_first_pass": statistics.median(old_times[7:]),
            }
            emit("stage1_recovered", total_updates=56, checkpoint=str(stage1_checkpoint))
        torch.cuda.reset_peak_memory_stats(device)

        def assess(stage: str, stage_updates: int) -> None:
            entry = evaluate_a00_small(
                model,
                samples["validation"],
                truths,
                renderer,
                output,
                stage=stage,
                stage_updates=stage_updates,
                total_updates=total_updates,
                device=device,
            )
            histories.append(entry)
            emit("validation_complete", **entry)

        def save(stage: str, steps: int, optimizer: Any, scheduler: Any, scaler: Any) -> Path:
            path = output / f"{stage}-{steps:03d}.pt"
            torch.save(
                {
                    "network": model.native.state_dict(),
                    "stage": stage,
                    "total_updates": total_updates,
                    "stage_updates": steps,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state(device),
                    "identity": identity,
                },
                path,
            )
            return path

        final_checkpoint = None
        for stage in ("stage1", "stage2"):
            if stage == "stage1" and stage1_checkpoint is not None:
                continue
            native_cfg = model.native.cfg
            native_cfg.defrost()
            native_cfg.raft.mixed_precision = stage == "stage2"
            native_cfg.freeze()
            # Retain untouched network/head; stage-one GS modules receive no gradients.
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.0002, weight_decay=1e-5, eps=1e-8
            )
            stage_spec = config["arms"]["endo_e2e_gs"][stage]
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=0.0002,
                total_steps=stage_spec["scheduler"]["total_steps"],
                pct_start=0.01,
                anneal_strategy="linear",
                cycle_momentum=False,
            )
            scaler = torch.amp.GradScaler("cuda", enabled=stage == "stage2")
            model.train()
            model.native.raft_stereo.freeze_bn()
            assess(stage, 0)
            timings = []
            emit(
                "stage_started",
                stage=stage,
                total_updates=total_updates,
                planned_updates=stage_spec["updates"],
            )
            stage_started = time.monotonic()
            for step in range(1, stage_spec["updates"] + 1):
                sample = samples["fit"][(step - 1) % 7]
                gt = truths[sample.sample_id]

                def training_loss(
                    sample: ScaredA00Sample = sample, gt: ScaredSupervision = gt, stage: str = stage
                ) -> tuple[torch.Tensor, dict[str, float | None]]:
                    prediction = model(
                        sample.batch,
                        gt_disparity=gt.disparity,
                        gt_valid=gt.valid_mask,
                        with_gaussians=stage == "stage2",
                        mixed_precision=stage == "stage2",
                    )
                    loss = prediction.disparity_loss
                    assert loss is not None
                    rgb_l1 = rgb_ssim = None
                    if stage == "stage2":
                        render = render_a00_endo(prediction, sample.batch, renderer, view="left")
                        rgb_l1 = losses.l1_loss(render.image, sample.batch.left)
                        rgb_ssim = 1.0 - losses.ssim(render.image, sample.batch.left)
                        loss = loss + 0.8 * rgb_l1 + 0.2 * rgb_ssim
                    return loss, {
                        "disparity_loss": float(prediction.disparity_loss.detach()),
                        "rgb_l1": float(rgb_l1.detach()) if rgb_l1 is not None else None,
                        "rgb_one_minus_ssim": float(rgb_ssim.detach())
                        if rgb_ssim is not None
                        else None,
                    }

                update = finite_update(
                    training_loss,
                    model,
                    optimizer,
                    scaler,
                    device=device,
                    max_overflow_retries=stage_spec.get("max_amp_overflow_retries", 0),
                )
                scheduler.step()
                timings.append(update["step_seconds"])
                total_updates += 1
                row = {
                    "stage": stage,
                    "stage_updates": step,
                    "total_updates": total_updates,
                    "sample_id": sample.sample_id,
                    **update,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                with (output / "train.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                if step == 1 or step % 7 == 0:
                    emit("train_progress", **row)
                if step in (28, 56):
                    final_checkpoint = save(stage, step, optimizer, scheduler, scaler)
                    assess(stage, step)
            stage_costs[stage] = {
                "wall_seconds_including_eval": time.monotonic() - stage_started,
                "model_update_seconds": sum(timings),
                "median_step_seconds_after_first_pass": statistics.median(timings[7:]),
            }
        assert total_updates == 112 and final_checkpoint is not None
        peak = {
            "allocated_bytes": torch.cuda.max_memory_allocated(device),
            "reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        final_metrics = histories[-1]["metrics"]
        reference_path = manifest_root / config["comparison"]["student_reference"]["result"]
        if _sha(reference_path) != config["comparison"]["student_reference"]["file_sha256"]:
            raise ValueError("student reference changed")
        student = json.loads(reference_path.read_text())
        if student["updates"]["model"] != 112:
            raise ValueError("wrong student update budget")
        student_final = next(
            h for h in student["history"] if h["partition"] == "validation" and h["updates"] == 112
        )
        names = sorted(set(final_metrics) | set(student_final["metrics"]))
        lines = [
            "# Small A00 output comparison",
            "",
            "7 FIT / 14 keyframe-disjoint development images; no TEST8-9.",
            "",
            "Endo-E2E-GS A00 metric adaptation: fresh 56 stereo + 56 joint updates. Student: existing 112-update frozen-teacher KD pilot.",
            "Architectures, supervision, precision and compute differ. This is not native parity, paper reproduction or a convergence result.",
            "",
            "| Metric | Endo A00 adaptation | Multiscale KD student |",
            "|---|---:|---:|",
        ]
        comparison = []
        for name in names:
            a = final_metrics.get(name, {}).get("mean")
            b = student_final["metrics"].get(name, {}).get("mean")
            comparison.append({"metric": name, "endo_a00": a, "student_multiscale": b})
            values = ["N/A" if v is None else f"{v:.6g}" for v in (a, b)]
            lines.append(f"| {name} | {values[0]} | {values[1]} |")
        (output / "comparison.md").write_text("\n".join(lines) + "\n")
        _write(
            output / "comparison.json",
            {
                "rows": comparison,
                "student_reference_sha256": _sha(reference_path),
                "baseline_identity": identity,
                "history": histories,
            },
        )
        _write(
            output / "result.json",
            {
                "status": "complete",
                "updates": total_updates,
                "histories": histories,
                "stage_costs": stage_costs,
                "peak_vram": peak,
                "checkpoint": str(final_checkpoint),
                "checkpoint_sha256": _sha(final_checkpoint),
                "wall_seconds": time.monotonic() - started,
                "student_retrained": False,
                "phase_ii_started": False,
                "final_test_used": False,
            },
        )
        emit("complete", updates=112, comparison=str(output / "comparison.md"), peak_vram=peak)
    except BaseException as error:
        _write(
            output / f"failure-{time.time_ns()}.json",
            {"error": str(error), "traceback": traceback.format_exc()},
        )
        emit("failed", error=str(error))
        raise


def main() -> None:
    """Run the explicitly authorized bounded experiment, without auto-resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    args = parser.parse_args()
    run_small(
        repo=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        output=args.output_dir.resolve(),
        stage1_checkpoint=args.stage1_checkpoint.resolve() if args.stage1_checkpoint else None,
    )


if __name__ == "__main__":
    main()
