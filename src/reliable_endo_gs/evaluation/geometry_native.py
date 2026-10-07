"""Frozen native comparison of correspondence, learned stereo and Gaussian actions."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.actions.geometry_repair import GeometryRepair, GeometryRepairConfig
from reliable_endo_gs.actions.repair import Action, ActionResult, RepairState, eligible
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, load_a01_manifest
from reliable_endo_gs.evaluation.benchmark_geometry import evaluate_geometry
from reliable_endo_gs.evaluation.geometry_candidates import _metrics as depth_metrics
from reliable_endo_gs.evaluation.intervention_selection import _weights
from reliable_endo_gs.evaluation.rendering import psnr, windowed_ssim
from reliable_endo_gs.evaluation.tradeoff_outcomes import depth_snapshot, metric_changes
from reliable_endo_gs.geometry.rectified import reconstruct_rectified
from reliable_endo_gs.reconstruction import ReferenceConfig
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.regions.grid import Region, grid_regions
from reliable_endo_gs.rendering.production import ProductionFieldRenderer
from reliable_endo_gs.routing.config import RoutingConfig
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES, balanced_features
from reliable_endo_gs.routing.tradeoff import (
    ACTION_NAMES,
    COMPONENT_NAMES,
    load_tradeoff_config,
)
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.runtime.identity import source_sha256
from reliable_endo_gs.teachers import load_a01_roles
from reliable_endo_gs.training.a04_reference import _move_batch
from reliable_endo_gs.training.quality_components import component_targets, measure_quality
from reliable_endo_gs.training.student_checkpoint import (
    import_frozen_student_actions,
    read_student_checkpoint,
)
from reliable_endo_gs.training.student_config import StudentLearningConfig
from reliable_endo_gs.training.student_learning import StudentLearner, evaluation_mode
from reliable_endo_gs.utils.io import write_json_atomic


def _geometry_metrics(
    outcome: ActionResult,
    state: RepairState,
    gt_disparity: torch.Tensor,
    gt_mask: torch.Tensor,
) -> dict[str, Any]:
    values, _ = evaluate_geometry(
        outcome.disparity,
        gt_disparity,
        gt_mask,
        state.support,
        state.batch.camera,
        cloud_max_points=256,
        cloud_chunk_size=128,
        gaussian_centres=outcome.field.means3d,
        gaussian_valid=outcome.field.valid_mask,
        gaussian_source_index=state.base.prior.source_index,
        rendered_depth=outcome.left.depth,
    )
    return {key: value.to_dict() for key, value in values.items()}


def evaluate_native_geometry_candidates(
    *,
    checkpoint: Path,
    probe_report: Path,
    data_root: Path,
    manifest_path: Path,
    training_manifest: Path,
    roles_path: Path,
    output_dir: Path,
    tradeoff_config_path: Path | None = None,
    collection_role: str = "development",
    per_dataset: int = 4,
    timing_repeats: int = 3,
) -> dict[str, object]:
    """New pinned frozen experiment; no training, teacher, cache or router calls.

    Reuses the exact CPU-probe sample identities and candidate configuration.
    Timings include base and action kernels, exclude I/O, GT and oracle overhead;
    these are single-pass diagnostics, not a latency benchmark.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError(
            "native geometry comparison needs one visible GPU; Codex CUDA is unavailable"
        )
    repo = Path(__file__).resolve().parents[3]
    if output_dir.resolve() == repo or repo in output_dir.resolve().parents:
        raise ValueError("native geometry outputs must be outside repository")
    if output_dir.exists():
        raise FileExistsError("native geometry output exists")
    probe_bytes = probe_report.read_bytes()
    probe = json.loads(probe_bytes)
    ids = probe["sample_ids"]
    if (
        probe.get("status") != "cpu_geometry_candidate_screen_complete"
        or not isinstance(ids, list)
        or not 1 <= len(ids) <= 28
        or any(not isinstance(sid, str) for sid in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("requires a complete bounded CPU probe report")
    roles = load_a01_roles(roles_path)
    train = load_a01_manifest(training_manifest, roles=roles)
    development = load_a01_manifest(manifest_path, roles=roles)
    tradeoff = (
        load_tradeoff_config(tradeoff_config_path) if tradeoff_config_path is not None else None
    )
    if collection_role not in ("train", "development"):
        raise ValueError("outcome collection role must be train/development")
    if (
        type(per_dataset) is not int
        or not 1 <= per_dataset <= 16
        or type(timing_repeats) is not int
        or not 1 <= timing_repeats <= 5
    ):
        raise ValueError("bounded collection requires 1..16 samples/fold and 1..5 timings")
    if tradeoff is not None:
        pool_manifest = train if collection_role == "train" else development
        ids = []
        for fold in range(1, 8):
            pool = sorted(
                (
                    e.sample_id
                    for e in pool_manifest.entries
                    if e.data_role == "final_train" and e.sample_id.startswith(f"dataset_{fold}/")
                ),
                key=lambda sid: hashlib.sha256(f"{tradeoff.seed}:{sid}".encode()).hexdigest(),
            )
            # Round-robin across keyframes before taking additional frames.
            groups: dict[str, list[str]] = {}
            for sid in pool:
                groups.setdefault(sid.rsplit("/", 1)[0], []).append(sid)
            balanced = [
                group[i]
                for i in range(max((len(g) for g in groups.values()), default=0))
                for group in groups.values()
                if i < len(group)
            ]
            if len(balanced) < per_dataset:
                raise ValueError(f"insufficient {collection_role} samples in dataset_{fold}")
            ids.extend(balanced[:per_dataset])
    known = {e.sample_id for e in development.entries if e.data_role == "final_train"}
    trained = {e.sample_id for e in train.entries}
    if (
        not set(ids) <= (trained if tradeoff is not None and collection_role == "train" else known)
        or (collection_role != "train" and trained.intersection(ids))
        or (tradeoff is None and collection_role == "train")
        or any(sid.split("/")[0] not in {f"dataset_{i}" for i in range(1, 8)} for sid in ids)
    ):
        raise ValueError("native probe IDs must be disjoint original-SCARED development 1..7")
    if (
        probe["development_manifest_hash"] != development.manifest_hash
        or probe["training_manifest_hash"] != train.manifest_hash
        or probe["role_hash"] != roles.role_hash
    ):
        raise ValueError("CPU/native manifest or role identity mismatch")
    configurations = [GeometryRepairConfig(**c["resolved"]) for c in probe["configs"]]
    if any(
        c.config_hash != row["hash"]
        for c, row in zip(configurations, probe["configs"], strict=True)
    ):
        raise ValueError("CPU probe candidate hash mismatch")
    cfg = next((c for c in configurations if c.mode == "correspondence"), None)
    if cfg is None:
        raise ValueError("CPU probe contains no correspondence candidate")
    raw = read_student_checkpoint(checkpoint)
    if (
        raw["identity"]["manifest_hash"] != train.manifest_hash
        or raw["identity"]["role_hash"] != roles.role_hash
    ):
        raise ValueError("checkpoint training/role identity mismatch")
    construction = ReferenceConfig.from_dict(raw["construction_config"])
    renderer = ProductionFieldRenderer()
    if (
        renderer.identity != construction.renderer_identity
        or renderer.identity != raw["identity"]["renderer_identity"]
    ):
        raise ValueError("native renderer identity mismatch")
    device = torch.device("cuda:0")
    learning = StudentLearningConfig(**raw["student_config"])
    routing = RoutingConfig(**raw["routing_config"])
    protocol = InterventionConfig(**raw["intervention_config"])
    model = StudentReconstructionModel(
        renderer=renderer,
        config=construction,
        width=learning.width,
        bins=learning.bins,
        minimum_effective_px=learning.minimum_effective_px,
    ).to(device)
    learner = StudentLearner(model, learning, routing, intervention_config=protocol)
    imported = import_frozen_student_actions(
        checkpoint, learner, expected_sha256=probe["checkpoint_sha256"]
    )
    candidate = GeometryRepair(cfg)
    adapter = ScaredA00Adapter(
        data_root,
        train if tradeoff is not None and collection_role == "train" else development,
        roles=roles,
    )
    before_weights = _weights(learner)
    hw = construction.runtime.resolution_hw
    if hw is None or min(hw) < 2:
        raise ValueError("native geometry requires a pinned image grid with four cores")
    regions = grid_regions(hw, ((hw[0] + 1) // 2, (hw[1] + 1) // 2), routing.halo)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    proxy_choices: Counter[str] = Counter()
    depth_choices: Counter[str] = Counter()
    torch.cuda.reset_peak_memory_stats(device)
    with (
        torch.no_grad(),
        evaluation_mode(model, learner.actions),
        (output_dir / "samples.jsonl").open("x") as stream,
    ):
        warm = adapter.load(
            ids[0], data_role="final_train", include_supervision=False, output_hw=hw
        )
        batch = _move_batch(warm.batch, device)
        state = RepairState(
            batch,
            model(
                batch,
                sample_ids=(ids[0],),
                data_role="final_train",
                calibration_metadata=warm.calibration_metadata,
                include_matching_evidence=tradeoff is not None,
            ),
        )
        first = next(r for r in regions if eligible(state, r))
        candidate.execute(state, first, model)
        learner.actions.execute(state, first, Action.STEREO, model)
        learner.actions.execute(state, first, Action.GAUSSIAN, model)
        del state, batch, warm
        for sid in ids:
            sample = adapter.load(
                sid, data_role="final_train", include_supervision=False, output_hw=hw
            )
            assert sample.supervision is None
            batch = _move_batch(sample.batch, device)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            state = RepairState(
                batch,
                model(
                    batch,
                    sample_ids=(sid,),
                    data_role="final_train",
                    calibration_metadata=sample.calibration_metadata,
                    include_matching_evidence=tradeoff is not None,
                ),
            )
            torch.cuda.synchronize(device)
            base_ms = (time.perf_counter() - t0) * 1000
            base_ms_without_matching = base_ms
            feature_values: list[list[float]] | None = None
            feature_ms = 0.0
            if tradeoff is not None:
                # Fixed/constant policies can omit matching-evidence extraction.
                # Measure their actual base path instead of charging router work.
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                plain_base = model(
                    batch,
                    sample_ids=(sid,),
                    data_role="final_train",
                    calibration_metadata=sample.calibration_metadata,
                    include_matching_evidence=False,
                )
                torch.cuda.synchronize(device)
                base_ms_without_matching = (time.perf_counter() - t0) * 1000
                if not torch.equal(plain_base.target.disparity, state.base.target.disparity):
                    raise ValueError("matching evidence changed base disparity")
                for name in ("means3d", "scales", "rotations", "opacities", "colors", "valid_mask"):
                    if not torch.equal(
                        getattr(plain_base.gaussians, name), getattr(state.base.gaussians, name)
                    ):
                        raise ValueError("matching evidence changed base Gaussian field")
                del plain_base
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                feature_tensor = balanced_features(
                    state, regions, protocol, scale_unit_m=model.config.context_scale_m
                )
                torch.cuda.synchronize(device)
                feature_ms = (time.perf_counter() - t0) * 1000
                feature_values = feature_tensor.cpu().tolist()
            stop = learner.actions.execute(state, regions[0], Action.STOP, model)
            pending: list[tuple[Region, str, ActionResult, dict[str, Any]]] = []
            for region in regions:
                if not eligible(state, region):
                    continue
                quality_before = measure_quality(state, stop, region).values()
                for name in ("correspondence", "learned_stereo", "gaussian"):
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    outcome = (
                        candidate.execute(state, region, model)
                        if name == "correspondence"
                        else learner.actions.execute(
                            state,
                            region,
                            Action.STEREO if name == "learned_stereo" else Action.GAUSSIAN,
                            model,
                        )
                    )
                    torch.cuda.synchronize(device)
                    action_ms = (time.perf_counter() - t0) * 1000
                    action_times = [action_ms]
                    if tradeoff is not None:
                        for _ in range(timing_repeats - 1):
                            torch.cuda.synchronize(device)
                            t0 = time.perf_counter()
                            if name == "correspondence":
                                candidate.execute(state, region, model)
                            else:
                                learner.actions.execute(
                                    state,
                                    region,
                                    Action.STEREO if name == "learned_stereo" else Action.GAUSSIAN,
                                    model,
                                )
                            torch.cuda.synchronize(device)
                            action_times.append((time.perf_counter() - t0) * 1000)
                        action_ms = statistics.median(action_times)
                    quality_after = measure_quality(state, outcome, region).values()
                    gain, local_gain, harm = component_targets(
                        quality_before, quality_after, protocol
                    )
                    pending.append(
                        (
                            region,
                            name,
                            outcome,
                            {
                                "action_ms": action_ms,
                                "action_ms_samples": action_times,
                                "action_ms_max": max(action_times),
                                "before": quality_before,
                                "after": quality_after,
                                "global_gain": gain,
                                "local_gain": local_gain,
                                "proxy_harm": harm,
                                "within_budget": base_ms + action_ms <= protocol.total_budget_ms,
                                "changed_disparity_pixels": int(
                                    (outcome.disparity != stop.disparity).sum()
                                ),
                            },
                        )
                    )
            # Ground truth is first resolved after ALL deployment interventions.
            supervision = adapter.load(
                sid, data_role="final_train", include_supervision=True, output_hw=hw
            ).supervision
            assert supervision is not None
            gt_d, gt_mask = supervision.disparity.to(device), supervision.valid_mask.to(device)
            gt_geometry = reconstruct_rectified(
                gt_d, batch.camera, gt_mask, min_effective_disparity=1e-6
            )

            base_metrics = _geometry_metrics(stop, state, gt_d, gt_mask)
            base_metrics.update(
                left_psnr=psnr(stop.left, batch.left).to_dict(),
                right_psnr=psnr(stop.right, batch.right).to_dict(),
                left_ssim=windowed_ssim(stop.left.image, batch.left).to_dict(),
                right_ssim=windowed_ssim(stop.right.image, batch.right).to_dict(),
            )
            outcomes: list[dict[str, Any]] = []
            for region, name, outcome, row in pending:
                measured = _geometry_metrics(outcome, state, gt_d, gt_mask)
                measured.update(
                    left_psnr=psnr(outcome.left, batch.left).to_dict(),
                    right_psnr=psnr(outcome.right, batch.right).to_dict(),
                    left_ssim=windowed_ssim(outcome.left.image, batch.left).to_dict(),
                    right_ssim=windowed_ssim(outcome.right.image, batch.right).to_dict(),
                )
                local_mask = gt_geometry.depth.valid_mask & state.support & region.mask(batch.left)
                local_before = depth_metrics(
                    stop.disparity, gt_d, gt_geometry.depth.depth, local_mask, batch
                )["depth_mae_mm"]
                local_after = depth_metrics(
                    outcome.disparity, gt_d, gt_geometry.depth.depth, local_mask, batch
                )["depth_mae_mm"]
                local_depth_gain = (
                    local_before - local_after
                    if local_before is not None and local_after is not None
                    else None
                )
                extra: dict[str, Any] = {}
                if tradeoff is not None:
                    before_depth = depth_snapshot(
                        stop,
                        stop,
                        state,
                        gt_geometry.depth.depth,
                        gt_geometry.depth.valid_mask,
                        region,
                    )
                    after_depth = depth_snapshot(
                        outcome,
                        stop,
                        state,
                        gt_geometry.depth.depth,
                        gt_geometry.depth.valid_mask,
                        region,
                    )
                    changes = metric_changes(
                        base_metrics,
                        measured,
                        before_depth,
                        after_depth,
                        row["before"]["outside_rgb"],
                        row["after"]["outside_rgb"],
                        row["before"]["local_rgb"],
                        row["after"]["local_rgb"],
                    )
                    extra = {
                        "metric_changes": changes,
                        "metric_observation_available": changes is not None,
                        "depth_before_fixed_support": before_depth,
                        "depth_after_fixed_support": after_depth,
                    }
                outcomes.append(
                    {
                        "region_id": region.index,
                        "action": name,
                        **row,
                        "metrics": measured,
                        "local_depth_gain_mm": local_depth_gain,
                        **extra,
                    }
                )
            safe = [
                r
                for r in outcomes
                if not r["proxy_harm"]
                and r["within_budget"]
                and r["global_gain"] > protocol.minimum_gain
            ]
            best = max(safe, key=lambda r: r["global_gain"], default=None)
            proxy_choices[best["action"] if best else "STOP"] += 1
            base_depth = base_metrics["stereo_depth_mae_mm"]["value"]
            possible = [
                r
                for r in outcomes
                if r["local_depth_gain_mm"] is not None
                and r["local_depth_gain_mm"] >= probe["minimum_local_gain_mm"]
                and r["metrics"]["stereo_depth_mae_mm"]["value"] is not None
            ]
            depth_best = min(
                possible, key=lambda r: r["metrics"]["stereo_depth_mae_mm"]["value"], default=None
            )
            depth_choices[
                depth_best["action"]
                if depth_best
                and base_depth is not None
                and depth_best["metrics"]["stereo_depth_mae_mm"]["value"] < base_depth
                else "STOP"
            ] += 1
            record = {
                "sample_id": sid,
                "base_ms": base_ms,
                "base_ms_without_matching": base_ms_without_matching,
                "base_metrics": base_metrics,
                "feature_ms": feature_ms,
                "features_raw": feature_values,
                "valid_regions": [eligible(state, region) for region in regions],
                "outcomes": outcomes,
                "proxy_oracle": {"action": best["action"], "region_id": best["region_id"]}
                if best
                else {"action": "STOP"},
            }
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "event": "native_geometry_progress",
                        "completed": len(records),
                        "sample_id": sid,
                    }
                ),
                flush=True,
            )
    summary = {}
    for name in ("correspondence", "learned_stereo", "gaussian"):
        rows = [row for record in records for row in record["outcomes"] if row["action"] == name]
        gains = [
            row["local_depth_gain_mm"] for row in rows if row["local_depth_gain_mm"] is not None
        ]
        summary[name] = {
            "regions": len(rows),
            "GT_valid_regions": len(gains),
            "proxy_harm_regions": sum(row["proxy_harm"] for row in rows),
            "positive_proxy_gain_regions": sum(row["global_gain"] > 0 for row in rows),
            "depth_improved_at_least_margin": sum(
                g >= probe["minimum_local_gain_mm"] for g in gains
            ),
            "depth_worsened_at_least_margin": sum(
                g <= -probe["minimum_local_gain_mm"] for g in gains
            ),
            "mean_local_depth_gain_mm": statistics.mean(gains) if gains else None,
            "mean_action_ms": statistics.mean(row["action_ms"] for row in rows),
            "budget_overruns": sum(not row["within_budget"] for row in rows),
        }
    if _weights(learner) != before_weights:
        raise ValueError("native geometry evaluation mutated model/action state")
    report: dict[str, object] = {
        "status": "native_geometry_comparison_complete",
        "samples": len(records),
        "summary": summary,
        "proxy_oracle_actions": dict(proxy_choices),
        "depth_oracle_actions": dict(depth_choices),
        "imported_weights": imported,
        "source_sha256": source_sha256(repo),
        "git_commit": get_git_commit(repo),
        "git_dirty": is_git_dirty(repo),
        "probe_report_sha256": hashlib.sha256(probe_bytes).hexdigest(),
        "candidate_config_hash": cfg.config_hash,
        "training_manifest_hash": train.manifest_hash,
        "development_manifest_hash": development.manifest_hash,
        "role_hash": roles.role_hash,
        "renderer_identity": renderer.identity,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": time.perf_counter() - started,
        "weights_unchanged": True,
        "training_executed": False,
        "teacher_inference_executed": False,
        "selector_executed": False,
        "limitations": [
            "14-frame development diagnostic by default, not final test or gate certification.",
            "Single-pass timings exclude I/O, GT/oracle overhead; not a latency benchmark.",
            "Depth oracle and proxy oracle have different objectives; no learned deployment selector is evaluated.",
            "Cloud metrics use a 256-point approximation; no paper benchmark parity claim.",
        ],
    }
    if tradeoff is not None:
        from dataclasses import asdict

        report.update(
            status="metric_outcome_table_complete",
            table_schema="metric_outcome_table.v1",
            collection_role=collection_role,
            sample_ids=ids,
            sample_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            action_names=list(ACTION_NAMES),
            component_names=list(COMPONENT_NAMES),
            feature_names=list(BALANCED_FEATURE_NAMES),
            tradeoff_config=asdict(tradeoff),
            tradeoff_config_hash=tradeoff.config_hash,
            candidate_config=asdict(cfg),
            intervention_config=protocol.to_dict(),
            outcome_sha256=hashlib.sha256((output_dir / "samples.jsonl").read_bytes()).hexdigest(),
            timing_repeats=timing_repeats,
            output_hw=list(hw),
            resolved_data_root=str(data_root.resolve()),
            timing_protocol="warm base and each action; synchronized wall-ms repeated action; median/max; base/features single-pass; excludes I/O and offline metrics",
            GT_semantics="offline metric targets only, resolved after pre-action features and all interventions",
        )
    write_json_atomic(output_dir / "result.json", report)
    return report
