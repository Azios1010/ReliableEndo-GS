"""A bounded, stationary TRAIN refit followed by untouched native development."""

from __future__ import annotations

import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.config.selector_refit import (
    load_selector_refit_config,
    partition_selector_training,
)
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, load_a01_manifest
from reliable_endo_gs.data.selector_cohort import (
    validate_selector_worklist,
    verify_selector_source_snapshot,
)
from reliable_endo_gs.data.target_masks import TargetMaskIndex
from reliable_endo_gs.evaluation.intervention_selection import evaluate_intervention_selection
from reliable_endo_gs.reconstruction import ReferenceConfig
from reliable_endo_gs.reconstruction.checkpoint import _atomic_torch_save, _cpu_tree
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.rendering.production import ProductionFieldRenderer
from reliable_endo_gs.routing.calibrated_selector import CalibratedSelector
from reliable_endo_gs.routing.config import RoutingConfig
from reliable_endo_gs.routing.cost import (
    CostCalibrationPolicy,
    load_cost_calibration,
)
from reliable_endo_gs.routing.masks import MaskProposalConfig
from reliable_endo_gs.routing.metric_selection import module_state_sha256
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.runtime.identity import source_sha256
from reliable_endo_gs.teachers import load_a01_roles
from reliable_endo_gs.training.deployment_costs import calibrate_student_costs
from reliable_endo_gs.training.harness import capture_rng, restore_rng
from reliable_endo_gs.training.mask_validation import MaskValidationSamples
from reliable_endo_gs.training.metric_quality import OfflineMetricTruth
from reliable_endo_gs.training.selector_fitting import calibrate_selector, fit_selector_heads
from reliable_endo_gs.training.selector_outcomes import (
    load_selector_outcomes,
)
from reliable_endo_gs.training.selector_shards import collect_selector_shards
from reliable_endo_gs.training.student_checkpoint import (
    import_frozen_student_actions,
    read_student_checkpoint,
)
from reliable_endo_gs.training.student_config import StudentLearningConfig
from reliable_endo_gs.training.student_learning import StudentLearner
from reliable_endo_gs.utils.io import write_json_atomic


def _validate_deployment_cohort(
    adapter: ScaredA00Adapter,
    masks: TargetMaskIndex,
    sample_ids: tuple[str, ...],
    output_hw: tuple[int, int],
    role: str,
    *,
    reuse_rgb_validation: bool = False,
) -> None:
    """Two bounded CPU I/O workers; fully decode RGB, validate camera and ROI, no GT."""

    def validate(sample_id: str) -> None:
        if not reuse_rgb_validation:
            adapter.validate_deployment_files(sample_id, data_role="final_train")
        masks.load(sample_id, output_hw, torch.device("cpu"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        for index, _ in enumerate(executor.map(validate, sample_ids)):
            if index == 0 or (index + 1) % (250 if len(sample_ids) > 256 else 8) == 0:
                print(
                    json.dumps(
                        {
                            "event": "selector_refit_preflight_progress",
                            "role": role,
                            "samples": index + 1,
                            "total": len(sample_ids),
                            "io_workers": 2,
                            "rgb_validation_reused": reuse_rgb_validation,
                        }
                    ),
                    flush=True,
                )


def run_selector_refit(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    data_root: Path,
    training_manifest: Path,
    development_manifest: Path,
    roles_path: Path,
    worklist_path: Path,
    target_mask_index: Path,
    refit_config_path: Path,
    cost_calibration_path: Path | None,
    output_dir: Path,
    device_name: str = "cuda:0",
    preflight_only: bool = False,
    prepare_costs: bool = False,
    resume: bool = False,
    collect_only: bool = False,
    routing_profile_reference: Path | None = None,
    routing_profile_selector: Path | None = None,
) -> dict[str, Any]:
    """New immutable outputs; model/actions, original splits and teacher stay fixed."""
    repo = Path(__file__).resolve().parents[3]
    if (routing_profile_reference is None) != (routing_profile_selector is None):
        raise ValueError("routing profiling needs both archived source and pinned selector")
    if (
        (output_dir.exists() and not resume)
        or output_dir.resolve() == repo
        or repo in output_dir.resolve().parents
    ):
        raise ValueError("refit requires a new external output directory")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_sha256:
        raise ValueError("pinned source checkpoint digest mismatch")
    policy = load_selector_refit_config(refit_config_path)
    raw = read_student_checkpoint(checkpoint)
    roles = load_a01_roles(roles_path)
    train, dev = (
        load_a01_manifest(p, roles=roles) for p in (training_manifest, development_manifest)
    )
    work = json.loads(worklist_path.read_text())
    train_ids, dev_ids = validate_selector_worklist(
        work,
        {e.sample_id for e in train.entries if e.data_role == "final_train"},
        {e.sample_id for e in dev.entries if e.data_role == "final_train"},
        policy,
    )
    if (
        work.get("training_manifest_hash") != train.manifest_hash
        or work.get("development_manifest_hash") != dev.manifest_hash
        or work.get("role_hash") != roles.role_hash
        or raw["identity"]["manifest_hash"] != train.manifest_hash
        or raw["identity"]["role_hash"] != roles.role_hash
    ):
        raise ValueError("original parent manifest/role identity mismatch")
    construction = ReferenceConfig.from_dict(raw["construction_config"])
    hw = construction.runtime.resolution_hw
    if (
        hw is None
        or construction.runtime.precision != "fp32"
        or construction.runtime.device != device_name
    ):
        raise ValueError("refit requires checkpoint native FP32 output grid/device")
    config = StudentLearningConfig(**raw["student_config"])
    routing = RoutingConfig(**raw["routing_config"])
    intervention = InterventionConfig(**raw["intervention_config"])
    if intervention.selector_gain_mode != "regression":
        raise ValueError("scalar selector-refit requires component-aware signed gain calibration")
    if intervention.selector_risk_mode != "union":
        raise ValueError(
            "scalar selector-refit calibration does not support local/global/any risk; use joint training and event evaluation"
        )
    mask_config = MaskProposalConfig(**raw["mask_config"])
    masks = TargetMaskIndex(target_mask_index)
    if masks.identity != work.get("target_mask_index_sha256"):
        raise ValueError("deployment target mask identity mismatch")
    resolved_cost_path = cost_calibration_path or output_dir / "cost_calibration.pt"
    costs = load_cost_calibration(resolved_cost_path) if resolved_cost_path.is_file() else None
    if costs is not None and (
        not costs.matches_training_identity(raw["identity"])
        or costs.binding.get("training_manifest_hash") != train.manifest_hash
        or costs.binding.get("target_mask_index_sha256") != masks.identity
    ):
        raise ValueError("TRAIN cost calibration checkpoint/mask mismatch")
    masks.require_samples((*train_ids, *dev_ids), hw)
    source_snapshot_sha = work.get("data_source_index_sha256")
    if source_snapshot_sha is not None:
        verify_selector_source_snapshot(
            data_root, worklist_path.parent / "data-source-index.json", str(source_snapshot_sha)
        )
    partition = partition_selector_training(train_ids, policy)
    current_source = source_sha256(repo)
    provenance = {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_source_sha256": raw["identity"]["source_sha256"],
        "source_sha256": current_source,
        "git_commit": get_git_commit(repo),
        "git_dirty": is_git_dirty(repo),
        "training_manifest_hash": train.manifest_hash,
        "development_manifest_hash": dev.manifest_hash,
        "role_hash": roles.role_hash,
        "target_mask_index_sha256": masks.identity,
        "worklist_sha256": hashlib.sha256(worklist_path.read_bytes()).hexdigest(),
        "partition_sha256": partition["partition_sha256"],
        "refit_config_hash": policy.config_hash,
        "data_source_index_sha256": source_snapshot_sha,
    }
    reuse_rgb_validation = False
    if output_dir.exists():
        saved = json.loads((output_dir / "preflight.json").read_text())
        if (
            any(saved.get(k) != v for k, v in provenance.items())
            or saved.get("partition") != partition
        ):
            raise ValueError("resume preflight source/checkpoint/cohort/partition mismatch")
        # Source snapshot stats were just checked; only the exact sealed, fully
        # validated cohort may reuse RGB/calibration decoding. Masks are rehashed.
        reuse_rgb_validation = (
            source_snapshot_sha is not None and saved.get("deployment_files_verified") is True
        )
    adapter = ScaredA00Adapter(data_root, train, roles=roles)
    dev_adapter = ScaredA00Adapter(data_root, dev, roles=roles)
    for role, source, ids in (("train", adapter, train_ids), ("development", dev_adapter, dev_ids)):
        _validate_deployment_cohort(
            source, masks, ids, hw, role, reuse_rgb_validation=reuse_rgb_validation
        )
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
    # Seal before GT collection. Unknown/harm labels cannot alter this partition.
    write_json_atomic(output_dir / "partition.json", partition)
    preflight = {
        **provenance,
        "training_samples": len(train_ids),
        "development_samples": len(dev_ids),
        "partition": partition,
        "ground_truth_loaded": False,
        "teacher_inference_executed": False,
        "training_executed": False,
        "deployment_files_verified": True,
        "rgb_validation_reused": reuse_rgb_validation,
        "status": "selector_refit_assets_verified"
        if costs is not None
        else "selector_refit_costs_pending",
        "cost_calibration_path": str(resolved_cost_path),
    }
    write_json_atomic(output_dir / "preflight.json", preflight)
    print(
        json.dumps(
            {
                "event": "selector_refit_preflight_passed",
                "output_dir": str(output_dir),
                "training_samples": len(train_ids),
                "development_samples": len(dev_ids),
            }
        ),
        flush=True,
    )
    if preflight_only:
        return preflight
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("refit requires exactly one visible CUDA GPU")
    device = torch.device(device_name)
    rng = capture_rng(device)
    try:
        torch.manual_seed(policy.seed)
        model = StudentReconstructionModel(
            renderer=ProductionFieldRenderer(),
            config=construction,
            width=config.width,
            bins=config.bins,
            minimum_effective_px=config.minimum_effective_px,
        ).to(device)
        if model._renderer_identity() != raw["identity"]["renderer_identity"]:
            raise ValueError("native renderer identity mismatch")
        learner = StudentLearner(
            model,
            config,
            routing,
            intervention_config=intervention,
            mask_config=mask_config,
            target_masks=masks,
        )
        imported = import_frozen_student_actions(
            checkpoint, learner, expected_sha256=checkpoint_sha256
        )
        action_identity = learner.frozen_action_identity()
        original_selector = module_state_sha256(learner.selector)
        samples = MaskValidationSamples(adapter, train_ids, hw)
        if costs is None:
            if not prepare_costs:
                raise ValueError(
                    "missing dedicated TRAIN costs; run --prepare-costs before collection"
                )
            groups: dict[str, list[str]] = {}
            for sid in train_ids:
                groups.setdefault(sid.rsplit("/", 1)[0], []).append(sid)
            cost_ids = tuple(
                s for ids in groups.values() for s in (sorted(ids)[0], sorted(ids)[-1])
            )
            costs = calibrate_student_costs(
                learner,
                MaskValidationSamples(adapter, cost_ids, hw),
                raw["identity"],
                CostCalibrationPolicy("max", None, 1, 3),
            )
            if resolved_cost_path.exists():
                raise FileExistsError("refusing to overwrite dedicated TRAIN costs")
            _atomic_torch_save(resolved_cost_path, costs.to_payload())
        learner.attach_cost_calibration(costs)
        if prepare_costs:
            if source_snapshot_sha is not None:
                verify_selector_source_snapshot(
                    data_root,
                    worklist_path.parent / "data-source-index.json",
                    str(source_snapshot_sha),
                )
            if source_sha256(repo) != current_source:
                raise ValueError("source changed during cost preparation")
            result = {
                **preflight,
                "status": "ready_for_selector_outcome_collection",
                "cost_calibration_sha256": hashlib.sha256(
                    resolved_cost_path.read_bytes()
                ).hexdigest(),
                "ground_truth_loaded": False,
                "training_executed": False,
            }
            write_json_atomic(output_dir / "readiness.json", result)
            return result
        provenance["cost_calibration_sha256"] = hashlib.sha256(
            resolved_cost_path.read_bytes()
        ).hexdigest()
        if routing_profile_reference is not None and routing_profile_selector is not None:
            from reliable_endo_gs.profiling.routing_components import profile_routing_components

            profile = profile_routing_components(
                learner, samples[0], routing_profile_reference, routing_profile_selector
            )
            write_json_atomic(output_dir / "routing_profile.json", profile)

        def ground_truth(sid: str) -> OfflineMetricTruth:
            gt_sample = adapter.load(
                sid, data_role="final_train", include_supervision=True, output_hw=hw
            )
            return OfflineMetricTruth.from_a00(gt_sample)

        collection = collect_selector_shards(
            learner,
            train_ids,
            lambda sid: adapter.load(
                sid, data_role="final_train", include_supervision=False, output_hw=hw
            ),
            ground_truth,
            provenance=provenance,
            output_dir=output_dir / "train_outcomes",
            shard_size=policy.collection_shard_size,
            resume=resume,
        )
        if source_snapshot_sha is not None:
            verify_selector_source_snapshot(
                data_root, worklist_path.parent / "data-source-index.json", str(source_snapshot_sha)
            )
        table = load_selector_outcomes(output_dir / "train_outcomes", "train")
        if collect_only:
            if (
                learner.frozen_action_identity() != action_identity
                or source_sha256(repo) != current_source
            ):
                raise ValueError("source/model/actions changed during collection")
            result = {
                **provenance,
                "status": "corrected_TRAIN_outcomes_collected",
                "collection": collection,
                "label_observations": table.observation_summary(),
                "selector_fitted": False,
                "training_executed": False,
                "teacher_inference_executed": False,
                "development_evaluation_executed": False,
                "model_actions_unchanged": True,
            }
            write_json_atomic(output_dir / "collection_result.json", result)
            return result
        views = {
            name: table.view(tuple(partition[name + "_sample_ids"]))
            for name in ("fit", "validation", "calibration")
        }
        artifact = output_dir / "selector.pt"
        if artifact.exists():
            if not resume:
                raise ValueError("existing refit artifact requires resume")
            learner.load_refit_selector(artifact, current_source)
            saved_artifact = torch.load(artifact, map_location="cpu", weights_only=True)
            if (
                saved_artifact.get("partition") != partition
                or saved_artifact.get("refit_config_hash") != policy.config_hash
                or saved_artifact.get("train_outcomes_sha256") != table.metadata["samples_sha256"]
            ):
                raise ValueError("existing refit artifact outcome/partition/config mismatch")
            fit_report = saved_artifact["fit_report"]
            calibration = saved_artifact["calibration"]
        else:
            head, fit_report = fit_selector_heads(
                views["fit"], views["validation"], intervention, policy, routing.width, device
            )
            calibrated = CalibratedSelector(head, intervention, action_identity, masks.identity).to(
                device
            )
            calibration = calibrate_selector(
                calibrated,
                views["calibration"],
                policy,
                excluded_keyframes=set(partition["fit_keyframes"])
                | set(partition["validation_keyframes"]),
            )
            _atomic_torch_save(
                artifact,
                _cpu_tree(
                    {
                        **provenance,
                        "schema": "calibrated_selector.v1",
                        "width": routing.width,
                        "head_config": head.config.to_dict(),
                        "decision_config": intervention.to_dict(),
                        "action_identity": action_identity,
                        "state": calibrated.state_dict(),
                        "partition": partition,
                        "refit_config": asdict(policy),
                        "calibration": calibration,
                        "fit_report": fit_report,
                        "train_outcomes_sha256": table.metadata["samples_sha256"],
                        "development_used_for_fitting": False,
                    }
                ),
            )
            del head, calibrated
            learner.load_refit_selector(artifact, current_source)
        # Actual deployment entry point: GT-free sample, real action and renderer.
        deployment = learner.infer(samples[0], selected=True)
        smoke = {
            "sample_id": train_ids[0],
            "diagnostics": dict(deployment.diagnostics),
            "render_left_finite": bool(torch.isfinite(deployment.render_left.image).all()),
            "render_right_finite": None
            if deployment.render_right is None
            else bool(torch.isfinite(deployment.render_right.image).all()),
            "teacher_inference_executed": False,
            "ground_truth_loaded": False,
        }
        write_json_atomic(output_dir / "deployment_smoke.json", smoke)
        if (
            learner.frozen_action_identity() != action_identity
            or module_state_sha256(learner.selector) != original_selector
        ):
            raise ValueError("refit mutated original model/actions/selector")
        del deployment, learner, model
        torch.cuda.empty_cache()
        reports = {}
        for name, selector_path in (("original", None), ("refit", artifact)):
            stage = output_dir / name
            if stage.exists() and not (stage / "result.json").is_file():
                if not resume:
                    raise ValueError("partial evaluation requires resume")
                stage.rename(output_dir / (name + "-interrupted-" + uuid.uuid4().hex[:12]))
            if stage.exists():
                saved_report = json.loads((stage / "result.json").read_text())
                expected_selector = (
                    None
                    if selector_path is None
                    else hashlib.sha256(selector_path.read_bytes()).hexdigest()
                )
                if (
                    saved_report.get("checkpoint_sha256") != checkpoint_sha256
                    or saved_report.get("source_sha256") != current_source
                    or saved_report.get("selector_refit_sha256") != expected_selector
                    or saved_report.get("target_mask_index_sha256") != masks.identity
                    or saved_report.get("samples_sha256")
                    != hashlib.sha256((stage / "samples.jsonl").read_bytes()).hexdigest()
                    or [
                        json.loads(s)["sample_id"]
                        for s in (stage / "samples.jsonl").read_text().splitlines()
                    ]
                    != list(dev_ids)
                ):
                    raise ValueError("resume evaluation artifact mismatch")
                reports[name] = saved_report
                continue
            reports[name] = evaluate_intervention_selection(
                checkpoint=checkpoint,
                data_root=data_root,
                manifest_path=development_manifest,
                training_manifest=training_manifest,
                roles_path=roles_path,
                output_dir=output_dir / name,
                max_samples=len(dev_ids),
                device_name=device_name,
                target_mask_index=target_mask_index,
                sample_ids=dev_ids,
                cost_calibration_path=resolved_cost_path,
                expected_checkpoint_sha256=checkpoint_sha256,
                selector_refit_path=selector_path,
            )
        if (
            source_sha256(repo) != current_source
            or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_sha256
        ):
            raise ValueError("source/checkpoint changed during experiment")
        if source_snapshot_sha is not None:
            verify_selector_source_snapshot(
                data_root, worklist_path.parent / "data-source-index.json", str(source_snapshot_sha)
            )
        result = {
            **provenance,
            "status": "selector_refit_native_validation_complete",
            "partition": partition,
            "fit": fit_report,
            "calibration": calibration,
            "import": imported,
            "collection": collection,
            "artifact": str(artifact),
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "original_summary": reports["original"]["summary"],
            "refit_summary": reports["refit"]["summary"],
            "model_actions_unchanged": True,
            "joint_training_executed": False,
            "teacher_inference_executed": False,
            "final_test_executed": False,
            "development_used_for_fitting": False,
        }
        write_json_atomic(output_dir / "result.json", result)
        return result
    finally:
        restore_rng(rng, device)
