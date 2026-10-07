"""Bounded multi-fold mask learning and independent native development checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from reliable_endo_gs.config.intervention import load_intervention_config
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, ScaredA00Sample, load_a01_manifest
from reliable_endo_gs.data.target_masks import TargetMaskIndex
from reliable_endo_gs.evaluation.intervention_selection import evaluate_intervention_selection
from reliable_endo_gs.reconstruction import ReferenceConfig
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.rendering.production import ProductionFieldRenderer
from reliable_endo_gs.routing.config import load_routing_config
from reliable_endo_gs.routing.masks import load_mask_config
from reliable_endo_gs.routing.tradeoff import load_tradeoff_config
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.runtime.identity import source_sha256
from reliable_endo_gs.teachers import AlignedTeacherTarget, TargetCache, load_a01_roles
from reliable_endo_gs.teachers.external import build_foundation_stereo
from reliable_endo_gs.teachers.workflow import target_cache_key
from reliable_endo_gs.training.student_checkpoint import (
    initialize_student_model,
    read_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_config import StudentLearningConfig
from reliable_endo_gs.training.student_learning import StudentLearner
from reliable_endo_gs.training.student_run import run_student_training
from reliable_endo_gs.utils.io import write_json_atomic
from reliable_endo_gs.utils.seed import set_seed


class MaskValidationSamples:
    """Read worklist IDs through the ORIGINAL manifest, preserving cache identity."""

    def __init__(
        self, adapter: ScaredA00Adapter, ids: tuple[str, ...], hw: tuple[int, int]
    ) -> None:
        self.adapter, self.ids, self.hw = adapter, ids, hw

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, position: int) -> ScaredA00Sample:
        return self.adapter.load(
            self.ids[position],
            data_role="final_train",
            include_supervision=False,
            output_hw=self.hw,
        )


def validate_worklist(
    raw: Mapping[str, object], train_ids: set[str], dev_ids: set[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """A bounded worklist is not a replacement role/split manifest."""
    if raw.get("schema") != "mask_validation_worklist.v1":
        raise ValueError("unsupported mask validation worklist")
    groups = []
    for name, known, maximum in (("training", train_ids, 112), ("development", dev_ids, 56)):
        ids = raw.get(name)
        if (
            not isinstance(ids, list)
            or not 7 <= len(ids) <= maximum
            or any(not isinstance(s, str) for s in ids)
            or len(set(ids)) != len(ids)
            or not set(ids) <= known
        ):
            raise ValueError(f"invalid bounded {name} worklist")
        if {s.split("/")[0] for s in ids} != {f"dataset_{i}" for i in range(1, 8)}:
            raise ValueError("worklist must cover original SCARED datasets 1..7 only")
        groups.append(tuple(ids))
    if set(groups[0]).intersection(groups[1]) or train_ids.intersection(groups[1]):
        raise ValueError("development worklist overlaps training")
    return groups[0], groups[1]


def run_mask_validation(
    *,
    data_root: Path,
    training_manifest: Path,
    development_manifest: Path,
    roles_path: Path,
    worklist_path: Path,
    target_mask_index: Path,
    cache_root: Path,
    initial_checkpoint: Path,
    output_dir: Path,
    mask_config_path: Path,
    intervention_config_path: Path,
    routing_config_path: Path,
    metric_config_path: Path,
    epochs: int = 3,
    preflight_only: bool = False,
) -> dict[str, object]:
    """CPU asset checks first; one bounded joint run; read-only native comparisons.

    Original cohort manifests remain untouched. KD cache is read-only; misses
    fail before training. GT is first read by evaluation after deployment steps.
    Initial baseline and final masks use the same fixed development worklist.
    """
    repo = Path(__file__).resolve().parents[3]
    if output_dir.exists() or output_dir.resolve() == repo or repo in output_dir.resolve().parents:
        raise ValueError("validation needs a new output directory outside repository")
    if type(epochs) is not int or not 1 <= epochs <= 5:
        raise ValueError("bounded validation requires 1..5 joint epochs")
    roles = load_a01_roles(roles_path)
    mask_config = load_mask_config(mask_config_path)
    intervention_config = load_intervention_config(intervention_config_path)
    routing_config = load_routing_config(routing_config_path)
    metric_config = load_tradeoff_config(metric_config_path)
    train, dev = (
        load_a01_manifest(p, roles=roles) for p in (training_manifest, development_manifest)
    )
    content = worklist_path.read_bytes()
    work = json.loads(content)
    train_ids, dev_ids = validate_worklist(
        work,
        {e.sample_id for e in train.entries if e.data_role == "final_train"},
        {e.sample_id for e in dev.entries if e.data_role == "final_train"},
    )
    if (
        work.get("training_manifest_hash") != train.manifest_hash
        or work.get("development_manifest_hash") != dev.manifest_hash
        or work.get("role_hash") != roles.role_hash
    ):
        raise ValueError("worklist parent manifest identity mismatch")
    raw = read_student_checkpoint(initial_checkpoint)
    construction = ReferenceConfig.from_dict(raw["construction_config"])
    if (
        raw["identity"]["manifest_hash"] != train.manifest_hash
        or raw["identity"]["role_hash"] != roles.role_hash
    ):
        raise ValueError("initial model parent cohort/role mismatch")
    hw = construction.runtime.resolution_hw
    if (
        hw is None
        or construction.runtime.precision != "fp32"
        or construction.runtime.device != "cuda:0"
    ):
        raise ValueError("bounded native validation requires pinned FP32 cuda:0 output grid")
    masks = TargetMaskIndex(target_mask_index)
    if masks.identity != work.get("target_mask_index_sha256"):
        raise ValueError("worklist ROI index identity mismatch")
    masks.require_samples((*train_ids, *dev_ids), hw)
    teacher = build_foundation_stereo()
    cache = TargetCache(cache_root, roles=roles, read_only=True)
    adapter = ScaredA00Adapter(data_root, train, roles=roles)
    samples = MaskValidationSamples(adapter, train_ids, hw)
    cache_keys = {}
    print(
        json.dumps(
            {
                "event": "mask_validation_preflight_start",
                "training_samples": len(train_ids),
                "development_samples": len(dev_ids),
            }
        ),
        flush=True,
    )
    for i, sid in enumerate(train_ids):
        sample = samples[i]
        key = target_cache_key(
            sample, teacher, split_identity=construction.split_identity, roles=roles
        )
        target = cache.get(key)
        if target is None:
            raise ValueError(f"missing exact pinned training KD cache: {sid}")
        masks.load(sid, hw, torch.device("cpu"))
        cache_keys[sid] = key.digest
        if i == 0 or (i + 1) % 8 == 0 or i + 1 == len(train_ids):
            print(
                json.dumps(
                    {
                        "event": "mask_validation_cache_verified",
                        "samples": i + 1,
                        "total": len(train_ids),
                    }
                ),
                flush=True,
            )
    dev_adapter = ScaredA00Adapter(data_root, dev, roles=roles)
    for sid in dev_ids:
        dev_adapter.validate_deployment_files(sid, data_role="final_train")
        masks.load(sid, hw, torch.device("cpu"))
    output_dir.mkdir(parents=True, exist_ok=False)
    preflight = {
        "status": "mask_validation_assets_verified",
        "training_samples": len(train_ids),
        "development_samples": len(dev_ids),
        "joint_epochs": epochs,
        "planned_updates": epochs * len(train_ids),
        "cache_hits": len(cache_keys),
        "cache_digests": cache_keys,
        "worklist_sha256": hashlib.sha256(content).hexdigest(),
        "target_mask_index_sha256": masks.identity,
        "source_sha256": source_sha256(repo),
        "mask_config_hash": mask_config.config_hash,
        "intervention_config_hash": intervention_config.config_hash,
        "routing_config_hash": routing_config.config_hash,
        "metric_config_hash": metric_config.config_hash,
        "ground_truth_loaded": False,
        "teacher_inference_executed": False,
        "training_executed": False,
    }
    write_json_atomic(output_dir / "preflight.json", preflight)
    print(
        json.dumps(
            {
                "event": "mask_validation_preflight_passed",
                "cache_hits": len(cache_keys),
                "roi_samples": len(train_ids) + len(dev_ids),
            }
        ),
        flush=True,
    )
    if preflight_only:
        return preflight
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError(
            "native mask validation needs one visible GPU; CPU preflight passed but CUDA is unavailable"
        )
    device = torch.device("cuda:0")
    config = StudentLearningConfig(
        **{
            **raw["student_config"],
            "student_epochs": 0,
            "joint_epochs": epochs,
            "checkpoint_every": 25,
        }
    )
    set_seed(config.seed)
    model = StudentReconstructionModel(
        renderer=ProductionFieldRenderer(),
        config=construction,
        width=config.width,
        bins=config.bins,
        minimum_effective_px=config.minimum_effective_px,
    ).to(device)
    learner = StudentLearner(
        model,
        config,
        routing_config,
        intervention_config=intervention_config,
        mask_config=mask_config,
        target_masks=masks,
    )
    imported = initialize_student_model(initial_checkpoint, learner)
    identity = {
        "run_kind": "bounded_mask_validation",
        "source_sha256": source_sha256(repo),
        "manifest_hash": train.manifest_hash,
        "role_hash": roles.role_hash,
        "renderer_identity": model._renderer_identity(),
        "git_commit": get_git_commit(repo),
        "git_dirty": is_git_dirty(repo),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "seed": config.seed,
        "samples_per_epoch": len(samples),
        "training_worklist": list(train_ids),
        "worklist_sha256": preflight["worklist_sha256"],
        "model_initialization": imported,
        "target_mask_index_sha256": masks.identity,
    }
    initial = output_dir / "initial.pt"
    save_student_checkpoint(
        initial,
        learner,
        identity,
        {"epoch": 0, "next_sample_index": 0, "samples_seen": 0, "optimizer_steps": 0},
    )

    def evaluate(path: Path, destination: Path, *, base_only: bool = False) -> dict[str, object]:
        return evaluate_intervention_selection(
            checkpoint=path,
            output_dir=destination,
            data_root=data_root,
            manifest_path=development_manifest,
            training_manifest=training_manifest,
            roles_path=roles_path,
            max_samples=len(dev_ids),
            target_mask_index=target_mask_index,
            sample_ids=dev_ids,
            metric_config_path=metric_config_path,
            base_only=base_only,
        )

    before = evaluate(initial, output_dir / "base_before_training", base_only=True)

    def provide(sample: ScaredA00Sample) -> AlignedTeacherTarget:
        key = target_cache_key(
            sample, teacher, split_identity=construction.split_identity, roles=roles
        )
        target = cache.get(key)
        if target is None:
            raise ValueError("pinned KD cache disappeared after preflight")
        return target

    trained = run_student_training(
        samples, learner, identity, output_dir / "training", target_provider=provide
    )
    if not trained["schedule_complete"]:
        report: dict[str, object] = {
            "status": "mask_validation_stopped_before_evaluation",
            "training": trained,
        }
        write_json_atomic(output_dir / "result.json", report)
        return report
    evaluated = evaluate(Path(str(trained["checkpoint"])), output_dir / "development")
    before_summary = before["summary"]
    assert isinstance(before_summary, dict)
    report = {
        "status": "bounded_mask_validation_complete",
        "training": trained,
        "development": evaluated,
        "base_before_training": before_summary["base"],
        "teacher_inference_executed": False,
        "gate_II_passed": False,
        "limitations": [
            "Manual RGB input ROIs; no automatic foreground model.",
            "Frame-disjoint development shares keyframes with training.",
            "Proxy-trained selector; independent metric guards are diagnostic, not calibrated risk.",
            "Bounded pilot cannot establish publication performance or FPS.",
        ],
    }
    write_json_atomic(output_dir / "result.json", report)
    return report
