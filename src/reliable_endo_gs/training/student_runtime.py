"""External asset resolution for real student runs; optional frozen KD provider."""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import replace
from pathlib import Path

import torch

from reliable_endo_gs.actions.repair import Action
from reliable_endo_gs.data.scared_a00 import (
    ScaredA00Adapter,
    ScaredA00Dataset,
    ScaredA00Sample,
    load_a01_manifest,
)
from reliable_endo_gs.reconstruction import ReferenceConfig, load_reference_config
from reliable_endo_gs.reconstruction.checkpoint import _atomic_torch_save
from reliable_endo_gs.reconstruction.model import _freeze_injected_teacher
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.regions.grid import grid_regions
from reliable_endo_gs.rendering.production import ProductionFieldRenderer
from reliable_endo_gs.routing.config import RoutingConfig, load_routing_config
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.teachers import AlignedTeacherTarget, TargetCache, load_a01_roles
from reliable_endo_gs.training.harness import capture_rng, restore_rng
from reliable_endo_gs.training.student_checkpoint import (
    load_student_checkpoint,
    read_student_checkpoint,
)
from reliable_endo_gs.training.student_config import StudentLearningConfig, load_student_config
from reliable_endo_gs.training.student_learning import StudentLearner
from reliable_endo_gs.training.student_run import run_student_training
from reliable_endo_gs.utils.io import write_json_atomic
from reliable_endo_gs.utils.seed import set_seed


def run_student_command(arguments: argparse.Namespace) -> dict[str, object]:
    """Native runs are explicit CLI requests; inference never resolves a teacher."""
    infer = arguments.command == "infer"
    smoke = arguments.command == "smoke"
    if infer and (arguments.checkpoint is None or arguments.sample_id is None):
        raise ValueError("infer requires --checkpoint and --sample-id")
    if infer and (arguments.cache_root is not None or arguments.teacher_factory is not None):
        raise ValueError("student-only inference does not accept teacher/cache inputs")
    if not infer and arguments.data_role != "final_train":
        raise ValueError("learning/smoke cannot consume final_test")
    if smoke and (arguments.sample_id is None or arguments.checkpoint is not None):
        raise ValueError("smoke requires --sample-id and fresh initialization")
    if arguments.command == "train" and arguments.sample_id is not None:
        raise ValueError("training uses the unchanged final_train manifest, never final_test")
    if arguments.selected and not infer:
        raise ValueError("--selected is an inference option")
    if arguments.max_steps is not None and (
        arguments.command != "train" or arguments.max_steps < 1
    ):
        raise ValueError("--max-steps is a positive training bound")
    repo = Path(__file__).resolve().parents[3]
    output = arguments.output_dir.resolve()
    if output == repo or repo in output.parents:
        raise ValueError("student artifacts must be outside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError(
            "real student rendering requires one visible GPU; use CPU synthetic tests locally"
        )
    device = torch.device("cuda:0")
    raw = read_student_checkpoint(arguments.checkpoint) if infer else None
    if raw is not None:
        config = StudentLearningConfig(**raw["student_config"])
        routing_data = dict(raw["routing_config"])
        routing_data["tile_hw"] = tuple(routing_data["tile_hw"])
        routing = RoutingConfig(**routing_data)
        construction = ReferenceConfig.from_dict(raw["construction_config"])
        if arguments.config is not None and load_student_config(arguments.config) != config:
            raise ValueError("inference student config mismatch")
        if (
            arguments.routing_config is not None
            and load_routing_config(arguments.routing_config) != routing
        ):
            raise ValueError("inference routing config mismatch")
        if arguments.construction_config is not None:
            raise ValueError("inference construction is pinned by the checkpoint")
    else:
        config = load_student_config(
            arguments.config or Path("configs/training/student_joint.yaml")
        )
        routing = load_routing_config(
            arguments.routing_config or Path("configs/routing/frozen_a04.yaml")
        )
        construction = load_reference_config(
            arguments.construction_config or Path("configs/reconstruction/a04_reference.yaml")
        )
        # This experiment deliberately uses FP32 training; no unsaved AMP scaler.
        construction = replace(
            construction, seed=config.seed, runtime=replace(construction.runtime, precision="fp32")
        )
        if smoke and (config.student_epochs != 1 or config.joint_epochs != 1):
            raise ValueError("bounded smoke requires exactly one student and one joint epoch")
    if construction.runtime.device != "cuda:0" or construction.data_role != "final_train":
        raise ValueError("native student construction must pin cuda:0 and final_train")
    if construction.runtime.precision != "fp32":
        raise ValueError("student runtime/checkpoint must declare FP32 precision")
    roles = load_a01_roles(arguments.roles)
    manifest = load_a01_manifest(arguments.manifest, roles=roles)
    adapter = ScaredA00Adapter(arguments.data_root, manifest, roles=roles)
    dataset = ScaredA00Dataset(
        adapter,
        include_supervision=False,
        data_role="final_train",
        output_hw=construction.runtime.resolution_hw,
    )
    set_seed(config.seed)
    model = StudentReconstructionModel(
        renderer=ProductionFieldRenderer(),
        config=construction,
        width=config.width,
        bins=config.bins,
        minimum_effective_px=config.minimum_effective_px,
    ).to(device)
    learner = StudentLearner(model, config, routing)
    digest = hashlib.sha256()
    for path in sorted((repo / "src/reliable_endo_gs").rglob("*.py")):
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    identity: dict[str, object] = {
        "source_sha256": digest.hexdigest(),
        "manifest_hash": manifest.manifest_hash,
        "role_hash": roles.role_hash,
        "renderer_identity": model.config.renderer_identity,
        "git_commit": get_git_commit(repo),
        "git_dirty": is_git_dirty(repo),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
        "samples_per_epoch": len(dataset),
        "seed": config.seed,
        "precision": "fp32",
        "run_kind": "bounded_smoke" if smoke else "full_manifest",
    }
    if infer:
        assert arguments.checkpoint is not None
        load_student_checkpoint(arguments.checkpoint, learner, identity, len(dataset), resume=False)
        sample = adapter.load(
            arguments.sample_id,
            data_role=arguments.data_role,
            include_supervision=False,
            output_hw=construction.runtime.resolution_hw,
        )
        torch.cuda.reset_peak_memory_stats(device)
        result = learner.infer(sample, selected=arguments.selected)
        output.mkdir(parents=True, exist_ok=False)
        _atomic_torch_save(
            output / "prediction.pt",
            {
                "disparity": result.stereo.disparity.cpu(),
                "render_left": result.render_left.image.cpu(),
                "render_right": None
                if result.render_right is None
                else result.render_right.image.cpu(),
                "means3d": result.gaussians.means3d.cpu(),
                "source_index": result.prior.source_index.cpu(),
            },
        )
        report: dict[str, object] = {
            "sample_id": sample.sample_id,
            "teacher_loaded": False,
            "provenance": result.provenance_record.to_dict(),
            "selected_action": result.diagnostics.get("selected_action", 0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
    else:
        target_provider = None
        if config.kd_weight:
            # Lazy branch: none of this is entered by student-only inference.
            from reliable_endo_gs.teachers.workflow import (
                infer_and_align,
                load_teacher_factory,
                target_cache_key,
            )

            teacher = load_teacher_factory(
                arguments.teacher_factory
                or "reliable_endo_gs.teachers.external:build_foundation_stereo"
            )
            identity["teacher_identity"] = teacher.identity.to_dict()
            cache = TargetCache(arguments.cache_root, roles=roles) if arguments.cache_root else None
            # Cached supervision requires only the pinned identity descriptor.
            # LazyTeacherAdapter.freeze() loads weights, so call it only when
            # live inference is requested. No cache miss may trigger inference.
            if cache is None:
                _freeze_injected_teacher(teacher)

            def provide(sample: ScaredA00Sample) -> AlignedTeacherTarget:
                if cache is not None:
                    key = target_cache_key(
                        sample, teacher, split_identity=construction.split_identity, roles=roles
                    )
                    target = cache.get(key)
                    if target is None:
                        raise ValueError(
                            f"missing pinned KD cache item: {sample.sample_id}; no cache generation"
                        )
                    return target
                rng = capture_rng(device)
                try:
                    return infer_and_align(sample, teacher, device=device)
                finally:
                    restore_rng(rng, device)

            target_provider = provide
        elif arguments.teacher_factory is not None or arguments.cache_root is not None:
            raise ValueError("teacher/cache inputs require positive KD weight")
        samples = dataset
        if smoke:
            sample = adapter.load(
                arguments.sample_id,
                data_role="final_train",
                include_supervision=False,
                output_hw=construction.runtime.resolution_hw,
            )
            from reliable_endo_gs.training.student_learning import evaluation_mode

            with evaluation_mode(model, learner.actions):
                state = learner.state(sample)
                height, width = state.batch.left.shape[-2:]
                region = grid_regions((height, width), (height, width), routing.halo)[0]
                for action in (Action.STEREO, Action.GAUSSIAN):
                    outcome = learner.actions.execute(state, region, action, model)
                    for name in ("means3d", "scales", "rotations", "opacities"):
                        torch.testing.assert_close(
                            getattr(outcome.field, name),
                            getattr(state.base.gaussians, name),
                            rtol=1e-5,
                            atol=1e-7,
                        )
                    for repaired, original in (
                        (outcome.left, state.base.render_left),
                        (outcome.right, state.base.render_right),
                    ):
                        assert original is not None
                        torch.testing.assert_close(
                            repaired.image, original.image, rtol=1e-5, atol=2e-6
                        )
            smoke_samples = [sample]
            identity["samples_per_epoch"] = 1
            identity["smoke_sample_id"] = sample.sample_id
            smoke_modules = {
                "student": model.stereo_student,
                "head": model.head,
                "stereo_repair": learner.actions.stereo,
                "gaussian_repair": learner.actions.gaussian,
                "selector": learner.selector,
            }
            before = {
                name: {
                    key: value.detach().cpu().clone() for key, value in module.state_dict().items()
                }
                for name, module in smoke_modules.items()
            }
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
        report = run_student_training(
            smoke_samples if smoke else samples,
            learner,
            identity,
            output,
            max_steps=2 if smoke else arguments.max_steps,
            resume=arguments.checkpoint,
            target_provider=target_provider,
        )
        if smoke:
            torch.cuda.synchronize(device)
            report["smoke_only_not_full_training"] = True
            report["elapsed_training_seconds"] = time.perf_counter() - started
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            changed = {
                name: any(
                    not torch.equal(value.detach().cpu(), before[name][key])
                    for key, value in module.state_dict().items()
                )
                for name, module in smoke_modules.items()
            }
            if not all(changed.values()):
                raise ValueError(
                    f"smoke failed to update every scheduled parameter group: {changed}"
                )
            report["parameters_changed"] = changed
    write_json_atomic(output / "result.json", report)
    return report
