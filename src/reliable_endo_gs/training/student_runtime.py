"""External asset resolution for real student runs; optional frozen KD provider."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import threading
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

import torch

from reliable_endo_gs.actions.repair import Action
from reliable_endo_gs.config.intervention import InterventionConfig, load_intervention_config
from reliable_endo_gs.config.selector_study import load_selector_study_config
from reliable_endo_gs.data.scared_a00 import (
    ScaredA00Adapter,
    ScaredA00Dataset,
    ScaredA00Sample,
    load_a01_manifest,
)
from reliable_endo_gs.data.target_masks import TargetMaskIndex
from reliable_endo_gs.reconstruction import ReferenceConfig, load_reference_config
from reliable_endo_gs.reconstruction.checkpoint import _atomic_torch_save
from reliable_endo_gs.reconstruction.model import _freeze_injected_teacher
from reliable_endo_gs.reconstruction.student import StudentReconstructionModel
from reliable_endo_gs.regions.grid import grid_regions
from reliable_endo_gs.rendering.production import ProductionFieldRenderer
from reliable_endo_gs.routing.config import RoutingConfig, load_routing_config
from reliable_endo_gs.routing.cost import load_cost_calibration
from reliable_endo_gs.routing.masks import MaskProposalConfig, load_mask_config
from reliable_endo_gs.routing.support_grid import (
    AutomaticSupportGrid,
    SupportGridConfig,
    automatic_study_regions,
)
from reliable_endo_gs.routing.tokens import selector_feature_names
from reliable_endo_gs.routing.tradeoff import load_tradeoff_config
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.teachers import AlignedTeacherTarget, TargetCache, load_a01_roles
from reliable_endo_gs.training.harness import capture_rng, restore_rng
from reliable_endo_gs.training.sample_order import SeededSampleOrder
from reliable_endo_gs.training.student_checkpoint import (
    initialize_student_model,
    load_student_checkpoint,
    read_student_checkpoint,
)
from reliable_endo_gs.training.student_config import StudentLearningConfig, load_student_config
from reliable_endo_gs.training.student_learning import StudentLearner
from reliable_endo_gs.training.student_run import StudentSamples, run_student_training
from reliable_endo_gs.utils.io import write_json_atomic
from reliable_endo_gs.utils.seed import set_seed


def _check_zero_repair_render(repaired: torch.Tensor, original: torch.Tensor) -> dict[str, float]:
    """Bound FP32 quaternion roundtrip error in native RGB smoke comparisons.

    Gaussian composition converts quaternion -> matrix -> quaternion. A one-ULP
    change can amplify at rasterization boundaries. Bound both the worst RGB
    error (2e-4 on 0..1 RGB) and mean error (1e-7); field identity is checked
    separately. Neither threshold applies to training loss or geometry.
    """
    torch.testing.assert_close(repaired, original, rtol=0.0, atol=2e-4)
    difference = (repaired - original).abs()
    maximum, mean = float(difference.max()), float(difference.mean())
    if mean > 1e-7:
        raise ValueError(f"zero-repair mean RGB error exceeds 1e-7: {mean}")
    return {"max_abs": maximum, "mean_abs": mean}


def run_student_command(arguments: argparse.Namespace) -> dict[str, object]:
    """Own input-worker lifetime even when model or teacher initialization fails."""
    with ExitStack() as resources:
        return _run_student_command(arguments, resources)


def _run_student_command(arguments: argparse.Namespace, resources: ExitStack) -> dict[str, object]:
    """Native runs are explicit CLI requests; inference never resolves a teacher."""
    infer = arguments.command == "infer"
    smoke = arguments.command == "smoke"
    gpu_input = bool(getattr(arguments, "gpu_input", False))
    if gpu_input and infer:
        raise ValueError("--gpu-input is a training/smoke input-pipeline option")
    if gpu_input and torch.get_num_threads() != 1:
        raise ValueError("--gpu-input requires OMP_NUM_THREADS=1 to pin CPU reference RGB rounding")
    initialize_from = getattr(arguments, "initialize_model_from", None)
    metric_selector_path = getattr(arguments, "metric_selector", None)
    metric_config_path = getattr(arguments, "metric_config", None)
    metric_candidate_hash = getattr(arguments, "metric_candidate_hash", None)
    cost_calibration_path = getattr(arguments, "cost_calibration", None)
    deployment_artifacts = (
        metric_selector_path,
        metric_config_path,
        metric_candidate_hash,
        cost_calibration_path,
    )
    if not infer and any(value is not None for value in deployment_artifacts):
        raise ValueError("metric selector/cost calibration inputs are inference-only")
    metric_selector_inputs = (
        metric_selector_path,
        metric_config_path,
        metric_candidate_hash,
    )
    if any(value is not None for value in metric_selector_inputs) and not all(
        value is not None for value in metric_selector_inputs
    ):
        raise ValueError("--metric-selector requires --metric-config and --metric-candidate-hash")
    if metric_candidate_hash is not None and not str(metric_candidate_hash).strip():
        raise ValueError("--metric-candidate-hash must be nonempty")
    if initialize_from is not None and (infer or arguments.checkpoint is not None):
        raise ValueError("model initialization is a new run, incompatible with inference/resume")
    if infer and (arguments.checkpoint is None or arguments.sample_id is None):
        raise ValueError("infer requires --checkpoint and --sample-id")
    if infer and (arguments.cache_root is not None or arguments.teacher_factory is not None):
        raise ValueError("student-only inference does not accept teacher/cache inputs")
    if arguments.populate_cache and (arguments.command != "train" or arguments.cache_root is None):
        raise ValueError("--populate-cache requires train and --cache-root")
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
    if arguments.populate_cache:
        cache_root = arguments.cache_root.resolve()
        if cache_root == repo or repo in cache_root.parents:
            raise ValueError("populated KD cache must be outside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    device = torch.device("cuda:0")
    raw = read_student_checkpoint(arguments.checkpoint) if infer else None
    intervention_path = getattr(arguments, "intervention_config", None)
    intervention_config = (
        load_intervention_config(intervention_path) if intervention_path is not None else None
    )
    mask_path = getattr(arguments, "mask_config", None)
    mask_config = None if mask_path is None else load_mask_config(mask_path)
    mask_index_path = getattr(arguments, "target_mask_index", None)
    if raw is not None:
        recorded_mask = raw.get("mask_config")
        if recorded_mask is not None:
            pinned_mask = MaskProposalConfig(**recorded_mask)
            if mask_config is not None and mask_config != pinned_mask:
                raise ValueError("inference mask proposal config mismatch")
            mask_config = pinned_mask
        elif mask_config is not None:
            raise ValueError(
                "grid selector checkpoint cannot acquire mask routing; train a new mask selector"
            )
        if raw.get("intervention_config") is not None:
            recorded_intervention = InterventionConfig(**raw["intervention_config"])
            if intervention_config is not None and intervention_config != recorded_intervention:
                raise ValueError("inference intervention config mismatch")
            intervention_config = recorded_intervention
        elif intervention_config is not None:
            raise ValueError("legacy checkpoint cannot acquire a different selection policy")
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
        expected_joint = 1
        if (
            intervention_config is not None
            and intervention_config.schema == "intervention_selection.v2"
        ):
            expected_joint = 3 if intervention_config.normalization_mode == "burn_in" else 2
            if smoke and expected_joint == 3 and intervention_config.normalization_frames > 2:
                raise ValueError(
                    "bounded burn-in smoke needs normalization_frames <= 2; use the remediation smoke config"
                )
        if (
            intervention_config is not None
            and intervention_config.student_update_mode == "frozen"
            and initialize_from is None
            and arguments.checkpoint is None
        ):
            raise ValueError(
                "frozen training requires --initialize-model-from a Phase I checkpoint"
            )
        if smoke and (config.student_epochs not in (0, 1) or config.joint_epochs != expected_joint):
            raise ValueError(
                f"bounded smoke requires zero or one student and {expected_joint} joint updates"
            )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError(
            "real student rendering requires one visible GPU; use CPU synthetic tests locally"
        )
    if construction.runtime.device != "cuda:0" or construction.data_role != "final_train":
        raise ValueError("native student construction must pin cuda:0 and final_train")
    if construction.runtime.precision != "fp32":
        raise ValueError("student runtime/checkpoint must declare FP32 precision")
    automatic_regions = None
    study_path = getattr(arguments, "study_protocol_path", None)
    study = None if study_path is None else load_selector_study_config(study_path)
    if study is not None:
        if mask_path is not None or mask_index_path is not None:
            raise ValueError("automatic study regions cannot consume operator mask arguments")
        if intervention_config is None or intervention_config.student_update_mode != "frozen":
            raise ValueError(
                "automatic study training requires the explicit frozen-student protocol"
            )
        study_mask, automatic_regions = automatic_study_regions(study)
        if raw is not None and (
            raw.get("automatic_support_grid") != asdict(automatic_regions.config)
            or mask_config != study_mask
            or raw.get("identity", {}).get("study_config_hash") != study.config_hash
        ):
            raise ValueError("automatic study inference protocol differs from its checkpoint")
        mask_config = study_mask
    elif raw is not None and raw.get("automatic_support_grid") is not None:
        if mask_config is None or mask_index_path is not None:
            raise ValueError(
                "automatic checkpoint pins its mask limits and excludes operator masks"
            )
        automatic_regions = AutomaticSupportGrid(
            SupportGridConfig(**raw["automatic_support_grid"]), mask_config
        )
    if automatic_regions is None and (mask_config is None) != (mask_index_path is None):
        raise ValueError("mask protocol requires --target-mask-index and proposal config")
    target_masks = automatic_regions
    if mask_index_path is not None:
        mask_index_path = mask_index_path.resolve()
        if mask_index_path == repo or repo in mask_index_path.parents:
            raise ValueError(
                "medical target-mask index and arrays must stay outside the repository"
            )
        target_masks = TargetMaskIndex(mask_index_path)  # type: ignore[assignment]
    roles = load_a01_roles(arguments.roles)
    manifest = load_a01_manifest(arguments.manifest, roles=roles)
    adapter = ScaredA00Adapter(arguments.data_root, manifest, roles=roles)
    dataset = ScaredA00Dataset(
        adapter,
        include_supervision=False,
        data_role="final_train",
        output_hw=construction.runtime.resolution_hw,
        device=device if gpu_input else None,
    )
    staged = None
    stage_from = getattr(arguments, "stage_from", None)
    stage_inventory = getattr(arguments, "stage_inventory", None)
    if (stage_from is None) != (stage_inventory is None):
        raise ValueError("--stage-from and --stage-inventory must be supplied together")
    if stage_from is not None:
        if infer or smoke or config.joint_epochs or intervention_config is not None:
            raise ValueError(
                "asynchronous staging currently supports ordered student-only training"
            )
        from reliable_endo_gs.data.staging import FileStager, deployment_file_index
        from reliable_endo_gs.training.input_prefetch import OrderedInputPrefetch

        member_index = deployment_file_index(
            [entry.sample_id for entry in manifest.entries if entry.data_role == "final_train"],
            stage_inventory,  # type: ignore[arg-type]
        )
        event_lock = threading.Lock()
        event_path = Path(str(output) + ".input.jsonl")

        def input_event(event: str, **fields: object) -> None:
            record = {"event": event, "time_unix": time.time(), **fields}
            with event_lock:
                event_path.parent.mkdir(parents=True, exist_ok=True)
                with event_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")

        stager = FileStager(
            stage_from,
            arguments.data_root,
            journal=Path(str(output) + ".staging.jsonl"),
            previous_journals=arguments.staging_journal,
            retries=arguments.staging_retries,
            event=input_event,
        )
        start = 0
        if arguments.checkpoint is not None:
            start = read_student_checkpoint(arguments.checkpoint)["cursor"]["next_sample_index"]
        staged = resources.enter_context(
            OrderedInputPrefetch(
                dataset,
                lambda index: stager.prepare(member_index[index]),
                start=start,
                ahead=arguments.prefetch_samples,
                workers=arguments.staging_workers,
                release=lambda index: stager.release(member_index[index]),
                event=input_event,
            )
        )
        input_event(
            "prefetch_started",
            start_index=start,
            ahead=staged.ahead,
            workers=staged.workers,
            tensor_loader="unchanged_main_thread",
        )
    if target_masks is not None:
        if construction.runtime.resolution_hw is None:
            raise ValueError("mask routing requires an explicit model output resolution")
        required = (
            [arguments.sample_id]
            if infer or smoke
            else [entry.sample_id for entry in manifest.entries if entry.data_role == "final_train"]
        )
        target_masks.require_samples(required, construction.runtime.resolution_hw)
    set_seed(config.seed)
    model = StudentReconstructionModel(
        renderer=ProductionFieldRenderer(),
        config=construction,
        width=config.width,
        bins=config.bins,
        minimum_effective_px=config.minimum_effective_px,
        stereo_architecture=config.stereo_architecture,
    ).to(device)
    learner = StudentLearner(
        model,
        config,
        routing,
        intervention_config=intervention_config,
        mask_config=mask_config,
        target_masks=target_masks,
    )
    digest = hashlib.sha256()
    for path in sorted((repo / "src/reliable_endo_gs").rglob("*.py")):
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    identity: dict[str, object] = {
        "source_sha256": digest.hexdigest(),
        "manifest_hash": manifest.manifest_hash,
        "role_hash": roles.role_hash,
        "renderer_identity": model._renderer_identity(),
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
    if gpu_input:
        identity["input_pipeline"] = {
            "schema": "cuda_a00_input.v2",
            "tensor_device": "cuda:0",
            "rgb_normalization": "cuda_uint8_lookup_round_once_float32",
            "rgb_resize": "cuda_cpu_single_thread_order_bilinear_identity_or_twofold",
            "rgb_reference_cpu_threads": 1,
            "supervision": "same_canonical_nearest_exact_without_rgb_reload",
        }
    if intervention_config is not None:
        identity["intervention_config_hash"] = intervention_config.config_hash
        identity["selection_protocol"] = intervention_config.schema
        if intervention_config.schema == "intervention_selection.v2":
            identity["feature_schema"] = list(selector_feature_names(intervention_config))
            identity["normalization_protocol"] = "train_median_IQR_and_common_action_gain_q75.v1"
    if mask_config is not None:
        assert target_masks is not None
        identity.update(
            region_protocol=mask_config.schema,
            mask_config_hash=mask_config.config_hash,
            target_mask_index_sha256=target_masks.identity,
            matching_evidence_protocol="student_matching_shared_pass.v1",
        )
        if automatic_regions is not None:
            identity["automatic_support_grid"] = asdict(automatic_regions.config)
            identity["region_protocol"] = automatic_regions.config.schema
            if study is not None:
                identity["study_config_hash"] = study.config_hash
            elif raw is not None and "study_config_hash" in raw["identity"]:
                identity["study_config_hash"] = raw["identity"]["study_config_hash"]
    if initialize_from is not None:
        identity["model_initialization"] = initialize_student_model(
            initialize_from, learner, require_trained=learner.student_frozen
        )
    if infer:
        assert arguments.checkpoint is not None
        load_student_checkpoint(arguments.checkpoint, learner, identity, len(dataset), resume=False)
        if cost_calibration_path is not None:
            calibration = load_cost_calibration(cost_calibration_path)
            learner.attach_cost_calibration(calibration)
        if metric_selector_path is not None:
            assert metric_config_path is not None and metric_candidate_hash is not None
            metric_config = load_tradeoff_config(metric_config_path)
            learner.load_metric_selector(
                metric_selector_path,
                metric_config,
                str(metric_candidate_hash),
            )
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
                collect_one_target,
                infer_and_align,
                load_teacher_factory,
                target_cache_key,
            )

            teacher = load_teacher_factory(
                arguments.teacher_factory
                or "reliable_endo_gs.teachers.external:build_foundation_stereo"
            )
            identity["teacher_identity"] = teacher.identity.to_dict()
            identity["target_mode"] = (
                "cache_populate"
                if arguments.populate_cache
                else "cache_readonly"
                if arguments.cache_root
                else "live"
            )
            cache = (
                TargetCache(
                    arguments.cache_root, roles=roles, read_only=not arguments.populate_cache
                )
                if arguments.cache_root
                else None
            )
            # Read-only cached supervision requires only the pinned identity descriptor.
            # LazyTeacherAdapter.freeze() loads weights, so call it only when
            # live inference is requested. Population loads on an explicitly
            # authorized miss; read-only misses never trigger inference.
            if cache is None:
                _freeze_injected_teacher(teacher)

            def provide(sample: ScaredA00Sample) -> AlignedTeacherTarget:
                if cache is not None:
                    key = target_cache_key(
                        sample, teacher, split_identity=construction.split_identity, roles=roles
                    )
                    target = cache.get(key)
                    if target is None:
                        if arguments.populate_cache:
                            rng = capture_rng(device)
                            try:
                                _freeze_injected_teacher(teacher)
                                return collect_one_target(
                                    sample,
                                    teacher,
                                    cache,
                                    split_identity=construction.split_identity,
                                    roles=roles,
                                    device=device,
                                ).target
                            finally:
                                restore_rng(rng, device)
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
        gt_provider = None
        if config.gt_weight:
            from reliable_endo_gs.training.distillation import GroundTruthDisparity

            identity["gt_disparity_supervision"] = "a00_supervision_smooth_l1.v1"

            def provide_gt(sample: ScaredA00Sample) -> GroundTruthDisparity:
                if gpu_input:
                    supervision = adapter.load_supervision(sample)
                else:
                    supervision = adapter.load(  # type: ignore[assignment]
                        sample.sample_id,
                        data_role="final_train",
                        include_supervision=True,
                        output_hw=construction.runtime.resolution_hw,
                    ).supervision
                assert supervision is not None
                return GroundTruthDisparity(
                    supervision.disparity.to(device),
                    supervision.valid_mask.to(device),
                    sample.batch.support.to(device),
                )

            gt_provider = provide_gt
        metric_truth_provider = None
        if intervention_config is not None and intervention_config.metric_quality is not None:
            from reliable_endo_gs.training.metric_quality import OfflineMetricTruth

            def provide_metric_truth(sample: ScaredA00Sample) -> OfflineMetricTruth:
                if gpu_input:
                    supervision = adapter.load_supervision(sample)
                    return OfflineMetricTruth.from_a00(replace(sample, supervision=supervision))
                truth_sample = adapter.load(
                    sample.sample_id,
                    data_role="final_train",
                    include_supervision=True,
                    output_hw=construction.runtime.resolution_hw,
                )
                return OfflineMetricTruth.from_a00(truth_sample)

            metric_truth_provider = provide_metric_truth
        samples: StudentSamples = staged if staged is not None else dataset
        if intervention_config is not None and not smoke:
            ordered = SeededSampleOrder(dataset, config.seed)
            samples = ordered
            identity["sample_order"] = "fixed_seeded_permutation.v1"
            identity["sample_order_hash"] = ordered.order_hash
        if smoke:
            sample = adapter.load(
                arguments.sample_id,
                data_role="final_train",
                include_supervision=False,
                output_hw=construction.runtime.resolution_hw,
                device=device if gpu_input else None,
            )
            from reliable_endo_gs.training.student_learning import evaluation_mode

            zero_repair_render_errors: dict[str, dict[str, float]] = {}
            with evaluation_mode(model, learner.actions):
                state = learner.state(sample)
                height, width = state.batch.left.shape[-2:]
                region = grid_regions((height, width), (height, width), routing.halo)[0]
                if mask_config is not None:
                    candidates = learner.candidate_regions(state, sample.sample_id)
                    if not candidates:
                        raise ValueError(
                            "mask smoke needs one eligible target candidate; change sample or audit proposal thresholds"
                        )
                    region = candidates[0]
                for action in (Action.STEREO, Action.GAUSSIAN):
                    outcome = learner.actions.execute(state, region, action, model)
                    exact = intervention_config is not None and action == Action.GAUSSIAN
                    for name in (
                        "means3d",
                        "colors",
                        "scales",
                        "rotations",
                        "opacities",
                        "valid_mask",
                        "cov_surface",
                    ):
                        torch.testing.assert_close(
                            getattr(outcome.field, name),
                            getattr(state.base.gaussians, name),
                            rtol=0.0 if exact else 1e-5,
                            atol=0.0 if exact else 1e-7,
                        )
                    for view, repaired, original in (
                        ("left", outcome.left, state.base.render_left),
                        ("right", outcome.right, state.base.render_right),
                    ):
                        assert original is not None
                        try:
                            errors = _check_zero_repair_render(repaired.image, original.image)
                        except (AssertionError, ValueError) as error:
                            raise ValueError(
                                f"zero {action.name.lower()} repair, {view} render: {error}"
                            ) from error
                        zero_repair_render_errors[f"{action.name.lower()}_{view}"] = errors
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
                    key: value.detach().cpu().clone() for key, value in module.named_parameters()
                }
                for name, module in smoke_modules.items()
            }
            before_model_state = copy.deepcopy(model.state_dict())
            before_model_adam = copy.deepcopy(learner.model_optimizer.state_dict())
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
        report = run_student_training(
            smoke_samples if smoke else samples,
            learner,
            identity,
            output,
            max_steps=config.student_epochs + config.joint_epochs if smoke else arguments.max_steps,
            resume=arguments.checkpoint,
            target_provider=target_provider,
            metric_truth_provider=metric_truth_provider,
            gt_provider=gt_provider,
        )
        if staged is not None:
            io_report = {
                "schema": "ordered_byte_prefetch.v1",
                "source_root": str(stage_from),
                "inventory": str(stage_inventory),
                "prefetch_samples": staged.ahead,
                "workers": staged.workers,
                "consumed": staged.consumed,
                "input_wait_seconds": staged.wait_seconds,
                "peak_pending": staged.peak_pending,
                "model_reloads_at_queue_boundaries": 0,
            }
            write_json_atomic(output / "input_pipeline.json", io_report)
        if smoke:
            torch.cuda.synchronize(device)
            report["smoke_only_not_full_training"] = True
            report["elapsed_training_seconds"] = time.perf_counter() - started
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            changed = {
                name: any(
                    not torch.equal(value.detach().cpu(), before[name][key])
                    for key, value in module.named_parameters()
                )
                for name, module in smoke_modules.items()
            }
            expected_changes = {
                name: not learner.student_frozen if name in ("student", "head") else True
                for name in smoke_modules
            }
            model_state_unchanged = _state_equal(before_model_state, model.state_dict())
            model_adam_unchanged = _state_equal(
                before_model_adam, learner.model_optimizer.state_dict()
            )
            if changed != expected_changes or (
                learner.student_frozen and not (model_state_unchanged and model_adam_unchanged)
            ):
                raise ValueError(
                    "smoke parameter/Adam ownership mismatch: "
                    f"changed={changed}, expected={expected_changes}, "
                    f"model_state_unchanged={model_state_unchanged}, "
                    f"model_adam_unchanged={model_adam_unchanged}"
                )
            report["parameters_changed"] = changed
            report["model_state_unchanged"] = model_state_unchanged
            report["model_adam_unchanged"] = model_adam_unchanged
            report["zero_repair_render_errors"] = zero_repair_render_errors
    write_json_atomic(output / "result.json", report)
    return report


def _state_equal(first: object, second: object) -> bool:
    """Exact nested tensor equality for smoke ownership checks, including Adam."""
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(_state_equal(value, second[key]) for key, value in first.items())
        )
    if isinstance(first, (list, tuple)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(_state_equal(a, b) for a, b in zip(first, second, strict=True))
        )
    return first == second
