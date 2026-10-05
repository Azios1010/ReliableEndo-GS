"""Thin experimental Phase II entrypoint: smoke, train, or selected inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, load_a01_manifest
from reliable_endo_gs.reconstruction.config import ReferenceConfig
from reliable_endo_gs.rendering.production import ProductionFieldRenderer
from reliable_endo_gs.routing.config import load_routing_config
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.teachers import TargetCache, load_a01_roles
from reliable_endo_gs.teachers.workflow import load_teacher_factory
from reliable_endo_gs.training.a04_reference import A04ReferenceTrainer
from reliable_endo_gs.training.action_checkpoint import load_action_checkpoint
from reliable_endo_gs.training.action_learning import ActionLearner
from reliable_endo_gs.training.action_run import (
    frozen_state,
    known_skip_reason,
    run_action_training,
)
from reliable_endo_gs.utils.io import write_json_atomic
from reliable_endo_gs.utils.seed import set_seed


def sha256_file(path: Path) -> str:
    """Hash an exact external artifact in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    """Expose explicit roots and checkpoint identities, without guessed paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "train", "infer"))
    parser.add_argument("--config", type=Path, default=Path("configs/routing/frozen_a04.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, default=Path("configs/teachers/a01_scared.yaml"))
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--teacher-factory", default="reliable_endo_gs.teachers.external:build_foundation_stereo"
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Phase II checkpoint to resume/load")
    parser.add_argument(
        "--sample-id",
        help="Required for smoke/infer; train uses the unchanged final_train manifest",
    )
    parser.add_argument("--data-role", choices=("final_train", "final_test"), default="final_train")
    parser.add_argument(
        "--max-steps", type=int, help="Bound consumed samples across the two training stages"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New external artifact directory"
    )
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    """Resolve pinned A04, then delegate all scientific work to first-party modules."""
    if arguments.command == "infer" and arguments.checkpoint is None:
        raise ValueError("infer requires a learned Phase II checkpoint")
    if arguments.command == "smoke" and arguments.checkpoint is not None:
        raise ValueError("smoke checks fresh zero repairs; use infer for a learned checkpoint")
    if arguments.command != "train" and arguments.sample_id is None:
        raise ValueError("smoke/infer require an explicit --sample-id")
    if arguments.command != "infer" and arguments.data_role != "final_train":
        raise ValueError("learning/smoke cannot consume final_test")
    if arguments.command == "train" and arguments.sample_id is not None:
        raise ValueError("train uses the role manifest; use --max-steps to bound the run")
    repo = Path(__file__).resolve().parents[3]
    output = arguments.output_dir.resolve()
    if output == repo or repo in output.parents:
        raise ValueError("Phase II artifacts must be written outside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    config = load_routing_config(arguments.config)
    if not torch.cuda.is_available():
        raise ValueError("real A04 Phase II execution requires CUDA; use synthetic CPU tests")
    if torch.cuda.device_count() != 1:
        raise ValueError(
            "mask one physical GPU with CUDA_VISIBLE_DEVICES before invoking native rendering"
        )
    payload: Any = torch.load(arguments.reference_checkpoint, map_location="cpu", weights_only=True)
    reference_config = ReferenceConfig.from_dict(payload["config"])
    if reference_config.runtime.device != "cuda:0" or reference_config.data_role != "final_train":
        raise ValueError("reference must pin logical cuda:0 and original final_train role")
    reference_sha = sha256_file(arguments.reference_checkpoint)
    roles = load_a01_roles(arguments.roles)
    manifest = load_a01_manifest(arguments.manifest, roles=roles)
    adapter = ScaredA00Adapter(arguments.data_root, manifest, roles=roles)
    teacher = load_teacher_factory(arguments.teacher_factory)
    source = A04ReferenceTrainer(
        config=reference_config,
        adapter=adapter,
        teacher=teacher,
        renderer=ProductionFieldRenderer(),
        cache=None
        if arguments.cache_root is None
        else TargetCache(arguments.cache_root, roles=roles),
    )
    source.resume(arguments.reference_checkpoint, restore_training=False)
    set_seed(config.seed)
    learner = ActionLearner(source.model, config)
    reference_before = {
        name: value.detach().cpu().clone() for name, value in source.model.state_dict().items()
    }
    digest = hashlib.sha256()
    for path in sorted((repo / "src/reliable_endo_gs").rglob("*.py")):
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    identity: dict[str, object] = {
        "reference_sha256": reference_sha,
        "reference_config_hash": source.config.config_hash,
        "teacher_identity": dict(source.model._teacher_identity()),
        "manifest_hash": manifest.manifest_hash,
        "role_hash": roles.role_hash,
        "source_sha256": digest.hexdigest(),
        "git_commit": get_git_commit(repo),
        "git_dirty": is_git_dirty(repo),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
        "renderer_identity": source.config.renderer_identity,
        "seed": config.seed,
        "samples_per_epoch": len(source._dataset),
    }
    metadata = {
        "identity": identity,
        "config": asdict(config),
        "reference_checkpoint": str(arguments.reference_checkpoint.resolve()),
        "data_root": str(arguments.data_root.resolve()),
        "command": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "experimental_gate_status": "user_authorized_extension_not_gate_certification",
        "selector_parameters": sum(p.numel() for p in learner.selector.parameters()),
        "repair_parameters": sum(p.numel() for p in learner.actions.parameters()),
    }
    if arguments.command == "train":
        result = run_action_training(
            source,
            learner,
            identity,
            output,
            max_steps=arguments.max_steps,
            resume=arguments.checkpoint,
            run_metadata=metadata,
        )
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json_atomic(output / "run.json", metadata)
        if arguments.checkpoint is not None:
            load_action_checkpoint(arguments.checkpoint, learner, identity, len(source._dataset))
        sample = adapter.load(
            arguments.sample_id,
            data_role=arguments.data_role,
            include_supervision=False,
            output_hw=source.config.runtime.resolution_hw,
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        state = frozen_state(source, sample)
        torch.cuda.synchronize()
        base_ms = (time.perf_counter() - started) * 1000
        skip = known_skip_reason(state, source)
        if skip is not None:
            raise ValueError(f"explicit smoke/inference sample is ineligible: {skip}")
        result = {"sample_id": sample.sample_id, "phase1_frozen": True}
        result["base_inference_ms_including_teacher"] = base_ms
        if arguments.command == "smoke":
            result["zero_repair_max_rgb_errors"] = learner.check_zero_repairs(state)
            result["action_learning"] = learner.warmup_step(state)
            result["selector_learning"] = learner.selector_step(state, 0)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result["selected_inference"] = learner.infer(state)
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000
        result["phase2_inference_ms_including_features_selection_render_assessment"] = inference_ms
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    if sha256_file(arguments.reference_checkpoint) != reference_sha:
        raise ValueError("external reference checkpoint changed during the run")
    if not all(
        torch.equal(value, source.model.state_dict()[name].detach().cpu())
        for name, value in reference_before.items()
    ):
        raise ValueError("in-memory Phase I weights changed during Phase II learning")
    result["phase1_parameters_unchanged"] = True
    write_json_atomic(output / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Return a nonzero status on explicit preflight or runtime failures."""
    arguments = build_parser().parse_args(argv)
    try:
        result = run(arguments)
    except (OSError, ValueError, RuntimeError, KeyError, AssertionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
