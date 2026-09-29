"""Minimal command-line interface for infrastructure validation."""

import argparse
import json
from dataclasses import asdict, replace
from collections.abc import Sequence
from pathlib import Path

from reliable_endo_gs.config import ConfigError, config_to_dict, hash_config, load_config
from reliable_endo_gs.data.paths import resolve_data_root
from reliable_endo_gs.data.registry import get_dataset_adapter
from reliable_endo_gs.data.scared_a00 import (
    ScaredA00Adapter,
    ScaredA00SampleError,
    build_a01_manifest,
)
from reliable_endo_gs.reconstruction import load_reference_config
from reliable_endo_gs.rendering import ProductionFieldRenderer
from reliable_endo_gs.rendering.cross_view import ReferenceRenderer
from reliable_endo_gs.rendering.field import ExistingRendererBridge
from reliable_endo_gs.data.schema import load_dataset_config
from reliable_endo_gs.data.validation import DatasetValidationReport
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.runtime.logging import configure_logging
from reliable_endo_gs.runtime.manifest import (
    create_run_id,
    create_run_manifest,
    utc_timestamp,
    write_run_manifest,
)
from reliable_endo_gs.utils.io import create_unique_directory, write_yaml_atomic
from reliable_endo_gs.utils.seed import set_seed
from reliable_endo_gs.teachers import TargetCache, load_a01_roles
from reliable_endo_gs.teachers.workflow import (
    A01WorkflowError,
    collect_targets,
    infer_and_align,
    load_teacher_factory,
    target_cache_key,
    validate_cache,
)
from reliable_endo_gs.training.a04_reference import A04ReferenceTrainer, A04TrainingError


def build_parser() -> argparse.ArgumentParser:
    """Build the ReliableEndo-GS command-line parser."""

    parser = argparse.ArgumentParser(prog="reg", description="ReliableEndo-GS utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser(
        "smoke", description="Validate the development infrastructure"
    )
    smoke_parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config")

    data_parser = subparsers.add_parser(
        "data", description="Resolve and validate externally mounted datasets"
    )
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)

    data_resolve_parser = data_subparsers.add_parser(
        "resolve", description="Resolve a portable dataset root"
    )
    data_resolve_parser.add_argument(
        "--config", required=True, type=Path, help="Path to a dataset YAML config"
    )
    data_resolve_parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Explicit absolute global data root overriding RELIABLE_ENDO_DATA_ROOT",
    )

    data_validate_parser = data_subparsers.add_parser(
        "validate", description="Validate an explicitly resolved dataset root"
    )
    data_validate_parser.add_argument(
        "--config", required=True, type=Path, help="Path to a dataset YAML config"
    )

    data_index_parser = data_subparsers.add_parser(
        "a01-index", description="Build an explicit active SCARED dataset_1..dataset_7 sample manifest"
    )
    data_index_parser.add_argument("--data-root", required=True, type=Path)
    data_index_parser.add_argument("--output", required=True, type=Path)
    fold_policy = data_index_parser.add_mutually_exclusive_group()
    fold_policy.add_argument(
        "--require-all-folds",
        dest="require_all_folds",
        action="store_true",
        help="Require every active SCARED fold 1-7 (the default)",
    )
    fold_policy.add_argument(
        "--allow-incomplete",
        dest="require_all_folds",
        action="store_false",
        help="Allow partial manifests for local fixtures only",
    )
    data_index_parser.set_defaults(require_all_folds=True)
    data_index_parser.add_argument(
        "--roles", type=Path, default=Path("configs/teachers/a01_scared.yaml")
    )

    a01_parser = subparsers.add_parser("a01", description="A01 frozen teacher and target-cache commands")
    a01_subparsers = a01_parser.add_subparsers(dest="a01_command", required=True)
    for name, description in (
        ("teacher-smoke", "Run one explicit injected teacher and A00 alignment"),
        ("cache", "Generate/resume A01 targets from SCARED through a frozen teacher"),
        ("cache-validate", "Validate every A01 cache item and manifest checksum"),
    ):
        command_parser = a01_subparsers.add_parser(name, description=description)
        command_parser.add_argument("--cache-root", type=Path, default=Path("artifacts/targets/a01"))
        command_parser.add_argument("--data-root", type=Path)
        command_parser.add_argument("--manifest", type=Path)
        command_parser.add_argument("--teacher-factory", help="Injected module:function Teacher factory")
        command_parser.add_argument("--roles", type=Path, default=Path("configs/teachers/a01_scared.yaml"))
        command_parser.add_argument("--split-identity", default="a01-scared-dataset1-7-v1")
        command_parser.add_argument(
            "--device",
            default="cpu",
            help="Explicit teacher target device; use cuda:0 on the qualification server",
        )
        command_parser.add_argument(
            "--resolution-hw",
            nargs=2,
            type=int,
            metavar=("HEIGHT", "WIDTH"),
            help="Explicit A00 output grid; match the A04 runtime resolution",
        )
    a01_subparsers.choices["cache-validate"].set_defaults(cache_root=Path("artifacts/targets/a01"))

    a04_parser = subparsers.add_parser("a04", description="A04 bounded reference smoke/train/eval/profile commands")
    a04_subparsers = a04_parser.add_subparsers(dest="a04_command", required=True)
    for name, description in (
        ("smoke", "Run one bounded A04 batch"),
        ("train", "Train the independent A04 head for bounded steps"),
        ("eval", "Evaluate a frozen A04 checkpoint"),
        ("profile", "Profile one bounded A04 batch"),
    ):
        command_parser = a04_subparsers.add_parser(name, description=description)
        command_parser.add_argument("--config", type=Path, default=Path("configs/reconstruction/a04_reference.yaml"))
        command_parser.add_argument("--data-root", required=True, type=Path)
        command_parser.add_argument("--manifest", required=True, type=Path)
        command_parser.add_argument("--roles", type=Path, default=Path("configs/teachers/a01_scared.yaml"))
        command_parser.add_argument("--cache-root", type=Path, default=Path("artifacts/targets/a01"))
        command_parser.add_argument("--teacher-factory", help="Injected module:function Teacher factory")
        command_parser.add_argument("--renderer", choices=("reference", "native"), default="reference")
        command_parser.add_argument(
            "--data-role",
            choices=("teacher_fit", "qualification"),
            help="Override the role selected by the resolved A04 config",
        )
        command_parser.add_argument("--device", default=None)
        command_parser.add_argument("--checkpoint", type=Path)
        command_parser.add_argument("--output-checkpoint", type=Path)
        command_parser.add_argument("--epochs", type=int, default=1)
        command_parser.add_argument("--max-steps", type=int)
        command_parser.add_argument("--warmup", type=int, default=0)
        command_parser.add_argument("--repetitions", type=int, default=1)
    data_validate_parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Explicit absolute global data root overriding RELIABLE_ENDO_DATA_ROOT",
    )
    return parser


def run_smoke(config_path: Path) -> Path:
    """Execute the infrastructure-only smoke workflow and return its run directory."""

    config = load_config(config_path)
    logger = configure_logging()
    logger.info("Loaded configuration from %s", config_path)

    set_seed(config.experiment.seed)
    config_hash = hash_config(config)
    repo_root = Path.cwd()
    project_git_commit = get_git_commit(repo_root)
    git_dirty = is_git_dirty(repo_root)

    timestamp = utc_timestamp()
    base_run_id = create_run_id(config.experiment.name, config_hash, timestamp)
    run_directory = create_unique_directory(config.experiment.output_root, base_run_id)

    manifest = create_run_manifest(
        run_id=run_directory.name,
        timestamp_utc=timestamp,
        project_git_commit=project_git_commit,
        git_dirty=git_dirty,
        config_hash=config_hash,
        seed=config.experiment.seed,
    )
    write_yaml_atomic(run_directory / "resolved_config.yaml", config_to_dict(config))
    write_run_manifest(run_directory / "run_manifest.json", manifest)

    logger.info("Created infrastructure smoke run at %s", run_directory)
    return run_directory


def run_data_resolve(config_path: Path, global_root: Path | None = None) -> Path:
    """Resolve a configured external dataset root without inspecting it."""

    config = load_dataset_config(config_path)
    get_dataset_adapter(config.name)
    resolved_root = resolve_data_root(config.root, global_root=global_root)
    logger = configure_logging()
    logger.info("Resolved dataset %s root to %s", config.name, resolved_root)
    return resolved_root


def run_data_validate(
    config_path: Path, global_root: Path | None = None
) -> DatasetValidationReport:
    """Resolve and explicitly validate a configured external dataset root."""

    config = load_dataset_config(config_path)
    adapter = get_dataset_adapter(config.name)
    resolved_root = resolve_data_root(config.root, global_root=global_root)
    report = adapter.validate_root(resolved_root)
    logger = configure_logging()
    for message in report.messages:
        logger.info("%s: %s", config.name, message)
    return report


def run_a01_index(
    data_root: Path, output: Path, *, roles_path: Path, require_all_folds: bool
) -> dict[str, object]:
    roles = load_a01_roles(roles_path)
    manifest = build_a01_manifest(
        data_root,
        output,
        roles=roles,
        require_all_folds=require_all_folds,
    )
    return {
        "manifest": output.as_posix(),
        "entries": len(manifest.entries),
        "role_contract_hash": manifest.role_contract_hash,
    }


def _load_a01_adapter(data_root: Path, manifest_path: Path, roles_path: Path) -> tuple[ScaredA00Adapter, object]:
    roles = load_a01_roles(roles_path)
    # Reuse the explicit role object for cache and adapter validation while
    # keeping all sample path resolution inside the first-party adapter.
    from reliable_endo_gs.data.scared_a00 import load_a01_manifest

    manifest = load_a01_manifest(manifest_path, roles=roles)
    return ScaredA00Adapter(data_root, manifest, roles=roles), roles


def _load_cli_teacher(factory_spec: str | None):
    if not factory_spec:
        raise A01WorkflowError(
            "an explicit --teacher-factory module:function is required; no external teacher is bundled"
        )
    return load_teacher_factory(factory_spec)


def _run_a01(arguments: argparse.Namespace) -> dict[str, object]:
    roles = load_a01_roles(arguments.roles)
    if arguments.a01_command == "cache-validate":
        if arguments.manifest is None or arguments.data_root is None or arguments.teacher_factory is None:
            raise A01WorkflowError(
                "cache validation requires --manifest, --data-root and --teacher-factory "
                "to verify expected samples and teacher-bound cache keys"
            )
        cache = TargetCache(arguments.cache_root, roles=roles)
        expected_sample_ids = None
        expected_key_digests = None
        from reliable_endo_gs.data.scared_a00 import load_a01_manifest

        expected_sample_ids = load_a01_manifest(arguments.manifest, roles=roles).sample_ids
        adapter, _ = _load_a01_adapter(arguments.data_root, arguments.manifest, arguments.roles)
        teacher = _load_cli_teacher(arguments.teacher_factory)
        output_hw = None if arguments.resolution_hw is None else tuple(arguments.resolution_hw)
        expected_key_digests = [
            target_cache_key(
                sample,
                teacher,
                split_identity=arguments.split_identity,
                roles=roles,
            ).digest
            for sample in adapter.iter_samples(include_supervision=False, output_hw=output_hw)
        ]
        report = validate_cache(
            cache,
            expected_sample_ids=expected_sample_ids,
            expected_key_digests=expected_key_digests,
        )
        if (
            report["stale_or_missing"]
            or report["missing_expected"]
            or report["missing_expected_keys"]
        ):
            raise A01WorkflowError(
                "A01 cache validation failed: "
                f"stale_or_missing={report['stale_or_missing']}, "
                f"missing_expected={report['missing_expected']}, "
                f"missing_expected_keys={report['missing_expected_keys']}"
            )
        return report
    if arguments.data_root is None or arguments.manifest is None:
        raise A01WorkflowError("--data-root and --manifest are required for A01 teacher/cache commands")
    adapter, _ = _load_a01_adapter(arguments.data_root, arguments.manifest, arguments.roles)
    teacher = _load_cli_teacher(arguments.teacher_factory)
    if arguments.a01_command == "teacher-smoke":
        output_hw = None if arguments.resolution_hw is None else tuple(arguments.resolution_hw)
        sample = next(adapter.iter_samples(include_supervision=False, output_hw=output_hw), None)
        if sample is None:
            raise A01WorkflowError("active A01 manifest is empty")
        target = infer_and_align(sample, teacher, device=arguments.device)
        return {
            "sample_id": sample.sample_id,
            "target_shape": list(target.disparity.shape),
            "target_valid_count": int(target.validity.sum().item()),
            "teacher_identity_hash": teacher.identity.identity_hash,
        }
    cache = TargetCache(arguments.cache_root, roles=roles)
    report = collect_targets(
        adapter,
        teacher,
        cache,
        split_identity=arguments.split_identity,
        roles=roles,
        device=arguments.device,
        output_hw=(
            None if arguments.resolution_hw is None else tuple(arguments.resolution_hw)
        ),
    )
    return {
        "processed": report.processed,
        "cache_hits": report.cache_hits,
        "computed": report.computed,
        "failed": list(report.failed),
        "teacher_identity_hash": report.teacher_identity_hash,
        "cache_root": report.cache_root.as_posix(),
    }


def _run_a04(arguments: argparse.Namespace) -> dict[str, object]:
    roles = load_a01_roles(arguments.roles)
    config = load_reference_config(arguments.config)
    if arguments.device is not None:
        config = replace(config, runtime=replace(config.runtime, device=arguments.device))
    if arguments.data_role is not None:
        config = replace(config, data_role=arguments.data_role)
    adapter, _ = _load_a01_adapter(arguments.data_root, arguments.manifest, arguments.roles)
    teacher = load_teacher_factory(arguments.teacher_factory) if arguments.teacher_factory else None
    cache = TargetCache(arguments.cache_root, roles=roles) if arguments.cache_root.exists() else None
    if arguments.renderer == "native":
        renderer = ProductionFieldRenderer()
    else:
        renderer = ExistingRendererBridge(ReferenceRenderer())
    trainer = A04ReferenceTrainer(
        config=config,
        adapter=adapter,
        teacher=teacher,
        renderer=renderer,
        cache=cache,
        device=config.runtime.device,
    )
    if arguments.a04_command in ("eval", "profile") and arguments.checkpoint is None:
        raise A04TrainingError(
            f"a04 {arguments.a04_command} requires --checkpoint; use a04 smoke for an untrained forward"
        )
    if arguments.checkpoint is not None and arguments.a04_command in ("eval", "profile", "train"):
        if not arguments.checkpoint.is_file():
            raise A04TrainingError(f"checkpoint does not exist: {arguments.checkpoint}")
        trainer.resume(arguments.checkpoint)
    if arguments.a04_command == "smoke":
        result = trainer.smoke()
        return {"completed_steps": result.completed_steps, "steps": [asdict(step) for step in result.steps]}
    if arguments.a04_command == "train":
        result = trainer.train(epochs=arguments.epochs, max_steps=arguments.max_steps)
        destination = arguments.output_checkpoint
        if destination is None and arguments.checkpoint is not None:
            destination = arguments.checkpoint.with_name(
                f"{arguments.checkpoint.stem}-resumed{arguments.checkpoint.suffix}"
            )
        checkpoint = trainer.save(destination)
        return {"completed_steps": result.completed_steps, "checkpoint": checkpoint.as_posix()}
    if arguments.a04_command == "eval":
        result = trainer.evaluate()
        return {"completed_steps": result.completed_steps, "steps": [asdict(step) for step in result.steps]}
    profile, measured = trainer.profile(warmup=arguments.warmup, repetitions=arguments.repetitions)
    return {"profile": profile.to_dict(), "metrics": dict(measured.metrics)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ReliableEndo-GS CLI."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "smoke":
            run_smoke(arguments.config)
            return 0

        if arguments.command == "data":
            if arguments.data_command == "resolve":
                resolved_root = run_data_resolve(arguments.config, arguments.data_root)
                print(resolved_root)
                return 0
            if arguments.data_command == "validate":
                report = run_data_validate(arguments.config, arguments.data_root)
                print(
                    json.dumps(
                        report.to_dict(redact_paths=True),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0 if report.valid else 1
            if arguments.data_command == "a01-index":
                print(json.dumps(run_a01_index(arguments.data_root, arguments.output, roles_path=arguments.roles, require_all_folds=arguments.require_all_folds), indent=2, sort_keys=True))
                return 0
        if arguments.command == "a01":
            print(json.dumps(_run_a01(arguments), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if arguments.command == "a04":
            print(json.dumps(_run_a04(arguments), ensure_ascii=False, indent=2, sort_keys=True, default=str))
            return 0
    except (A01WorkflowError, A04TrainingError, ScaredA00SampleError, ConfigError, LookupError, OSError, RuntimeError, ValueError) as error:
        parser.exit(status=2, message=f"error: {error}\n")

    parser.error(f"Unsupported command: {arguments.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
