"""Minimal command-line interface for infrastructure validation."""

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from reliable_endo_gs.config import ConfigError, config_to_dict, hash_config, load_config
from reliable_endo_gs.data.paths import resolve_data_root
from reliable_endo_gs.data.registry import get_dataset_adapter
from reliable_endo_gs.data.scared_a00 import (
    A00Protocol,
    A01Manifest,
    A01ManifestEntry,
    ScaredA00Adapter,
    ScaredA00ProtocolError,
    ScaredA00SampleError,
    build_a01_manifest,
)
from reliable_endo_gs.data.scared_a00_preprocess import (
    ScaredA00PreprocessError,
    _validate_calibration_evidence,
    a00_smoke_command,
    prepare_scared_a00_sample,
)
from reliable_endo_gs.data.schema import load_dataset_config
from reliable_endo_gs.data.validation import DatasetValidationReport
from reliable_endo_gs.reconstruction import load_reference_config, reference_checkpoint_path
from reliable_endo_gs.rendering import ProductionFieldRenderer
from reliable_endo_gs.rendering.cross_view import ReferenceRenderer
from reliable_endo_gs.rendering.field import ExistingRendererBridge
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.runtime.logging import configure_logging
from reliable_endo_gs.runtime.manifest import (
    create_run_id,
    create_run_manifest,
    utc_timestamp,
    write_run_manifest,
)
from reliable_endo_gs.teachers import TargetCache, Teacher, load_a01_roles
from reliable_endo_gs.teachers.workflow import (
    A01WorkflowError,
    collect_targets,
    infer_and_align,
    load_teacher_factory,
    target_cache_key,
    validate_cache,
)
from reliable_endo_gs.training.a04_reference import A04ReferenceTrainer, A04TrainingError
from reliable_endo_gs.training.harness import TrainingLog, TrainingSignals
from reliable_endo_gs.training.preflight import preflight_a04_data
from reliable_endo_gs.utils.io import create_unique_directory, write_json_atomic, write_yaml_atomic
from reliable_endo_gs.utils.seed import set_seed

_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "a01-index",
        description="Build the active SCARED dataset_1..dataset_9 train/test manifest",
    )
    data_index_parser.add_argument("--data-root", required=True, type=Path)
    data_index_parser.add_argument("--output", required=True, type=Path)
    fold_policy = data_index_parser.add_mutually_exclusive_group()
    fold_policy.add_argument(
        "--require-all-folds",
        dest="require_all_folds",
        action="store_true",
        help="Require every active SCARED fold 1-9 (the default)",
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

    a00_parser = subparsers.add_parser(
        "a00", description="Original SCARED A00 sample preparation and deployment preflight"
    )
    a00_subparsers = a00_parser.add_subparsers(dest="a00_command", required=True)
    a00_prepare_parser = a00_subparsers.add_parser(
        "prepare", description="Prepare one original SCARED sample outside the repository"
    )
    a00_prepare_parser.add_argument("--data-root", required=True, type=Path)
    a00_prepare_parser.add_argument("--sample-id", required=True)
    a00_prepare_parser.add_argument("--output-root", required=True, type=Path)
    a00_prepare_parser.add_argument(
        "--config", type=Path, default=Path("configs/data/a00_scared_sample_v1.yaml")
    )
    a00_prepare_parser.add_argument("--calibration-evidence", required=True, type=Path)

    a00_prepare_dataset_parser = a00_subparsers.add_parser(
        "prepare-dataset",
        description="Stream original SCARED folds 1-9 into an external A00 preprocessing root",
    )
    a00_prepare_dataset_parser.add_argument("--data-root", required=True, type=Path)
    a00_prepare_dataset_parser.add_argument("--output-root", required=True, type=Path)
    a00_prepare_dataset_parser.add_argument(
        "--config", type=Path, default=Path("configs/data/a00_scared_sample_v1.yaml")
    )
    a00_prepare_dataset_parser.add_argument(
        "--roles", type=Path, default=Path("configs/teachers/a01_scared.yaml")
    )

    for name, description in (
        (
            "preflight",
            "Check one sample's paths, hashes, protocol, and provenance without loading images or GT",
        ),
        ("smoke", "Load one GT-free A00 DeploymentStereoBatch"),
    ):
        a00_check_parser = a00_subparsers.add_parser(name, description=description)
        a00_check_parser.add_argument("--data-root", required=True, type=Path)
        a00_check_parser.add_argument("--sample-id", required=True)
        a00_check_parser.add_argument(
            "--roles", type=Path, default=Path("configs/teachers/a01_scared.yaml")
        )
        a00_check_parser.add_argument(
            "--config", type=Path, default=Path("configs/data/a00_scared_sample_v1.yaml")
        )

    a01_parser = subparsers.add_parser(
        "a01", description="A01 frozen teacher and target-cache commands"
    )
    a01_subparsers = a01_parser.add_subparsers(dest="a01_command", required=True)
    teacher_preflight_parser = a01_subparsers.add_parser(
        "teacher-preflight",
        description="Verify external teacher artifact pins without loading a model",
    )
    teacher_preflight_parser.add_argument("--teacher-config", type=Path, required=True)
    for name, description in (
        ("teacher-smoke", "Run one explicit injected teacher and A00 alignment"),
        ("cache", "Generate/resume A01 targets from SCARED through a frozen teacher"),
        ("cache-validate", "Validate every A01 cache item and manifest checksum"),
    ):
        command_parser = a01_subparsers.add_parser(name, description=description)
        command_parser.add_argument(
            "--cache-root", type=Path, default=Path("artifacts/targets/a01")
        )
        command_parser.add_argument("--data-root", type=Path)
        command_parser.add_argument("--manifest", type=Path)
        command_parser.add_argument(
            "--teacher-factory", help="Injected module:function Teacher factory"
        )
        command_parser.add_argument(
            "--roles", type=Path, default=Path("configs/teachers/a01_scared.yaml")
        )
        command_parser.add_argument(
            "--split-identity", default="a01-scared-final-train-1-7-test-8-9-v1"
        )
        command_parser.add_argument(
            "--device",
            default="cpu",
            help="Explicit teacher target device; use cuda:0 on the training server",
        )
        command_parser.add_argument(
            "--resolution-hw",
            nargs=2,
            type=int,
            metavar=("HEIGHT", "WIDTH"),
            help="Explicit A00 output grid; match the A04 runtime resolution",
        )
    a01_subparsers.choices["cache-validate"].set_defaults(cache_root=Path("artifacts/targets/a01"))

    a04_parser = subparsers.add_parser(
        "a04", description="A04 bounded reference smoke/train/eval/profile commands"
    )
    a04_subparsers = a04_parser.add_subparsers(dest="a04_command", required=True)
    for name, description in (
        ("preflight", "Decode and validate every deployment sample in the selected role on CPU"),
        ("smoke", "Run one bounded A04 batch"),
        ("train", "Train the independent A04 head for bounded steps"),
        ("eval", "Evaluate a frozen A04 checkpoint"),
        ("profile", "Profile one bounded A04 batch"),
    ):
        command_parser = a04_subparsers.add_parser(name, description=description)
        command_parser.add_argument(
            "--config", type=Path, default=Path("configs/reconstruction/a04_reference.yaml")
        )
        command_parser.add_argument("--data-root", required=True, type=Path)
        command_parser.add_argument("--manifest", required=True, type=Path)
        command_parser.add_argument(
            "--roles", type=Path, default=Path("configs/teachers/a01_scared.yaml")
        )
        command_parser.add_argument(
            "--cache-root", type=Path, default=Path("artifacts/targets/a01")
        )
        command_parser.add_argument(
            "--teacher-factory", help="Injected module:function Teacher factory"
        )
        command_parser.add_argument(
            "--renderer", choices=("reference", "native"), default="reference"
        )
        command_parser.add_argument(
            "--data-role",
            choices=("final_train", "final_test"),
            help="Override the role selected by the resolved A04 config",
        )
        command_parser.add_argument("--device", default=None)
        command_parser.add_argument("--checkpoint", type=Path)
        command_parser.add_argument("--output-checkpoint", type=Path)
        command_parser.add_argument(
            "--epochs",
            type=int,
            default=1,
            help="Total epoch budget, including epochs completed before training resume",
        )
        command_parser.add_argument("--max-steps", type=int)
        command_parser.add_argument("--warmup", type=int, default=0)
        command_parser.add_argument("--repetitions", type=int, default=1)
    train_parser = a04_subparsers.choices["train"]
    train_parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Log every N consumed samples (also first/last/skipped samples)",
    )
    train_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Save every N successful optimizer updates",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Periodic checkpoint directory; defaults beside the final checkpoint",
    )
    train_parser.add_argument(
        "--log-file", type=Path, help="New JSONL session log; refuses to overwrite an existing file"
    )
    train_parser.add_argument(
        "--max-consecutive-skips",
        type=int,
        default=100,
        help="Stop with a recovery checkpoint if unsupported samples form a sustained sequence",
    )
    preflight_parser = a04_subparsers.choices["preflight"]
    preflight_parser.add_argument("--workers", type=int, default=4)
    preflight_parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="New JSON report outside the data/source files",
    )
    benchmark_parser = subparsers.add_parser("benchmark", description="Frozen benchmark harness")
    benchmark_commands = benchmark_parser.add_subparsers(dest="benchmark_command", required=True)
    benchmark_a04 = benchmark_commands.add_parser("a04", description="A04 RGB/GT geometry/timing")
    benchmark_a04.add_argument("--checkpoint", type=Path, required=True)
    benchmark_a04.add_argument("--data-root", type=Path, required=True)
    benchmark_a04.add_argument("--manifest", type=Path, required=True)
    benchmark_a04.add_argument(
        "--roles", type=Path, default=Path("configs/teachers/a01_scared.yaml")
    )
    benchmark_a04.add_argument(
        "--protocol", type=Path, default=Path("configs/evaluation/a04_benchmark.yaml")
    )
    benchmark_a04.add_argument("--output-dir", type=Path, required=True)
    benchmark_a04.add_argument(
        "--lpips-backbone",
        type=Path,
        required=True,
        help="Existing local torchvision AlexNet state_dict",
    )
    benchmark_a04.add_argument(
        "--lpips-calibration",
        type=Path,
        required=True,
        help="Existing official LPIPS v0.1 alex.pth",
    )
    benchmark_a04.add_argument("--sample-id", action="append", default=[])
    benchmark_a04.add_argument("--max-samples", type=int)
    benchmark_a04.add_argument("--resume", action="store_true")
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


def _load_a01_adapter(
    data_root: Path, manifest_path: Path, roles_path: Path
) -> tuple[ScaredA00Adapter, object]:
    roles = load_a01_roles(roles_path)
    # Reuse the explicit role object for cache and adapter validation while
    # keeping all sample path resolution inside the first-party adapter.
    from reliable_endo_gs.data.scared_a00 import load_a01_manifest

    manifest = load_a01_manifest(manifest_path, roles=roles)
    return ScaredA00Adapter(data_root, manifest, roles=roles), roles


def run_a00_preflight(
    data_root: Path,
    sample_id: str,
    *,
    roles_path: Path,
    config_path: Path,
    print_smoke_command: bool,
) -> dict[str, object]:
    """Validate one processed sample's metadata and files without loading images or GT."""
    roles = load_a01_roles(roles_path)
    dataset_id = sample_id.split("/", maxsplit=1)[0]
    data_role = roles.role_for_dataset(dataset_id)
    normalized = roles.validate_sample_id(sample_id, role=data_role, cache=False)
    resolved_data_root = Path(data_root).expanduser().resolve(strict=True)
    data_dir = resolved_data_root / Path(*normalized.split("/")[:2]) / "data"
    frame_id = normalized.rsplit("/", maxsplit=1)[-1]
    batch_provenance_path = data_dir / "a00_provenance" / f"{frame_id}.json"
    provenance_path = (
        batch_provenance_path
        if batch_provenance_path.is_file()
        else data_dir / "a00_provenance.json"
    )
    try:
        provenance_value: object = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScaredA00SampleError(
            f"missing or invalid a00_provenance.json at {provenance_path}: {error}"
        ) from error
    if not isinstance(provenance_value, dict):
        raise ScaredA00SampleError("a00_provenance.json must contain a JSON object")
    if provenance_value.get("sample_id") != normalized:
        raise ScaredA00SampleError("A00 provenance sample_id does not match requested sample")
    if provenance_value.get("data_role") != data_role:
        raise ScaredA00SampleError("A00 provenance data role differs from the active role config")
    evidence_sha256 = provenance_value.get("calibration_evidence_sha256")
    if not isinstance(evidence_sha256, str) or _SHA256_HEX.fullmatch(evidence_sha256) is None:
        raise ScaredA00SampleError(
            "A00 provenance calibration-evidence identity must be a SHA-256 hex digest"
        )
    outputs = provenance_value.get("output_files_sha256")
    if not isinstance(outputs, dict) or not outputs:
        raise ScaredA00SampleError("A00 provenance must list hashes for processed output files")
    required_outputs = {
        f"left_finalpass/{frame_id}.png",
        f"right_finalpass/{frame_id}.png",
        f"frame_data/{frame_id}.json",
        f"newpram_data/{frame_id}.json",
        f"disparity/{frame_id}.tiff",
        f"reprojection_data/{frame_id}.json",
        "a00_protocol.json",
    }
    if not required_outputs.issubset(outputs):
        raise ScaredA00SampleError(
            f"A00 provenance is missing required processed hashes: {sorted(required_outputs - set(outputs))}"
        )
    data_root_resolved = data_dir.resolve(strict=True)
    for relative_name, expected_hash in outputs.items():
        if (
            not isinstance(relative_name, str)
            or not isinstance(expected_hash, str)
            or _SHA256_HEX.fullmatch(expected_hash) is None
        ):
            raise ScaredA00SampleError(
                "A00 provenance output hash entries must contain relative paths and full SHA-256 hex digests"
            )
        path = (data_root_resolved / relative_name).resolve(strict=True)
        if data_root_resolved not in path.parents or not path.is_file():
            raise ScaredA00SampleError(
                "A00 provenance output path escapes the sample data directory"
            )
        digest = _sha256_file(path)
        if digest != expected_hash:
            raise ScaredA00SampleError(f"processed output hash mismatch: {relative_name}")

    protocol_path = data_dir / "a00_protocol.json"
    try:
        protocol_value: object = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScaredA00ProtocolError(f"invalid A00 protocol {protocol_path}: {error}") from error
    if not isinstance(protocol_value, dict) or not isinstance(
        protocol_value.get("preprocessing"), dict
    ):
        raise ScaredA00ProtocolError("A00 protocol must include preprocessing identity metadata")
    if "images_grid" not in protocol_value["preprocessing"]:
        raise ScaredA00ProtocolError("A00 protocol preprocessing.images_grid must be explicit")
    parsed_protocol = A00Protocol.from_mapping(protocol_value, source=str(protocol_path))
    preprocessing = protocol_value["preprocessing"]
    for field in (
        "code_sha256",
        "config_sha256",
        "calibration_evidence_sha256",
        "calibration_value_source_sha256",
    ):
        value = preprocessing.get(field)
        if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
            raise ScaredA00ProtocolError(
                f"A00 protocol preprocessing.{field} must be a full SHA-256"
            )

    from reliable_endo_gs.data.scared_a00_preprocess import _read_config

    provenance_preprocessing = provenance_value.get("preprocessing")
    if not isinstance(provenance_preprocessing, dict):
        raise ScaredA00SampleError("A00 provenance is missing preprocessing identity metadata")
    requested_config = Path(config_path).expanduser().resolve(strict=True)
    batch_code_digest = provenance_preprocessing.get("batch_code_sha256")
    if batch_code_digest is not None:
        recorded_config = provenance_preprocessing.get("config_path")
        if not isinstance(recorded_config, str) or not Path(recorded_config).is_absolute():
            raise ScaredA00ProtocolError("batch A00 provenance must record an absolute config path")
        resolved_config = Path(recorded_config).resolve(strict=True)
        resolved_provenance_root = (resolved_data_root / "_provenance" / "configs").resolve(
            strict=True
        )
        if resolved_provenance_root not in resolved_config.parents:
            raise ScaredA00ProtocolError(
                "batch A00 config path escapes its external provenance directory"
            )
        # The checked-in sample config is the batch command's template. The
        # output-specific variant is pinned by provenance and is used for the
        # exact single-sample smoke command.
        if (
            requested_config != resolved_config
            and requested_config.name != "a00_scared_sample_v1.yaml"
        ):
            raise ScaredA00ProtocolError(
                "supplied config is neither the sample template nor the output-pinned batch config"
            )
    else:
        resolved_config = requested_config
    config, config_sha256 = _read_config(resolved_config)
    calibration_config = config.get("calibration")
    if not isinstance(calibration_config, dict):
        raise ScaredA00ProtocolError("A00 preprocessing config is missing calibration settings")
    trusted_value_source = calibration_config.get("trusted_calibration_value_source_sha256")
    if (
        not isinstance(trusted_value_source, str)
        or _SHA256_HEX.fullmatch(trusted_value_source) is None
    ):
        raise ScaredA00ProtocolError(
            "no calibration value-source SHA-256 is pinned; A00 preflight remains blocked"
        )

    invocation = provenance_value.get("invocation")
    if not isinstance(invocation, dict) or invocation.get("module") != "reliable_endo_gs.cli.main":
        raise ScaredA00SampleError("A00 provenance must record the actual CLI invocation")
    invocation_paths = invocation.get("resolved_paths")
    if not isinstance(invocation_paths, dict):
        raise ScaredA00SampleError("A00 provenance invocation must list resolved paths")
    evidence_path_value = invocation_paths.get("calibration_evidence")
    if not isinstance(evidence_path_value, str) or not Path(evidence_path_value).is_absolute():
        raise ScaredA00SampleError(
            "A00 provenance invocation must record an absolute calibration-evidence path"
        )
    evidence_path = Path(evidence_path_value)
    try:
        evidence_bytes = evidence_path.read_bytes()
    except OSError as error:
        raise ScaredA00SampleError(
            f"unable to read recorded calibration evidence {evidence_path}: {error}"
        ) from error
    if hashlib.sha256(evidence_bytes).hexdigest() != evidence_sha256:
        raise ScaredA00SampleError(
            "recorded calibration evidence file SHA-256 differs from the provenance identity"
        )
    try:
        evidence_value: object = json.loads(evidence_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScaredA00SampleError(
            f"recorded calibration evidence is not valid JSON: {error}"
        ) from error
    if not isinstance(evidence_value, dict):
        raise ScaredA00SampleError("recorded calibration evidence must be a JSON object")
    evidence_sample_id = provenance_value.get("calibration_evidence_sample_id", normalized)
    if not isinstance(evidence_sample_id, str):
        raise ScaredA00SampleError("A00 calibration evidence sample ID must be a string")
    evidence_role = roles.role_for_dataset(evidence_sample_id.split("/", maxsplit=1)[0])
    if evidence_role != data_role:
        raise ScaredA00SampleError("A00 calibration evidence crosses the active role boundary")
    validated_evidence = _validate_calibration_evidence(
        evidence_value,
        sample_id=evidence_sample_id,
        trusted_value_source_sha256=trusted_value_source,
    )
    evidence_member_hashes = validated_evidence["source_member_sha256"]
    if not isinstance(evidence_member_hashes, dict):
        raise ScaredA00SampleError("validated calibration evidence is missing source member hashes")
    frame_correspondence = provenance_value.get("frame_correspondence")
    if not isinstance(frame_correspondence, dict):
        raise ScaredA00SampleError("A00 provenance must record frame/member correspondence")
    expected_frame = normalized.rsplit("/", maxsplit=1)[-1]
    expected_scene = expected_frame.replace("frame_data", "scene_points")
    if (
        frame_correspondence.get("frame_data_id") != expected_frame
        or frame_correspondence.get("scene_points_id") != expected_scene
        or frame_correspondence.get("matched") is not True
        or frame_correspondence.get("ambiguous") is not False
    ):
        raise ScaredA00SampleError(
            "A00 provenance does not verify the requested frame_data/scene_points correspondence"
        )
    source_files = provenance_value.get("source_files")
    if not isinstance(source_files, dict):
        raise ScaredA00SampleError("A00 provenance must record original source-file identities")
    for key, expected_member, expected_hash in (
        ("frame_data_archive", f"{expected_frame}.json", evidence_member_hashes.get("frame_data")),
        (
            "scene_points_archive",
            f"{expected_scene}.tiff",
            evidence_member_hashes.get("scene_points"),
        ),
    ):
        source_record = source_files.get(key)
        if not isinstance(source_record, dict) or source_record.get("member") != expected_member:
            raise ScaredA00SampleError(f"A00 provenance {key} member differs from the sample ID")
        source_hash = source_record.get("member_sha256")
        recorded_frame_hash = frame_correspondence.get(
            "frame_data_member_sha256"
            if key == "frame_data_archive"
            else "scene_points_member_sha256"
        )
        if (
            not isinstance(source_hash, str)
            or _SHA256_HEX.fullmatch(source_hash) is None
            or source_hash != recorded_frame_hash
        ):
            raise ScaredA00SampleError(
                f"A00 provenance {key} member hash correspondence is inconsistent"
            )
        if evidence_sample_id == normalized and source_hash != expected_hash:
            raise ScaredA00SampleError(
                f"calibration evidence member identity differs from provenance {key}"
            )

    processor_path = Path(__file__).resolve().parents[1] / "data" / "scared_a00_preprocess.py"
    code_sha256 = _sha256_file(processor_path)
    if preprocessing.get("config_sha256") != config_sha256:
        raise ScaredA00ProtocolError("A00 protocol config SHA-256 differs from the supplied config")
    processor_compatibility = "exact_code_identity"
    if batch_code_digest is None:
        if preprocessing.get("code_sha256") != code_sha256:
            raise ScaredA00ProtocolError(
                "A00 protocol processor SHA-256 differs from this checkout"
            )
    else:
        from reliable_endo_gs.data.scared_a00_identity import a00_processor_compatibility

        if preprocessing.get("batch_code_sha256") != batch_code_digest:
            raise ScaredA00ProtocolError("A00 protocol/provenance batch code identity mismatch")
        batch_processor_path = Path(__file__).resolve().parents[1] / "data" / "scared_a00_batch.py"
        metric_summary = provenance_value.get("metric_validation")
        if not isinstance(metric_summary, dict):
            raise ScaredA00ProtocolError(
                "batch A00 provenance must include metric support metadata"
            )
        compatibility = a00_processor_compatibility(
            (preprocessing.get("code_sha256"), batch_code_digest),
            (code_sha256, _sha256_file(batch_processor_path)),
            [metric_summary],
        )
        if compatibility is None:
            raise ScaredA00ProtocolError(
                "A00 batch processor SHA-256 belongs to an unreviewed code identity"
            )
        processor_compatibility = compatibility
    if preprocessing.get("calibration_value_source_sha256") != trusted_value_source:
        raise ScaredA00ProtocolError(
            "A00 protocol calibration value-source identity differs from config"
        )
    if preprocessing.get("calibration_evidence_sha256") != evidence_sha256:
        raise ScaredA00ProtocolError("A00 protocol/evidence manifest identity mismatch")
    if protocol_value.get("metric_geometry_evidence_sha256") != evidence_sha256:
        raise ScaredA00ProtocolError("A00 protocol metric-geometry evidence identity mismatch")
    for field in (
        "code_sha256",
        "config_sha256",
        "calibration_evidence_sha256",
        "calibration_value_source_sha256",
    ):
        if provenance_preprocessing.get(field) != preprocessing.get(field):
            raise ScaredA00ProtocolError(f"A00 protocol/provenance identity mismatch for {field}")
    extrinsics = protocol_value.get("extrinsics")
    if not isinstance(extrinsics, dict):
        raise ScaredA00ProtocolError("A00 protocol is missing stereo extrinsics metadata")
    if (
        parsed_protocol.extrinsic_source != "newpram.T_R_from_L"
        or parsed_protocol.extrinsic_direction != "right_from_left"
        or parsed_protocol.extrinsic_units != "m"
        or extrinsics.get("calibration_value_source_sha256") != trusted_value_source
    ):
        raise ScaredA00ProtocolError("A00 protocol stereo extrinsics identity is inconsistent")
    calibration_evidence = validated_evidence.get("calibration")
    if not isinstance(calibration_evidence, dict):
        raise ScaredA00SampleError("validated calibration evidence is missing calibration metadata")
    evidence_source_field = calibration_evidence.get("source_field")
    evidence_translation_unit = calibration_evidence.get("translation_unit")
    if (
        extrinsics.get("raw_source_field") != evidence_source_field
        or extrinsics.get("raw_units") != evidence_translation_unit
        or extrinsics.get("unit_evidence_sha256") != evidence_sha256
        or preprocessing.get("source_translation_field") != evidence_source_field
        or preprocessing.get("source_translation_unit") != evidence_translation_unit
    ):
        raise ScaredA00ProtocolError(
            "A00 protocol/preprocessing calibration identity differs from validated evidence"
        )
    if preprocessing.get("full_sensor_support_verified") is not True:
        raise ScaredA00ProtocolError("A00 preprocessing did not verify full sensor support")

    executable = invocation.get("executable")
    working_directory = invocation.get("working_directory")
    if (
        not isinstance(executable, str)
        or not Path(executable).is_absolute()
        or not isinstance(working_directory, str)
        or not Path(working_directory).is_absolute()
        or not isinstance(invocation.get("pythonpath"), str)
    ):
        raise ScaredA00SampleError(
            "A00 provenance invocation must identify its executable and environment"
        )
    for path_name in ("data_root", "output_root", "config", "calibration_evidence"):
        path_value = invocation_paths.get(path_name)
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise ScaredA00SampleError(
                f"A00 provenance invocation resolved_paths.{path_name} must be absolute"
            )
    if (
        invocation_paths.get("output_root") != resolved_data_root.as_posix()
        or invocation_paths.get("config") != resolved_config.as_posix()
    ):
        raise ScaredA00SampleError(
            "A00 provenance resolved output/config paths do not match this preflight"
        )
    invocation_argv = invocation.get("argv")
    if not isinstance(invocation_argv, list) or not all(
        isinstance(value, str) for value in invocation_argv
    ):
        raise ScaredA00SampleError("A00 provenance invocation argv must be a string list")
    if provenance_preprocessing.get("invocation") != invocation:
        raise ScaredA00SampleError("A00 protocol/provenance invocation records differ")

    command = a00_smoke_command(data_root, normalized, roles_path, resolved_config)
    result: dict[str, object] = {
        "status": "ready_for_a00_smoke",
        "sample_id": normalized,
        "data_role": data_role,
        "sample_path": data_dir.as_posix(),
        "ground_truth_loaded": False,
        "deployment_batch_loaded": False,
        "protocol_identity": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "calibration_evidence_sha256": evidence_sha256,
        "processor_compatibility": processor_compatibility,
        "producer_code_sha256": preprocessing["code_sha256"],
        "producer_batch_code_sha256": batch_code_digest,
    }
    if print_smoke_command:
        result["a00_smoke_command"] = command
    if batch_code_digest is not None:
        result["metric_gt_valid_pixel_count"] = provenance_value["metric_validation"][
            "valid_pixel_count"
        ]
        result["metric_gt_status"] = provenance_value["metric_validation"].get(
            "status", "available"
        )
    return result


def run_a00_sample_check(
    data_root: Path,
    sample_id: str,
    *,
    roles_path: Path,
    config_path: Path,
) -> dict[str, object]:
    """Load one deployment-only A00 sample after metadata preflight."""
    metadata = run_a00_preflight(
        data_root,
        sample_id,
        roles_path=roles_path,
        config_path=config_path,
        print_smoke_command=False,
    )
    roles = load_a01_roles(roles_path)
    normalized = str(metadata["sample_id"])
    data_role = str(metadata["data_role"])
    manifest = A01Manifest(roles.role_hash, (A01ManifestEntry(normalized, data_role),))
    adapter = ScaredA00Adapter(data_root, manifest, roles=roles)
    sample = adapter.load(normalized, data_role=data_role, include_supervision=False)
    if sample.supervision is not None:
        raise ScaredA00SampleError("A00 deployment smoke unexpectedly loaded GT supervision")
    metadata.update(
        {
            "deployment_batch_loaded": True,
            "image_shape": list(sample.batch.left.shape),
            "support_pixel_count": int(sample.batch.support.sum().item()),
            "baseline_m": float(sample.batch.camera.baseline[0].item()),
            "protocol_identity": sample.calibration_metadata["protocol_identity"],
        }
    )
    return metadata


def _run_a00(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.a00_command == "prepare":
        return prepare_scared_a00_sample(
            data_root=arguments.data_root,
            sample_id=arguments.sample_id,
            output_root=arguments.output_root,
            config_path=arguments.config,
            calibration_evidence_path=arguments.calibration_evidence,
            invocation_argv=sys.argv[1:],
        )
    if arguments.a00_command == "prepare-dataset":
        from reliable_endo_gs.data.scared_a00_batch import prepare_scared_a00_dataset

        roles = load_a01_roles(arguments.roles)

        def report_progress(record: object) -> None:
            print(
                json.dumps(record, ensure_ascii=False, sort_keys=True, default=str),
                file=sys.stderr,
                flush=True,
            )

        return prepare_scared_a00_dataset(
            data_root=arguments.data_root,
            output_root=arguments.output_root,
            config_path=arguments.config,
            roles=roles,
            invocation_argv=sys.argv[1:],
            progress_callback=report_progress,
        )
    if arguments.a00_command == "preflight":
        return run_a00_preflight(
            arguments.data_root,
            arguments.sample_id,
            roles_path=arguments.roles,
            config_path=arguments.config,
            print_smoke_command=True,
        )
    return run_a00_sample_check(
        arguments.data_root,
        arguments.sample_id,
        roles_path=arguments.roles,
        config_path=arguments.config,
    )


def _load_cli_teacher(factory_spec: str | None) -> Teacher:
    if not factory_spec:
        raise A01WorkflowError(
            "an explicit --teacher-factory module:function is required; no external teacher is bundled"
        )
    return load_teacher_factory(factory_spec)


def _run_a01(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.a01_command == "teacher-preflight":
        from reliable_endo_gs.teachers.external import teacher_preflight

        return teacher_preflight(arguments.teacher_config)
    roles = load_a01_roles(arguments.roles)
    if arguments.a01_command == "cache-validate":
        if (
            arguments.manifest is None
            or arguments.data_root is None
            or arguments.teacher_factory is None
        ):
            raise A01WorkflowError(
                "cache validation requires --manifest, --data-root and --teacher-factory "
                "to verify expected samples and teacher-bound cache keys"
            )
        cache = TargetCache(arguments.cache_root, roles=roles)
        expected_sample_ids = None
        expected_key_digests = None
        from reliable_endo_gs.data.scared_a00 import load_a01_manifest

        expected_manifest = load_a01_manifest(arguments.manifest, roles=roles)
        expected_entries = tuple(
            entry for entry in expected_manifest.entries if entry.data_role in roles.cache_roles
        )
        expected_sample_ids = tuple(entry.sample_id for entry in expected_entries)
        adapter, _ = _load_a01_adapter(arguments.data_root, arguments.manifest, arguments.roles)
        teacher = _load_cli_teacher(arguments.teacher_factory)
        output_hw = None if arguments.resolution_hw is None else tuple(arguments.resolution_hw)
        expected_key_digests = [
            target_cache_key(
                adapter.load(
                    entry.sample_id,
                    data_role=entry.data_role,
                    include_supervision=False,
                    output_hw=output_hw,
                ),
                teacher,
                split_identity=arguments.split_identity,
                roles=roles,
            ).digest
            for entry in expected_entries
        ]
        validation_report = validate_cache(
            cache,
            expected_sample_ids=expected_sample_ids,
            expected_key_digests=expected_key_digests,
        )
        if (
            validation_report["stale_or_missing"]
            or validation_report["missing_expected"]
            or validation_report["missing_expected_keys"]
        ):
            raise A01WorkflowError(
                "A01 cache validation failed: "
                f"stale_or_missing={validation_report['stale_or_missing']}, "
                f"missing_expected={validation_report['missing_expected']}, "
                f"missing_expected_keys={validation_report['missing_expected_keys']}"
            )
        return validation_report
    if arguments.data_root is None or arguments.manifest is None:
        raise A01WorkflowError(
            "--data-root and --manifest are required for A01 teacher/cache commands"
        )
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
        output_hw=(None if arguments.resolution_hw is None else tuple(arguments.resolution_hw)),
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
    if arguments.a04_command == "preflight":
        if arguments.report.exists():
            raise A04TrainingError(f"refusing to overwrite existing preflight: {arguments.report}")
        report = preflight_a04_data(
            adapter,
            data_role=config.data_role,
            workers=arguments.workers,
            progress=lambda event: print(json.dumps(dict(event)), flush=True),
        )
        write_json_atomic(arguments.report, report)
        if report["status"] != "PASS":
            raise A04TrainingError(f"deployment preflight failed; inspect {arguments.report}")
        return report
    teacher = load_teacher_factory(arguments.teacher_factory) if arguments.teacher_factory else None
    cache = (
        TargetCache(arguments.cache_root, roles=roles) if arguments.cache_root.exists() else None
    )
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
        trainer.resume(arguments.checkpoint, restore_training=arguments.a04_command == "train")
    if arguments.a04_command == "smoke":
        result = trainer.smoke()
        return {
            "completed_steps": result.completed_steps,
            "steps": [asdict(step) for step in result.steps],
        }
    if arguments.a04_command == "train":
        destination = arguments.output_checkpoint
        if destination is None and arguments.checkpoint is not None:
            destination = arguments.checkpoint.with_name(
                f"{arguments.checkpoint.stem}-resumed{arguments.checkpoint.suffix}"
            )
        if destination is None:
            destination = reference_checkpoint_path(trainer.config)
        if destination.exists():
            raise A04TrainingError(f"refusing to overwrite existing checkpoint: {destination}")
        checkpoint_dir = (
            arguments.checkpoint_dir or destination.parent / f"{destination.stem}-checkpoints"
        )
        source_root = Path(__file__).resolve().parents[1]
        source_digest = hashlib.sha256()
        for source in sorted(source_root.rglob("*.py")):
            source_digest.update(source.relative_to(source_root).as_posix().encode("utf-8"))
            source_digest.update(b"\0")
            source_digest.update(source.read_bytes())
        import torch

        metadata = {
            "git_commit": get_git_commit(source_root.parents[1]),
            "git_dirty": is_git_dirty(source_root.parents[1]),
            "source_sha256": source_digest.hexdigest(),
            "config_hash": trainer.config.config_hash,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "device": str(trainer.device),
            "gpu": torch.cuda.get_device_name(trainer.device)
            if trainer.device.type == "cuda"
            else None,
            "seed": trainer.config.seed,
            "log_every": arguments.log_every,
            "checkpoint_every": arguments.checkpoint_every,
            "max_consecutive_skips": arguments.max_consecutive_skips,
            "resume_checkpoint": None
            if arguments.checkpoint is None
            else str(arguments.checkpoint),
        }
        log_path = arguments.log_file or destination.with_suffix(".jsonl")
        with TrainingLog(log_path) as log, TrainingSignals() as signals:
            try:
                result = trainer.train(
                    epochs=arguments.epochs,
                    max_steps=arguments.max_steps,
                    log_every=arguments.log_every,
                    checkpoint_every=arguments.checkpoint_every,
                    checkpoint_dir=checkpoint_dir,
                    progress=log.emit,
                    run_metadata=metadata,
                    max_consecutive_skips=arguments.max_consecutive_skips,
                    stop_requested=signals.requested,
                )
                checkpoint = trainer.save(destination)
                log.emit(
                    {
                        "event": "final_checkpoint_saved",
                        **trainer.training_progress,
                        "checkpoint": checkpoint.as_posix(),
                    }
                )
            except (Exception, KeyboardInterrupt) as error:
                log.emit(
                    {
                        "event": "train_error",
                        **trainer.training_progress,
                        "error": str(error),
                        "error_type": type(error).__name__,
                    }
                )
                try:
                    recovery = trainer.save_recovery(checkpoint_dir, error)
                except Exception as recovery_error:
                    log.emit(
                        {
                            "event": "recovery_checkpoint_unavailable",
                            "error": str(recovery_error),
                            **trainer.training_progress,
                        }
                    )
                else:
                    log.emit(
                        {
                            "event": "recovery_checkpoint_saved",
                            "checkpoint": str(recovery),
                            **trainer.training_progress,
                        }
                    )
                raise
        return {
            "completed_steps": result.completed_steps,
            "checkpoint": checkpoint.as_posix(),
            **trainer.training_progress,
            "stop_reason": result.stop_reason,
        }
    if arguments.a04_command == "eval":
        result = trainer.evaluate()
        return {
            "completed_steps": result.completed_steps,
            "steps": [asdict(step) for step in result.steps],
        }
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
                print(
                    json.dumps(
                        run_a01_index(
                            arguments.data_root,
                            arguments.output,
                            roles_path=arguments.roles,
                            require_all_folds=arguments.require_all_folds,
                        ),
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
        if arguments.command == "a01":
            print(json.dumps(_run_a01(arguments), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if arguments.command == "a00":
            print(json.dumps(_run_a00(arguments), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if arguments.command == "a04":
            print(
                json.dumps(
                    _run_a04(arguments), ensure_ascii=False, indent=2, sort_keys=True, default=str
                )
            )
            return 0
        if arguments.command == "benchmark":
            from reliable_endo_gs.evaluation.a04_benchmark import _safe, run_a04_benchmark

            result = run_a04_benchmark(
                checkpoint=arguments.checkpoint,
                data_root=arguments.data_root,
                manifest=arguments.manifest,
                roles_path=arguments.roles,
                protocol_path=arguments.protocol,
                output_dir=arguments.output_dir,
                lpips_backbone=arguments.lpips_backbone,
                lpips_calibration=arguments.lpips_calibration,
                sample_ids=arguments.sample_id,
                max_samples=arguments.max_samples,
                resume=arguments.resume,
            )
            print(json.dumps(_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    except (
        A01WorkflowError,
        A04TrainingError,
        ScaredA00PreprocessError,
        ScaredA00SampleError,
        ConfigError,
        LookupError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.exit(status=2, message=f"error: {error}\n")

    parser.error(f"Unsupported command: {arguments.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
