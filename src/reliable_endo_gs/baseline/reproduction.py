"""Development-only baseline reproduction preflight and contract smoke."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from reliable_endo_gs.baseline.config import BaselineConfig
from reliable_endo_gs.baseline.native import NativeBaselineRunner, NativeInferenceInput
from reliable_endo_gs.baseline.upstream import inspect_capabilities
from reliable_endo_gs.contracts.samples import StereoBatch
from reliable_endo_gs.data import (
    build_scared_c_index,
    load_dataset_config,
    load_scared_c_sample,
)
from reliable_endo_gs.data.index import SampleIndex, ScaredCRecord
from reliable_endo_gs.data.paths import DataRootResolutionError, resolve_data_root
from reliable_endo_gs.data.schema import DatasetConfig, dataset_config_to_dict, hash_dataset_config
from reliable_endo_gs.evaluation import BaselineArtifact, make_artifact, write_artifact
from reliable_endo_gs.profiling import profile_callable
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty


class ReproductionRefusal(RuntimeError):
    """Raised when native inference cannot be run without guessing."""


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ReproductionRefusal(f"{context} must be a mapping")
    return value


def _load_yaml(path: Path) -> Mapping[str, object]:
    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ReproductionRefusal(f"unable to read experiment config: {error}") from error
    except yaml.YAMLError as error:
        raise ReproductionRefusal(f"invalid experiment YAML: {error}") from error
    return _mapping(raw, "experiment configuration")


def _mapping_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hardware_metadata() -> dict[str, object]:
    """Collect portable runtime facts without serializing machine paths."""

    metadata: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
            }
        )
        if torch.cuda.is_available() and torch.cuda.device_count():
            metadata["cuda_device_name"] = torch.cuda.get_device_name(0)
    except (ImportError, OSError, RuntimeError) as error:
        metadata["torch_probe_error"] = type(error).__name__
    return metadata


def _development_artifact(
    *,
    config_path: Path,
    raw: Mapping[str, object],
    dataset_config_path: Path,
    dataset: DatasetConfig,
    index: SampleIndex,
    record: ScaredCRecord,
    sample: StereoBatch,
    reason: str,
    capabilities: Mapping[str, object] | None = None,
    profiling: Mapping[str, object] | None = None,
) -> BaselineArtifact:
    """Create a non-promotable, path-independent development manifest."""

    experiment = _mapping(raw.get("experiment"), "experiment")
    baseline = _mapping(raw.get("baseline"), "baseline")
    unresolved = {
        key: sample.metadata[key]
        for key in (
            "depth_units",
            "camera_axes",
            "camera_convention",
            "rectification",
            "pixel_center",
            "pose_source",
            "stereo_extrinsics",
        )
        if key in sample.metadata
    }
    return make_artifact(
        config={
            "config_name": config_path.name,
            "resolved": raw,
            "scientific_status": "development_only",
        },
        config_hash=_mapping_hash(raw),
        dataset={
            "dataset": "SCARED-C",
            "dataset_config": dataset_config_to_dict(dataset),
            "dataset_config_hash": hash_dataset_config(dataset),
            "protocol": dataset.options.get("protocol"),
            "original_scared_distinct": True,
            "unresolved_semantics": unresolved,
            "config_source": dataset_config_path.name,
        },
        split={
            "manifest": None,
            "status": "final_grouped_split_pending_full_protocol_data",
            "grouping": "sequence/keyframe",
            "index_hash": index.index_hash,
        },
        sample_selection={
            "index_count": len(index.records),
            "sample_ids": [record.sample_id],
            "sequence_ids": [record.sequence_id],
            "image_shape": list(sample.left.shape),
            "left_valid_xyz": int(sample.masks["left_depth_valid"].sum().item()),
            "right_valid_xyz": int(sample.masks["right_depth_valid"].sum().item()),
        },
        seeds={"seed": experiment.get("seed")},
        upstream={
            "project": "Endo-E2E-GS",
            "commit": baseline.get("upstream_commit"),
            "patch_ids": baseline.get("patch_ids", []),
            "source": "third_party/endo_e2e_gs",
        },
        checkpoint={
            "identity": baseline.get("checkpoint_id"),
            "sha256": baseline.get("checkpoint_sha256"),
            "available": False,
        },
        metric_schema={
            "name": "reliable_endo_gs.evaluation.v1",
            "configured": raw.get("evaluation", {}),
        },
        metrics={"status": "unavailable", "reason": reason},
        profiling=(
            profiling
            if profiling is not None
            else {"status": "unavailable", "configured": raw.get("profiling", {})}
        ),
        project_git_commit=get_git_commit(Path.cwd()),
        project_git_dirty=is_git_dirty(Path.cwd()),
        hardware=_hardware_metadata(),
        warnings=(
            "development_only; not an accepted baseline artifact",
            "SCARED-C is not original SCARED",
            reason,
        ),
        gate={
            "scientific_status": "development_only",
            "scientific_reproduction": False,
            "accepted": False,
            "reason": reason,
            "capabilities": capabilities or {},
        },
    )


def _attach_artifact(
    result: dict[str, object], artifact: BaselineArtifact, artifact_path: Path | None
) -> None:
    result["scientific_status"] = "development_only"
    result["scientific_reproduction"] = False
    result["artifact_id"] = artifact.artifact_id
    result["artifact"] = artifact.to_dict()
    if artifact_path is not None:
        write_artifact(artifact_path, artifact)
        result["artifact_written"] = True


def _baseline_config(raw: Mapping[str, object], config_dir: Path) -> BaselineConfig:
    baseline = _mapping(raw.get("baseline"), "baseline")
    required = (
        "upstream_root",
        "upstream_commit",
        "upstream_config",
        "entry_point",
        "device",
        "enforce_clean",
    )
    missing = [key for key in required if key not in baseline]
    if missing:
        raise ReproductionRefusal(f"baseline configuration missing fields: {', '.join(missing)}")
    values: dict[str, object] = dict(baseline)
    for key in ("upstream_root", "checkpoint_path"):
        if isinstance(values.get(key), str) and key == "checkpoint_path":
            values[key] = (config_dir / str(values[key])).resolve()
        elif key == "upstream_root":
            values[key] = (config_dir / str(values[key])).resolve()
    checkpoint_id = values.get("checkpoint_id")
    checkpoint_path = values.get("checkpoint_path")
    checkpoint_sha256 = values.get("checkpoint_sha256")
    return BaselineConfig(
        upstream_root=Path(str(values["upstream_root"])),
        upstream_commit=str(values["upstream_commit"]),
        upstream_config=Path(str(values["upstream_config"])),
        entry_point=str(values["entry_point"]),
        device=str(values["device"]),
        enforce_clean=bool(values["enforce_clean"]),
        checkpoint_id=checkpoint_id if isinstance(checkpoint_id, str) else None,
        checkpoint_path=checkpoint_path if isinstance(checkpoint_path, Path) else None,
        checkpoint_sha256=checkpoint_sha256 if isinstance(checkpoint_sha256, str) else None,
    )


def _run_contract_gpu_smoke(sample: StereoBatch) -> dict[str, object]:
    """Exercise only real-sample contract tensors on the explicitly selected GPU."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        return {
            "status": "blocked",
            "reason": "GPU policy requires CUDA_VISIBLE_DEVICES=1 for physical GPU 1",
        }
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            return {"status": "blocked", "reason": "one masked CUDA device is not available"}
        device = torch.device("cuda:0")

        def transfer_contract() -> tuple[torch.Tensor, ...]:
            return (
                sample.left.to(device),
                sample.right.to(device),
                sample.left_camera.intrinsics.to(device),
                sample.right_camera.intrinsics.to(device),
            )

        profile, output = profile_callable(
            transfer_contract,
            warmup=1,
            repetitions=3,
            batch_size=sample.batch_size,
            device=device,
            reset_peak_memory=torch.cuda.reset_peak_memory_stats,
            peak_memory=torch.cuda.max_memory_allocated,
        )
        output_tensors = cast(tuple[torch.Tensor, ...], output)
        output_shapes = [list(tensor.shape) for tensor in output_tensors]
        del output
        return {
            "status": "pass",
            "level": "LEVEL_1_CUDA_TENSOR_ALLOCATION",
            "operation": "StereoBatch contract tensor transfer only",
            "baseline_inference": False,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "output_shapes": output_shapes,
            "profile": profile.to_dict(),
        }
    except (ImportError, OSError, RuntimeError) as error:
        return {
            "status": "blocked",
            "reason": f"CUDA contract smoke failed: {type(error).__name__}",
        }


def _refuse(
    result: dict[str, object],
    *,
    config_path: Path,
    raw: Mapping[str, object],
    dataset_config_path: Path,
    dataset: DatasetConfig,
    index: SampleIndex,
    record: ScaredCRecord,
    sample: StereoBatch,
    reason: str,
    artifact_path: Path | None,
    capabilities: Mapping[str, object] | None = None,
) -> dict[str, object]:
    profiling_value = result.get("gpu_smoke")
    profiling = profiling_value if isinstance(profiling_value, Mapping) else None
    artifact = _development_artifact(
        config_path=config_path,
        raw=raw,
        dataset_config_path=dataset_config_path,
        dataset=dataset,
        index=index,
        record=record,
        sample=sample,
        reason=reason,
        capabilities=capabilities,
        profiling=profiling,
    )
    _attach_artifact(result, artifact, artifact_path)
    result["reason"] = reason
    return result


def reproduce_baseline(
    config_path: Path,
    *,
    data_root: Path | None = None,
    input_bridge: Callable[[object], NativeInferenceInput] | None = None,
    artifact_path: Path | None = None,
    contract_gpu_smoke: bool = False,
) -> dict[str, object]:
    """Run a development preflight and refuse unsupported native execution."""

    raw = _load_yaml(config_path)
    if raw.get("mode") != "development":
        raise ReproductionRefusal("this entry point only supports mode: development")
    dataset_config_path = config_path.parent / str(
        raw.get("dataset_config", "../data/scared_c.yaml")
    )
    dataset = load_dataset_config(dataset_config_path)
    result: dict[str, object] = {"status": "refused", "dataset": dataset.name}
    try:
        root = resolve_data_root(dataset.root, global_root=data_root)
    except DataRootResolutionError as error:
        result["reason"] = f"SCARED-C root cannot be resolved: {error}"
        return result
    result["data_root_resolved"] = True
    try:
        index = build_scared_c_index(root, strict=True)
    except (ValueError, OSError) as error:
        result["reason"] = f"SCARED-C sample index unavailable: {error}"
        return result
    record = index.records[0]
    try:
        sample = load_scared_c_sample(record)
    except (ValueError, OSError) as error:
        result["reason"] = f"SCARED-C sample decode failed: {error}"
        return result
    result["sample_id"] = record.sample_id
    result["sample_shape"] = list(sample.left.shape)
    if contract_gpu_smoke:
        result["gpu_smoke"] = _run_contract_gpu_smoke(sample)
    try:
        baseline = _baseline_config(raw, config_path.parent)
    except (KeyError, TypeError, ValueError, ReproductionRefusal) as error:
        return _refuse(
            result,
            config_path=config_path,
            raw=raw,
            dataset_config_path=dataset_config_path,
            dataset=dataset,
            index=index,
            record=record,
            sample=sample,
            reason=f"baseline configuration unavailable: {error}",
            artifact_path=artifact_path,
        )
    if (
        baseline.checkpoint_id is None
        or baseline.checkpoint_path is None
        or baseline.checkpoint_sha256 is None
    ):
        result["baseline_inference_capability"] = "BASELINE_INFERENCE_BLOCKED_ON_CHECKPOINT"
        return _refuse(
            result,
            config_path=config_path,
            raw=raw,
            dataset_config_path=dataset_config_path,
            dataset=dataset,
            index=index,
            record=record,
            sample=sample,
            reason="native inference refused: checkpoint identity/path/hash are not configured",
            artifact_path=artifact_path,
        )
    if input_bridge is None:
        result["baseline_inference_capability"] = "BASELINE_INFERENCE_BLOCKED_ON_INPUT_BRIDGE"
        return _refuse(
            result,
            config_path=config_path,
            raw=raw,
            dataset_config_path=dataset_config_path,
            dataset=dataset,
            index=index,
            record=record,
            sample=sample,
            reason=(
                "native inference refused: no authoritative SCARED-C to native-input bridge "
                "was supplied"
            ),
            artifact_path=artifact_path,
        )
    capabilities = inspect_capabilities(baseline.upstream_root)
    capability_record = {
        "native_inference_ready": capabilities.native_inference_ready,
        "cuda_baseline_runnable": capabilities.cuda_baseline_runnable,
        "missing_python_dependencies": list(capabilities.missing_python_dependencies),
    }
    result["capabilities"] = capability_record
    if not capabilities.cuda_baseline_runnable:
        result["baseline_inference_capability"] = "BASELINE_INFERENCE_BLOCKED_ON_RUNTIME"
        return _refuse(
            result,
            config_path=config_path,
            raw=raw,
            dataset_config_path=dataset_config_path,
            dataset=dataset,
            index=index,
            record=record,
            sample=sample,
            reason="native inference refused: pinned upstream/CUDA capabilities are unavailable",
            artifact_path=artifact_path,
            capabilities=capability_record,
        )
    native_input = input_bridge(sample)
    inference = NativeBaselineRunner(baseline).run(native_input)
    result.update(
        {
            "status": "completed",
            "native_output": True,
            "upstream_commit": inference.upstream_git_state.commit,
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/p0_baseline_reproduction.yaml")
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--artifact-path", type=Path, default=None)
    parser.add_argument(
        "--contract-gpu-smoke",
        action="store_true",
        help="run a Level 1 CUDA contract-tensor smoke on physical GPU 1",
    )
    args = parser.parse_args(argv)
    result = reproduce_baseline(
        args.config,
        data_root=args.data_root,
        artifact_path=args.artifact_path,
        contract_gpu_smoke=args.contract_gpu_smoke,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2
