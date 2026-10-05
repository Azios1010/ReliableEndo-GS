"""Explicit, lazy bindings to external FoundationStereo and DEFOM-Stereo.

Only RGB tensors cross the inference boundary. External source/checkpoints are
never imported or deserialized until ``infer``/``load`` is explicitly requested.
Both upstream projects own a top-level ``core`` package: use separate processes
and put exactly one source root on PYTHONPATH before starting Python.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from reliable_endo_gs.teachers.base import (
    CandidateAvailability,
    LazyTeacherAdapter,
    TeacherContractError,
    TeacherPrediction,
    TeacherUnavailableError,
)
from reliable_endo_gs.teachers.candidates import (
    make_defom_stereo_candidate,
    make_foundation_stereo_candidate,
    sha256_file,
    teacher_config_hash,
)

_MODELS = {
    "foundation_stereo": ("core.foundation_stereo", "FoundationStereo"),
    "defom_stereo": ("core.defom_stereo", "DEFOMStereo"),
}


def _path(value: object, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TeacherContractError(f"{name} must be a configured path")
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise TeacherContractError(f"{name} contains an unresolved environment variable: {value}")
    path = Path(expanded).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _digest(value: object, name: str, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(char not in "0123456789abcdefABCDEF" for char in value)
    ):
        raise TeacherContractError(f"{name} must be a pinned {length}-character hex digest")
    return value.lower()


def source_tree_sha256(root: Path) -> str:
    """Hash relative names and contents of all external Python source files.

    Excludes .git and __pycache__; includes vendored Python code. License and
    architecture YAML are separately pinned. No data/model files are scanned.
    """
    files = sorted(
        path
        for path in root.rglob("*.py")
        if not {".git", "__pycache__"}.intersection(path.relative_to(root).parts)
    )
    if not files:
        raise TeacherContractError(f"no Python source found in {root}")
    records = [(path.relative_to(root).as_posix(), sha256_file(path)) for path in files]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ExternalTeacherConfig:
    """Resolved configuration; all model artifacts remain external."""

    model: str
    source_root: Path
    source_revision: str
    source_sha256: str
    checkpoint: Path
    checkpoint_sha256: str
    license_file: Path
    license_sha256: str
    architecture: Path
    architecture_sha256: str
    valid_iters: int
    precision: str
    mode: str
    small_ratio: float
    scale_iters: int
    checkpoint_format: str

    def identity_config(self) -> dict[str, object]:
        # Absolute mount paths do not affect portable scientific identity.
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
            if key not in {"source_root", "checkpoint", "license_file", "architecture"}
        }


def load_external_teacher_config(config_path: Path | str | None = None) -> ExternalTeacherConfig:
    """Parse YAML from an explicit path or RELIABLE_ENDO_TEACHER_CONFIG."""
    configured = config_path or os.environ.get("RELIABLE_ENDO_TEACHER_CONFIG")
    if not configured:
        raise TeacherContractError("set RELIABLE_ENDO_TEACHER_CONFIG to a pinned teacher YAML")
    path = Path(configured).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as error:
            raise TeacherContractError(f"invalid teacher YAML: {error}") from error
    if (
        not isinstance(raw, dict)
        or type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != 1
    ):
        raise TeacherContractError("external teacher config must have schema_version: 1")
    if not all(isinstance(key, str) for key in raw):
        raise TeacherContractError("teacher config keys must be strings")
    fields = set(ExternalTeacherConfig.__dataclass_fields__)
    if set(raw) - fields - {"schema_version"}:
        raise TeacherContractError(
            f"unknown teacher config fields: {sorted(set(raw) - fields - {'schema_version'})}"
        )
    if fields - set(raw):
        raise TeacherContractError(f"missing teacher config fields: {sorted(fields - set(raw))}")
    values = {key: raw[key] for key in fields}
    for key in ("source_root", "checkpoint", "license_file", "architecture"):
        values[key] = _path(values[key], path.parent, key)
    for key in ("source_sha256", "checkpoint_sha256", "license_sha256", "architecture_sha256"):
        values[key] = _digest(values[key], key)
    values["source_revision"] = _digest(values["source_revision"], "source_revision", 40)
    if not isinstance(values["model"], str) or values["model"] not in _MODELS:
        raise TeacherContractError("model must be foundation_stereo or defom_stereo")
    for key in ("valid_iters", "scale_iters"):
        if type(values[key]) is not int or values[key] < 1:
            raise TeacherContractError(f"{key} must be a positive integer")
    if not isinstance(values["precision"], str) or values["precision"] not in {
        "float32",
        "float16",
    }:
        raise TeacherContractError("precision must be float32 or float16")
    if not isinstance(values["mode"], str) or values["mode"] not in {"regular", "hierarchical"}:
        raise TeacherContractError("mode must be regular or hierarchical")
    if values["model"] == "defom_stereo" and values["mode"] != "regular":
        raise TeacherContractError("DEFOM-Stereo does not expose hierarchical inference")
    if (
        not isinstance(values["small_ratio"], (int, float))
        or isinstance(values["small_ratio"], bool)
        or not 0 < values["small_ratio"] < 1
    ):
        raise TeacherContractError("small_ratio must be in (0,1)")
    if not isinstance(values["checkpoint_format"], str) or values["checkpoint_format"] not in {
        "model",
        "state_dict",
    }:
        raise TeacherContractError(
            "checkpoint_format must be model or state_dict; no format guessing"
        )
    return ExternalTeacherConfig(**values)


def _artifact_identity(config: ExternalTeacherConfig) -> dict[str, str]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(config.source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except subprocess.SubprocessError as error:
        raise TeacherContractError(
            f"cannot verify teacher Git HEAD at {config.source_root}: {error}"
        ) from error
    return {
        "source_revision": revision,
        "source_sha256": source_tree_sha256(config.source_root),
        "checkpoint_sha256": sha256_file(config.checkpoint),
        "license_sha256": sha256_file(config.license_file),
        "architecture_sha256": sha256_file(config.architecture),
    }


def _verify_artifacts(config: ExternalTeacherConfig) -> dict[str, str]:
    actual = _artifact_identity(config)
    for key, value in actual.items():
        if value != getattr(config, key):
            raise TeacherContractError(
                f"{key} mismatch: expected {getattr(config, key)}, actual {value}"
            )
    return actual


def _check_core_origin(root: Path) -> None:
    # Check the already-loaded namespace too: two teachers cannot coexist here.
    modules = [
        module for name, module in sys.modules.items() if name == "core" or name.startswith("core.")
    ]
    for module in modules:
        origin = getattr(module, "__file__", None)
        locations = getattr(module, "__path__", ())
        paths = [Path(origin).resolve()] if origin else [Path(item).resolve() for item in locations]
        if not paths or any(not path.is_relative_to(root) for path in paths):
            raise TeacherUnavailableError(
                "core namespace belongs to another project; start a separate teacher process"
            )
    spec = importlib.util.find_spec("core")
    namespace_locations = [] if spec is None else list(spec.submodule_search_locations or ())
    if not namespace_locations or any(
        not Path(path).resolve().is_relative_to(root) for path in namespace_locations
    ):
        raise TeacherUnavailableError(f"put only this teacher source root on PYTHONPATH: {root}")


def teacher_preflight(config_path: Path | str | None = None) -> dict[str, object]:
    """Verify pins/import namespace without importing a model or using CUDA.

    This is artifact preflight, not proof that native dependencies or upstream
    constructor caches are ready. No checkpoint is deserialized.
    """
    config = load_external_teacher_config(config_path)
    actual = _verify_artifacts(config)
    _check_core_origin(config.source_root)
    dependencies = (
        ("omegaconf", "timm", "einops")
        if config.model == "foundation_stereo"
        else ("timm", "einops")
    )
    missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
    return {
        "model": config.model,
        "artifacts_verified": True,
        "model_loaded": False,
        "missing_dependencies": missing,
        "runtime_verified": False,
        "identity": actual,
        "config_hash": teacher_config_hash(config.identity_config()),
        "source_root": str(config.source_root),
        "checkpoint": str(config.checkpoint),
    }


def _foundation_metadata_globals(names: set[str]) -> list[Any]:
    """Allow only reviewed training metadata present in the official checkpoint.

    Resolve an explicit list, never a checkpoint-supplied import path. NumPy's
    scalar dtype classes are built dynamically and omitted by PyTorch's scan.
    These globals are scoped to a weights-only read; metadata is then discarded.
    """
    supported = {
        "builtins.int", "builtins.dict", "builtins.list", "collections.defaultdict",
        "typing.Any", "numpy.dtype", "numpy.core.multiarray.scalar",
        "numpy._core.multiarray.scalar", "omegaconf.base.ContainerMetadata",
        "omegaconf.base.Metadata", "omegaconf.dictconfig.DictConfig",
        "omegaconf.listconfig.ListConfig", "omegaconf.nodes.AnyNode",
    }
    unknown = names - supported
    if unknown:
        raise TeacherContractError(f"unsupported FoundationStereo checkpoint globals: {sorted(unknown)}")
    known: dict[str, Any] = {
        "builtins.int": int, "builtins.dict": dict, "builtins.list": list,
        "collections.defaultdict": defaultdict, "typing.Any": Any,
    }
    dynamic: list[Any] = []
    if any(name.startswith("numpy.") for name in names):
        numpy = importlib.import_module("numpy")
        scalar_module = "numpy._core.multiarray" if hasattr(numpy, "_core") else "numpy.core.multiarray"
        scalar = importlib.import_module(scalar_module).scalar
        known.update({
            "numpy.dtype": numpy.dtype,
            "numpy.core.multiarray.scalar": (scalar, "numpy.core.multiarray.scalar"),
            "numpy._core.multiarray.scalar": (scalar, "numpy._core.multiarray.scalar"),
        })
        dynamic = [type(numpy.dtype(numpy.float32)), type(numpy.dtype(numpy.float64))]
    if any(name.startswith("omegaconf.") for name in names):
        for module_name, class_names in (
            ("omegaconf.base", ("ContainerMetadata", "Metadata")),
            ("omegaconf.dictconfig", ("DictConfig",)),
            ("omegaconf.listconfig", ("ListConfig",)),
            ("omegaconf.nodes", ("AnyNode",)),
        ):
            module = importlib.import_module(module_name)
            known.update({f"{module_name}.{name}": getattr(module, name) for name in class_names})
    return [known[name] for name in sorted(names)] + dynamic


def _checkpoint_state(config: ExternalTeacherConfig) -> Mapping[str, torch.Tensor]:
    """Read CPU parameter tensors, excluding optimizer and training metadata.

    Tensor shapes are the checkpoint's individual parameter shapes. ZIP-backed
    tensors are memory mapped to avoid allocating a full training checkpoint.
    Legacy tensor-only files retain their existing weights-only read behavior.
    """
    is_zip = zipfile.is_zipfile(config.checkpoint)
    metadata: list[Any] = []
    if config.model == "foundation_stereo" and config.checkpoint_format == "model" and is_zip:
        names = set(torch.serialization.get_unsafe_globals_in_checkpoint(config.checkpoint))
        metadata = _foundation_metadata_globals(names)
        existing = torch.serialization.get_safe_globals()
        metadata = [item for item in metadata if item not in existing]
    with torch.serialization.safe_globals(metadata):
        checkpoint = torch.load(
            config.checkpoint, map_location="cpu", weights_only=True, mmap=is_zip
        )
    state = (
        checkpoint.get("model")
        if config.checkpoint_format == "model" and isinstance(checkpoint, Mapping)
        else checkpoint
    )
    if (
        not isinstance(state, Mapping)
        or not state
        or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items())
    ):
        raise TeacherContractError("checkpoint does not match the declared state-dict format")
    return cast(Mapping[str, torch.Tensor], state)


def _build(config_path: Path | str | None, expected_model: str) -> LazyTeacherAdapter:
    config = load_external_teacher_config(config_path)
    if config.model != expected_model:
        raise TeacherContractError(f"factory expects {expected_model}, got {config.model}")
    _verify_artifacts(config)
    with config.architecture.open(encoding="utf-8") as handle:
        try:
            architecture = yaml.safe_load(handle)
        except yaml.YAMLError as error:
            raise TeacherContractError(f"invalid architecture YAML: {error}") from error
    if not isinstance(architecture, dict) or not architecture:
        raise TeacherContractError("architecture YAML must be a non-empty mapping")
    if "mixed_precision" in architecture and type(architecture["mixed_precision"]) is not bool:
        raise TeacherContractError("architecture mixed_precision must be boolean")
    if "mixed_precision" in architecture and architecture["mixed_precision"] != (
        config.precision == "float16"
    ):
        raise TeacherContractError("architecture mixed_precision disagrees with runtime precision")
    architecture = dict(architecture, mixed_precision=config.precision == "float16")

    def load() -> torch.nn.Module:
        _verify_artifacts(config)  # Reject artifacts changed after factory construction.
        _check_core_origin(config.source_root)
        state = _checkpoint_state(config)  # Reject checkpoint errors before constructor downloads.
        module_name, class_name = _MODELS[config.model]
        module = importlib.import_module(module_name)
        module_origin = module.__file__
        if module_origin is None or not Path(module_origin).resolve().is_relative_to(
            config.source_root
        ):
            raise TeacherUnavailableError("teacher model module is outside the pinned source root")
        if config.model == "foundation_stereo":
            omega = importlib.import_module("omegaconf").OmegaConf
            args = omega.create(architecture)
            if "vit_size" not in args:
                args["vit_size"] = "vitl"  # Official demo's explicit compatibility default.
        else:
            args = argparse.Namespace(**architecture)
        model = getattr(module, class_name)(args)
        if not isinstance(model, torch.nn.Module):
            raise TeacherContractError("upstream model must be a torch.nn.Module")
        model.load_state_dict(state, strict=True)
        return model

    def infer(model: object, left: torch.Tensor, right: torch.Tensor) -> TeacherPrediction:
        # Inputs [B,3,H,W], A00 RGB floats [0,1]; native RGB floats [0,255].
        if not isinstance(model, torch.nn.Module):
            raise TeacherContractError("upstream model must be a torch.nn.Module")
        for image in (left, right):
            if not torch.isfinite(image).all() or (image < 0).any() or (image > 1).any():
                raise TeacherContractError("teacher input must be finite A00 RGB in [0,1]")
        if config.precision == "float16" and left.device.type != "cuda":
            raise TeacherContractError("float16 teacher inference requires an explicit CUDA device")
        utils = importlib.import_module("core.utils.utils")
        padder = utils.InputPadder(left.shape, divis_by=32)
        padded_left, padded_right = padder.pad(
            left.float().contiguous() * 255, right.float().contiguous() * 255
        )
        kwargs: dict[str, Any] = {"iters": config.valid_iters, "test_mode": True}
        method: Callable[..., object] = model
        if config.model == "foundation_stereo" and config.mode == "hierarchical":
            hierarchical = getattr(model, "run_hierachical", None)  # Upstream API spelling.
            if not callable(hierarchical):
                raise TeacherContractError("this source does not expose run_hierachical")
            method = hierarchical
            kwargs["small_ratio"] = config.small_ratio
        elif config.model == "defom_stereo":
            kwargs["scale_iters"] = config.scale_iters
        with torch.autocast(
            device_type=left.device.type, dtype=torch.float16, enabled=config.precision == "float16"
        ):
            disparity = method(padded_left, padded_right, **kwargs)
        expected = (left.shape[0], 1, *padded_left.shape[-2:])
        if (
            not isinstance(disparity, torch.Tensor)
            or tuple(disparity.shape) != expected
            or not disparity.is_floating_point()
        ):
            raise TeacherContractError(
                f"upstream disparity must be a floating tensor with padded shape {expected}"
            )
        disparity = padder.unpad(disparity).float()
        if tuple(disparity.shape) != (left.shape[0], 1, *left.shape[-2:]):
            raise TeacherContractError("unpadding changed the A00 output grid")
        return TeacherPrediction(
            disparity=disparity,
            validity=torch.isfinite(disparity) & (disparity > 0),
            support=torch.ones_like(disparity, dtype=torch.bool),
            metadata={
                "input_range": "0..1",
                "native_input_range": "0..255",
                "padding_divisor": 32,
                "resize": None,
                "mode": config.mode,
            },
        )

    factory = (
        make_foundation_stereo_candidate
        if config.model == "foundation_stereo"
        else make_defom_stereo_candidate
    )
    return factory(
        loader=load,
        infer_fn=infer,
        availability_probe=lambda: _probe(config),
        api_version=f"{config.source_revision}:first-party-external-v1:{sha256_file(Path(__file__))}",
        checkpoint_id=config.checkpoint_sha256,
        checkpoint_sha256=config.checkpoint_sha256,
        license_name=f"external-license-sha256:{config.license_sha256}",
        preprocessing={
            "status": "configured",
            "image_range": "A00_0..1_to_native_0..255",
            "padding": "upstream_InputPadder_div32_unpad",
            "resize": None,
        },
        output={
            "status": "configured",
            "disparity_units": "teacher_grid_pixels",
            "disparity_sign": "u_left_minus_right",
            "confidence": None,
        },
        source=f"{config.source_revision}:python-sha256:{config.source_sha256}",
        config_hash=teacher_config_hash(config.identity_config()),
    )


def _probe(config: ExternalTeacherConfig) -> CandidateAvailability:
    _verify_artifacts(config)
    _check_core_origin(config.source_root)
    return CandidateAvailability(
        "FoundationStereo" if config.model == "foundation_stereo" else "DEFOM-Stereo",
        available=True,
        checked=False,
        reason="artifacts configured; model/native dependencies and constructor caches are not runtime-verified",
    )


def build_foundation_stereo(config_path: Path | str | None = None) -> LazyTeacherAdapter:
    """Build the primary frozen candidate, without importing/loading its model."""
    return _build(config_path, "foundation_stereo")


def build_defom_stereo(config_path: Path | str | None = None) -> LazyTeacherAdapter:
    """Build the separately identified frozen comparator, without model loading."""
    return _build(config_path, "defom_stereo")
