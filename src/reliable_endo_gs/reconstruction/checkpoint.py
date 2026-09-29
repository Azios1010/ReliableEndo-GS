"""Reproducible A04 reference-head checkpoint and provenance persistence."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import cast

import torch

from reliable_endo_gs.reconstruction.config import ReferenceConfig, ReferenceConfigError
from reliable_endo_gs.reconstruction.model import (
    ReferenceModelError,
    ReferenceProvenance,
    ReferenceReconstructionModel,
    _canonical,
)
from reliable_endo_gs.utils.io import write_json_atomic

CHECKPOINT_SCHEMA = "a04_reference_checkpoint.v1"


class ReferenceCheckpointError(ValueError):
    """Raised when an A04 checkpoint or provenance file is invalid."""


def reference_checkpoint_path(
    config: ReferenceConfig,
    *,
    root: Path | None = None,
) -> Path:
    """Return a deterministic checkpoint path from a resolved config."""

    resolved_root = Path(root) if root is not None else config.checkpoint_root
    config_hash = config.config_hash
    model_id = config.model_id
    if resolved_root.is_absolute() and root is None:
        raise ReferenceCheckpointError("default checkpoint root must remain repository-relative")
    return resolved_root / f"{model_id}-{config_hash[:16]}.pt"


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in module.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def _cpu_tree(value: object) -> object:
    """Move optimizer/scaler state tensors to CPU for portable checkpoints."""

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def save_reference_checkpoint(
    path: Path | str,
    model: ReferenceReconstructionModel,
    *,
    provenance: ReferenceProvenance | None = None,
    optimizer_state: Mapping[str, object] | None = None,
    scaler_state: Mapping[str, object] | None = None,
    training_state: Mapping[str, object] | None = None,
) -> None:
    """Save the independent A04 head, identities, and optional resume state."""

    destination = Path(path)
    if destination.exists():
        raise ReferenceCheckpointError(f"refusing to overwrite existing checkpoint: {destination}")
    if not isinstance(model, ReferenceReconstructionModel):
        raise TypeError("model must be a ReferenceReconstructionModel")
    record = provenance
    if record is None:
        raise ReferenceCheckpointError(
            "provenance is required; run a reference forward before checkpointing"
        )
    if record.config_hash != model.config.config_hash:
        raise ReferenceCheckpointError("provenance config_hash does not match model config")
    try:
        runtime_teacher_identity = model._teacher_identity()
    except ReferenceModelError as error:
        raise ReferenceCheckpointError(str(error)) from error
    if _canonical(runtime_teacher_identity) != _canonical(record.teacher_identity):
        raise ReferenceCheckpointError(
            "checkpoint teacher identity does not match the injected runtime teacher"
        )
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "model_id": model.config.model_id,
        "config": model.config.to_dict(),
        "config_hash": model.config.config_hash,
        "provenance": record.to_dict(),
        "state_dict": _cpu_state(model),
        "optimizer_state": None if optimizer_state is None else _cpu_tree(optimizer_state),
        "scaler_state": None if scaler_state is None else _cpu_tree(scaler_state),
        "training_state": None if training_state is None else _cpu_tree(training_state),
        "teacher_in_checkpoint": False,
        "renderer_in_checkpoint": False,
    }
    _atomic_torch_save(destination, payload)


def _load_payload(path: Path) -> Mapping[str, object]:
    try:
        try:
            payload: object = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise ReferenceCheckpointError(f"unable to load reference checkpoint {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ReferenceCheckpointError("reference checkpoint must contain a mapping")
    return cast(Mapping[str, object], payload)


def _provenance_from_dict(value: object) -> ReferenceProvenance:
    if not isinstance(value, Mapping):
        raise ReferenceCheckpointError("checkpoint provenance must be a mapping")
    raw = dict(value)
    try:
        return ReferenceProvenance(
            teacher_identity=cast(Mapping[str, object], raw["teacher_identity"]),
            target_identity=cast(Mapping[str, object], raw["target_identity"]),
            alignment=cast(Mapping[str, object], raw["alignment"]),
            calibration=cast(Mapping[str, object], raw["calibration"]),
            data_role=str(raw["data_role"]),
            split_identity=str(raw["split_identity"]),
            target_source=str(raw["target_source"]),
            renderer_identity=str(raw["renderer_identity"]),
            config_hash=str(raw["config_hash"]),
            supervision_views=tuple(str(item) for item in raw["supervision_views"]),
            assessment_views=tuple(str(item) for item in raw["assessment_views"]),
            dataset_name=str(raw.get("dataset_name", "scared")),
            active_dataset_ids=tuple(str(item) for item in raw.get("active_dataset_ids", ())),
            excluded_dataset_names=tuple(
                str(item) for item in raw.get("excluded_dataset_names", ("SCARED-C",))
            ),
            schema_version=str(raw.get("schema_version", "a04_reference_provenance.v1")),
            geometry_schema=str(raw.get("geometry_schema", "a03_geometry_prior.v1")),
            composition_schema=str(raw.get("composition_schema", "a04_bounded_composition.v1")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReferenceCheckpointError(f"invalid checkpoint provenance: {error}") from error


def load_reference_checkpoint(
    path: Path | str,
    model: ReferenceReconstructionModel,
    *,
    strict: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: object | None = None,
    training_state: MutableMapping[str, object] | None = None,
) -> ReferenceProvenance:
    """Load a validated head/checkpoint state without loading teacher weights."""

    if not isinstance(model, ReferenceReconstructionModel):
        raise TypeError("model must be a ReferenceReconstructionModel")
    source = Path(path)
    payload = _load_payload(source)
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ReferenceCheckpointError("unsupported A04 checkpoint schema")
    if payload.get("model_id") != model.config.model_id:
        raise ReferenceCheckpointError("checkpoint model_id does not match model config")
    if payload.get("config_hash") != model.config.config_hash:
        raise ReferenceCheckpointError("checkpoint config_hash does not match model config")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ReferenceCheckpointError("checkpoint state_dict is missing")
    state_dict = {
        str(name): value
        for name, value in state.items()
        if isinstance(name, str) and isinstance(value, torch.Tensor)
    }
    if len(state_dict) != len(state):
        raise ReferenceCheckpointError("checkpoint state_dict contains non-tensor values")
    try:
        model.load_state_dict(state_dict, strict=strict)
    except (RuntimeError, ValueError) as error:
        raise ReferenceCheckpointError(f"reference head state does not match model: {error}") from error
    record = _provenance_from_dict(payload.get("provenance"))
    if record.config_hash != model.config.config_hash:
        raise ReferenceCheckpointError("checkpoint provenance config_hash does not match model config")
    try:
        runtime_teacher_identity = model._teacher_identity()
    except ReferenceModelError as error:
        raise ReferenceCheckpointError(str(error)) from error
    if _canonical(runtime_teacher_identity) != _canonical(record.teacher_identity):
        raise ReferenceCheckpointError(
            "checkpoint teacher identity does not match the injected runtime teacher"
        )
    if payload.get("teacher_in_checkpoint") is not False:
        raise ReferenceCheckpointError("teacher weights must remain external to A04 checkpoints")
    if payload.get("renderer_in_checkpoint") is not False:
        raise ReferenceCheckpointError("renderer state must remain external to A04 checkpoints")
    if optimizer is not None:
        optimizer_state = payload.get("optimizer_state")
        if isinstance(optimizer_state, Mapping):
            try:
                optimizer.load_state_dict(optimizer_state)
            except (RuntimeError, ValueError, TypeError) as error:
                raise ReferenceCheckpointError(
                    f"reference optimizer state does not match model: {error}"
                ) from error
    if scaler is not None:
        scaler_state = payload.get("scaler_state")
        load_state_dict = getattr(scaler, "load_state_dict", None)
        if isinstance(scaler_state, Mapping) and callable(load_state_dict):
            try:
                load_state_dict(scaler_state)
            except (RuntimeError, ValueError, TypeError) as error:
                raise ReferenceCheckpointError(
                    f"reference AMP scaler state is invalid: {error}"
                ) from error
    if training_state is not None:
        raw_training_state = payload.get("training_state")
        if isinstance(raw_training_state, Mapping):
            training_state.update(raw_training_state)
    return record


def write_reference_provenance(path: Path | str, provenance: ReferenceProvenance) -> None:
    """Write a standalone provenance sidecar and refuse accidental replacement."""

    destination = Path(path)
    if destination.exists():
        raise ReferenceConfigError(f"refusing to overwrite existing provenance: {destination}")
    write_json_atomic(destination, provenance.to_dict())


__all__ = [
    "CHECKPOINT_SCHEMA",
    "ReferenceCheckpointError",
    "load_reference_checkpoint",
    "reference_checkpoint_path",
    "save_reference_checkpoint",
    "write_reference_provenance",
]
