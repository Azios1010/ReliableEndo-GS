"""Versioned A04 quality-reference configuration and round-trip helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import yaml

from reliable_endo_gs.config.deployment import DeploymentRuntimeProfile
from reliable_endo_gs.gaussians import GeometryPriorPolicy, SamplingPolicy
from reliable_endo_gs.gaussians.residual_head import ResidualBounds
from reliable_endo_gs.geometry.surface import SurfaceFramePolicy
from reliable_endo_gs.teachers import A01_DATASET_IDS, DEFAULT_TARGET_SCHEMA
from reliable_endo_gs.teachers.roles import DEFAULT_A01_SCARED_ROLES
from reliable_endo_gs.utils.io import write_yaml_atomic


class ReferenceConfigError(ValueError):
    """Raised when an A04 reference configuration is unsafe or incomplete."""


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ReferenceConfigError(f"configuration value {type(value).__name__} is not JSON-compatible")


def _surface_policy_to_dict(policy: SurfaceFramePolicy) -> dict[str, float | int]:
    return {
        "neighborhood_radius": policy.neighborhood_radius,
        "depth_discontinuity_abs_m": float(policy.depth_discontinuity_abs_m),
        "depth_discontinuity_rel": float(policy.depth_discontinuity_rel),
        "min_neighbor_count": policy.min_neighbor_count,
        "min_cross_area_m2": float(policy.min_cross_area_m2),
        "tangent_epsilon": float(policy.tangent_epsilon),
        "scale_factor": float(policy.scale_factor),
        "min_tangent_scale_m": float(policy.min_tangent_scale_m),
        "max_tangent_scale_m": float(policy.max_tangent_scale_m),
        "normal_thickness_m": float(policy.normal_thickness_m),
        "max_normal_thickness_m": float(policy.max_normal_thickness_m),
        "fallback_scale_m": float(policy.fallback_scale_m),
    }


def _prior_policy_to_dict(policy: GeometryPriorPolicy) -> dict[str, object]:
    sampling = policy.sampling
    return {
        "alpha_valid": float(policy.alpha_valid),
        "sampling": {
            "mode": sampling.mode,
            "stride": sampling.stride,
            "cap": sampling.cap,
            "max_samples": sampling.max_samples,
            "offset_xy": list(sampling.offset_xy),
        },
        "surface": _surface_policy_to_dict(policy.surface),
    }


def _runtime_to_dict(profile: DeploymentRuntimeProfile) -> dict[str, object]:
    return {
        "hardware": profile.hardware,
        "device": profile.device,
        "memory_gib": profile.memory_gib,
        "batch_size": profile.batch_size,
        "precision": profile.precision,
        "geometry_precision": profile.geometry_precision,
        "resolution_hw": None if profile.resolution_hw is None else list(profile.resolution_hw),
        "deadline_ms": profile.deadline_ms,
        "warmup": profile.warmup,
        "repetitions": profile.repetitions,
        "synchronize": profile.synchronize,
        "compilation": profile.compilation,
    }


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ReferenceConfigError(f"{name} must be a string-keyed mapping")
    return dict(cast(Mapping[str, object], value))


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ReferenceConfigError(f"{name} must be a sequence")
    return tuple(value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceConfigError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ReferenceConfigError(f"{name} must be an integer")
    try:
        return int(value)
    except ValueError as error:
        raise ReferenceConfigError(f"{name} must be an integer") from error


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int, str)):
        raise ReferenceConfigError(f"{name} must be numeric")
    try:
        return float(value)
    except ValueError as error:
        raise ReferenceConfigError(f"{name} must be numeric") from error


def _optional_integer(value: object, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _integer_pair(value: object, name: str) -> tuple[int, int]:
    values = _sequence(value, name)
    if len(values) != 2:
        raise ReferenceConfigError(f"{name} must contain exactly two integers")
    return _integer(values[0], f"{name}[0]"), _integer(values[1], f"{name}[1]")


def _prior_policy_from_dict(value: object) -> GeometryPriorPolicy:
    raw = _mapping(value, "prior_policy")
    unexpected = set(raw) - {"alpha_valid", "sampling", "surface"}
    if unexpected:
        raise ReferenceConfigError(
            f"prior_policy contains unsupported field(s): {', '.join(sorted(unexpected))}"
        )
    sampling_raw = _mapping(raw.get("sampling", {}), "prior_policy.sampling")
    surface_raw = _mapping(raw.get("surface", {}), "prior_policy.surface")
    sampling_unexpected = set(sampling_raw) - {"mode", "stride", "cap", "max_samples", "offset_xy"}
    if sampling_unexpected:
        raise ReferenceConfigError(
            "prior_policy.sampling contains unsupported field(s): "
            + ", ".join(sorted(sampling_unexpected))
        )
    surface_fields = {
        "neighborhood_radius",
        "depth_discontinuity_abs_m",
        "depth_discontinuity_rel",
        "min_neighbor_count",
        "min_cross_area_m2",
        "tangent_epsilon",
        "scale_factor",
        "min_tangent_scale_m",
        "max_tangent_scale_m",
        "normal_thickness_m",
        "max_normal_thickness_m",
        "fallback_scale_m",
    }
    surface_unexpected = set(surface_raw) - surface_fields
    if surface_unexpected:
        raise ReferenceConfigError(
            "prior_policy.surface contains unsupported field(s): "
            + ", ".join(sorted(surface_unexpected))
        )
    mode = _string(sampling_raw.get("mode", "dense"), "prior_policy.sampling.mode")
    if mode not in ("dense", "fixed", "sparse"):
        raise ReferenceConfigError("prior_policy.sampling.mode must be dense, fixed, or sparse")
    sampling = SamplingPolicy(
        mode=cast(Literal["dense", "fixed", "sparse"], mode),
        stride=_integer(sampling_raw.get("stride", 1), "prior_policy.sampling.stride"),
        cap=_optional_integer(sampling_raw.get("cap"), "prior_policy.sampling.cap"),
        max_samples=_optional_integer(
            sampling_raw.get("max_samples"), "prior_policy.sampling.max_samples"
        ),
        offset_xy=_integer_pair(
            sampling_raw.get("offset_xy", (0, 0)), "prior_policy.sampling.offset_xy"
        ),
    )
    surface = SurfaceFramePolicy(
        neighborhood_radius=_integer(
            surface_raw.get("neighborhood_radius", 1), "surface.neighborhood_radius"
        ),
        depth_discontinuity_abs_m=_number(
            surface_raw.get("depth_discontinuity_abs_m", 0.01), "surface.depth_discontinuity_abs_m"
        ),
        depth_discontinuity_rel=_number(
            surface_raw.get("depth_discontinuity_rel", 0.05), "surface.depth_discontinuity_rel"
        ),
        min_neighbor_count=_integer(
            surface_raw.get("min_neighbor_count", 2), "surface.min_neighbor_count"
        ),
        min_cross_area_m2=_number(
            surface_raw.get("min_cross_area_m2", 1.0e-10), "surface.min_cross_area_m2"
        ),
        tangent_epsilon=_number(
            surface_raw.get("tangent_epsilon", 1.0e-5), "surface.tangent_epsilon"
        ),
        scale_factor=_number(surface_raw.get("scale_factor", 0.5), "surface.scale_factor"),
        min_tangent_scale_m=_number(
            surface_raw.get("min_tangent_scale_m", 1.0e-4), "surface.min_tangent_scale_m"
        ),
        max_tangent_scale_m=_number(
            surface_raw.get("max_tangent_scale_m", 0.25), "surface.max_tangent_scale_m"
        ),
        normal_thickness_m=_number(
            surface_raw.get("normal_thickness_m", 1.0e-3), "surface.normal_thickness_m"
        ),
        max_normal_thickness_m=_number(
            surface_raw.get("max_normal_thickness_m", 0.25), "surface.max_normal_thickness_m"
        ),
        fallback_scale_m=_number(
            surface_raw.get("fallback_scale_m", 5.0e-3), "surface.fallback_scale_m"
        ),
    )
    return GeometryPriorPolicy(
        surface=surface,
        sampling=sampling,
        alpha_valid=_number(raw.get("alpha_valid", 1.0), "prior_policy.alpha_valid"),
    )


def _runtime_from_dict(value: object) -> DeploymentRuntimeProfile:
    raw = _mapping(value, "runtime")
    runtime_fields = {
        "hardware",
        "device",
        "memory_gib",
        "batch_size",
        "precision",
        "geometry_precision",
        "resolution_hw",
        "deadline_ms",
        "warmup",
        "repetitions",
        "synchronize",
        "compilation",
    }
    unexpected = set(raw) - runtime_fields
    if unexpected:
        raise ReferenceConfigError(
            "runtime contains unsupported field(s): " + ", ".join(sorted(unexpected))
        )
    resolution = raw.get("resolution_hw")
    resolution_hw = (
        None if resolution is None else _integer_pair(resolution, "runtime.resolution_hw")
    )
    return DeploymentRuntimeProfile(
        hardware=_string(raw.get("hardware", "NVIDIA RTX 5070 Ti 16 GB"), "runtime.hardware"),
        device=_string(raw.get("device", "cuda:0"), "runtime.device"),
        memory_gib=_integer(raw.get("memory_gib", 16), "runtime.memory_gib"),
        batch_size=_integer(raw.get("batch_size", 1), "runtime.batch_size"),
        precision=_string(raw.get("precision", "fp16"), "runtime.precision"),
        geometry_precision=_string(
            raw.get("geometry_precision", "fp32"), "runtime.geometry_precision"
        ),
        resolution_hw=resolution_hw,
        deadline_ms=(
            None
            if raw.get("deadline_ms") is None
            else _number(raw["deadline_ms"], "runtime.deadline_ms")
        ),
        warmup=_integer(raw.get("warmup", 20), "runtime.warmup"),
        repetitions=_integer(raw.get("repetitions", 1000), "runtime.repetitions"),
        synchronize=bool(raw.get("synchronize", True)),
        compilation=_string(raw.get("compilation", "eager"), "runtime.compilation"),
    )


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    """Resolved A04 reference settings with explicit experiment identities."""

    schema_version: int = 1
    model_id: str = "a04_teacher_backed_reference_v1"
    stage: str = "reference"
    quality_first: bool = True
    latency_optimization: bool = False
    data_role: str = "teacher_fit"
    split_identity: str = "a01-scared-dataset1-7-v1"
    active_dataset_ids: tuple[str, ...] = A01_DATASET_IDS
    excluded_dataset_names: tuple[str, ...] = ("SCARED-C",)
    teacher_identity: Mapping[str, object] = field(
        default_factory=lambda: {"status": "injected_at_runtime"}
    )
    target_schema_hash: str = DEFAULT_TARGET_SCHEMA.schema_hash
    alignment_identity: str = "a01-aligned-target-v1"
    calibration_identity: str = "explicit-calibration-at-forward"
    renderer_identity: str = "injected-field-renderer"
    supervision_views: tuple[str, ...] = ("left",)
    assessment_views: tuple[str, ...] = ("left", "right")
    context_scale_m: float = 1.0
    head_input_dim: int = 13
    head_hidden_dims: tuple[int, ...] = (128, 128)
    residual_bounds: ResidualBounds = field(default_factory=ResidualBounds)
    prior_policy: GeometryPriorPolicy = field(default_factory=GeometryPriorPolicy)
    runtime: DeploymentRuntimeProfile = field(default_factory=DeploymentRuntimeProfile)
    seed: int = 0
    checkpoint_root: Path = Path("artifacts/a04_reference")

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ReferenceConfigError("unsupported A04 reference config schema")
        for name in (
            "model_id",
            "stage",
            "split_identity",
            "alignment_identity",
            "calibration_identity",
            "renderer_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ReferenceConfigError(f"{name} must be a non-empty string")
        if self.stage != "reference":
            raise ReferenceConfigError("A04 config must describe the quality reference stage")
        if self.quality_first is not True or self.latency_optimization is not False:
            raise ReferenceConfigError(
                "A04 reference requires quality_first and disables latency optimization"
            )
        try:
            DEFAULT_A01_SCARED_ROLES.datasets_for_role(self.data_role)
        except ValueError as error:
            raise ReferenceConfigError(str(error)) from error
        if tuple(self.active_dataset_ids) != A01_DATASET_IDS:
            raise ReferenceConfigError(
                "active_dataset_ids must be exactly SCARED dataset_1 through dataset_7"
            )
        if not self.excluded_dataset_names or "SCARED-C" not in self.excluded_dataset_names:
            raise ReferenceConfigError("SCARED-C must remain explicitly excluded")
        if not isinstance(self.teacher_identity, Mapping):
            raise ReferenceConfigError("teacher_identity must be a mapping")
        if len(self.target_schema_hash) != 64:
            raise ReferenceConfigError("target_schema_hash must be a SHA-256 identity")
        for name, views in (
            ("supervision_views", self.supervision_views),
            ("assessment_views", self.assessment_views),
        ):
            if not views or any(view not in ("left", "right") for view in views):
                raise ReferenceConfigError(f"{name} must contain left/right view names")
        if any(view not in self.assessment_views for view in self.supervision_views):
            raise ReferenceConfigError("supervision views must be included in assessment_views")
        if "left" not in self.assessment_views:
            raise ReferenceConfigError("assessment_views must include the left reconstruction view")
        if not any(view not in self.supervision_views for view in self.assessment_views):
            raise ReferenceConfigError(
                "assessment_views must include a view held out from supervision_views"
            )
        if self.head_input_dim != 13:
            raise ReferenceConfigError("A04 local RGB/geometry context has exactly 13 channels")
        if not self.head_hidden_dims or any(width < 1 for width in self.head_hidden_dims):
            raise ReferenceConfigError("head_hidden_dims must contain positive widths")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ReferenceConfigError("seed must be a non-negative integer")
        if self.runtime.resolution_hw is None or self.runtime.deadline_ms is None:
            raise ReferenceConfigError(
                "A04 runtime requires explicit resolution_hw and deadline_ms resource limits"
            )
        if self.runtime.batch_size != 1:
            raise ReferenceConfigError("A04 server execution currently requires batch_size=1")
        if self.prior_policy.sampling.effective_cap is None:
            raise ReferenceConfigError(
                "A04 sampling requires a finite cap/max_samples for the 16 GiB deployment target"
            )
        if self.checkpoint_root.is_absolute() or ".." in self.checkpoint_root.parts:
            raise ReferenceConfigError(
                "checkpoint_root must be a portable repository-relative path"
            )
        object.__setattr__(self, "active_dataset_ids", tuple(self.active_dataset_ids))
        object.__setattr__(self, "excluded_dataset_names", tuple(self.excluded_dataset_names))
        object.__setattr__(self, "supervision_views", tuple(self.supervision_views))
        object.__setattr__(self, "assessment_views", tuple(self.assessment_views))
        object.__setattr__(self, "head_hidden_dims", tuple(self.head_hidden_dims))
        object.__setattr__(self, "teacher_identity", dict(self.teacher_identity))

    def to_dict(self) -> dict[str, object]:
        """Return the fully resolved configuration mapping."""

        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "stage": self.stage,
            "quality_first": self.quality_first,
            "latency_optimization": self.latency_optimization,
            "data_role": self.data_role,
            "split_identity": self.split_identity,
            "active_dataset_ids": list(self.active_dataset_ids),
            "excluded_dataset_names": list(self.excluded_dataset_names),
            "teacher_identity": _canonical(self.teacher_identity),
            "target_schema_hash": self.target_schema_hash,
            "alignment_identity": self.alignment_identity,
            "calibration_identity": self.calibration_identity,
            "renderer_identity": self.renderer_identity,
            "supervision_views": list(self.supervision_views),
            "assessment_views": list(self.assessment_views),
            "context_scale_m": float(self.context_scale_m),
            "head_input_dim": self.head_input_dim,
            "head_hidden_dims": list(self.head_hidden_dims),
            "residual_bounds": self.residual_bounds.to_dict(),
            "prior_policy": _prior_policy_to_dict(self.prior_policy),
            "runtime": _runtime_to_dict(self.runtime),
            "seed": self.seed,
            "checkpoint_root": self.checkpoint_root.as_posix(),
        }

    @property
    def config_hash(self) -> str:
        """Return the stable resolved-config SHA-256 identity."""

        encoded = json.dumps(
            _canonical(self.to_dict()), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ReferenceConfig:
        """Load a resolved config from YAML/JSON-compatible values."""

        if not isinstance(value, Mapping):
            raise ReferenceConfigError("A04 reference configuration must be a mapping")
        raw = dict(value)
        config_fields = {
            "schema_version",
            "model_id",
            "stage",
            "quality_first",
            "latency_optimization",
            "data_role",
            "split_identity",
            "active_dataset_ids",
            "excluded_dataset_names",
            "teacher_identity",
            "target_schema_hash",
            "alignment_identity",
            "calibration_identity",
            "renderer_identity",
            "supervision_views",
            "assessment_views",
            "context_scale_m",
            "head_input_dim",
            "head_hidden_dims",
            "residual_bounds",
            "prior_policy",
            "runtime",
            "seed",
            "checkpoint_root",
        }
        unexpected = set(raw) - config_fields
        if unexpected:
            raise ReferenceConfigError(
                "A04 configuration contains unsupported field(s): " + ", ".join(sorted(unexpected))
            )
        bounds = ResidualBounds.from_dict(raw.get("residual_bounds", ResidualBounds().to_dict()))
        prior_policy = _prior_policy_from_dict(
            raw.get("prior_policy", _prior_policy_to_dict(GeometryPriorPolicy()))
        )
        runtime = _runtime_from_dict(
            raw.get("runtime", _runtime_to_dict(DeploymentRuntimeProfile()))
        )
        checkpoint_root = Path(str(raw.get("checkpoint_root", "artifacts/a04_reference")))
        return cls(
            schema_version=_integer(raw.get("schema_version", 1), "schema_version"),
            model_id=_string(raw.get("model_id", "a04_teacher_backed_reference_v1"), "model_id"),
            stage=_string(raw.get("stage", "reference"), "stage"),
            quality_first=bool(raw.get("quality_first", True)),
            latency_optimization=bool(raw.get("latency_optimization", False)),
            data_role=_string(raw.get("data_role", "teacher_fit"), "data_role"),
            split_identity=_string(
                raw.get("split_identity", "a01-scared-dataset1-7-v1"), "split_identity"
            ),
            active_dataset_ids=tuple(
                _string(item, "active_dataset_ids[]")
                for item in _sequence(
                    raw.get("active_dataset_ids", A01_DATASET_IDS), "active_dataset_ids"
                )
            ),
            excluded_dataset_names=tuple(
                _string(item, "excluded_dataset_names[]")
                for item in _sequence(
                    raw.get("excluded_dataset_names", ("SCARED-C",)), "excluded_dataset_names"
                )
            ),
            teacher_identity=_mapping(
                raw.get("teacher_identity", {"status": "injected_at_runtime"}), "teacher_identity"
            ),
            target_schema_hash=_string(
                raw.get("target_schema_hash", DEFAULT_TARGET_SCHEMA.schema_hash),
                "target_schema_hash",
            ),
            alignment_identity=_string(
                raw.get("alignment_identity", "a01-aligned-target-v1"), "alignment_identity"
            ),
            calibration_identity=_string(
                raw.get("calibration_identity", "explicit-calibration-at-forward"),
                "calibration_identity",
            ),
            renderer_identity=_string(
                raw.get("renderer_identity", "injected-field-renderer"), "renderer_identity"
            ),
            supervision_views=tuple(
                _string(item, "supervision_views[]")
                for item in _sequence(raw.get("supervision_views", ("left",)), "supervision_views")
            ),
            assessment_views=tuple(
                _string(item, "assessment_views[]")
                for item in _sequence(
                    raw.get("assessment_views", ("left", "right")), "assessment_views"
                )
            ),
            context_scale_m=_number(raw.get("context_scale_m", 1.0), "context_scale_m"),
            head_input_dim=_integer(raw.get("head_input_dim", 13), "head_input_dim"),
            head_hidden_dims=tuple(
                _integer(item, "head_hidden_dims[]")
                for item in _sequence(raw.get("head_hidden_dims", (128, 128)), "head_hidden_dims")
            ),
            residual_bounds=bounds,
            prior_policy=prior_policy,
            runtime=runtime,
            seed=_integer(raw.get("seed", 0), "seed"),
            checkpoint_root=checkpoint_root,
        )


def load_reference_config(path: Path) -> ReferenceConfig:
    """Load and validate one A04 YAML config without constructing a model."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ReferenceConfigError(
            f"unable to read A04 reference config {path}: {error}"
        ) from error
    if not isinstance(raw, Mapping):
        raise ReferenceConfigError("A04 reference config must contain a mapping")
    return ReferenceConfig.from_dict(raw)


def write_reference_config(path: Path, config: ReferenceConfig) -> None:
    """Write one resolved config and refuse accidental replacement."""

    if path.exists():
        raise ReferenceConfigError(f"refusing to overwrite existing reference config: {path}")
    write_yaml_atomic(path, config.to_dict())


A04ReferenceConfig = ReferenceConfig


__all__ = [
    "A04ReferenceConfig",
    "ReferenceConfig",
    "ReferenceConfigError",
    "load_reference_config",
    "write_reference_config",
]
