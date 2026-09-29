"""Independent A04 teacher-backed Gaussian reconstruction reference.

The forward path is intentionally explicit:

``DeploymentStereoBatch -> A01 target -> A00/A03 GeometryPrior -> bounded
reference head -> GaussianField -> FieldRenderer``.

Teacher and renderer implementations are injected at runtime.  This module
does not import candidate model packages, load checkpoints, or initialize a
device during import.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Protocol, cast

import torch
from torch import nn

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera
from reliable_endo_gs.contracts.gaussians import GaussianField
from reliable_endo_gs.contracts.rendering import RenderOutput
from reliable_endo_gs.contracts.stereo import StereoPrediction
from reliable_endo_gs.gaussians import (
    GeometryPrior,
    compose_gaussian_field,
    context_from_geometry_prior,
)
from reliable_endo_gs.gaussians.residual_head import ReferenceAttributeHead
from reliable_endo_gs.profiling.runtime import RuntimeProfile, profile_callable
from reliable_endo_gs.reconstruction.config import ReferenceConfig
from reliable_endo_gs.rendering.field import FieldRenderer, FieldRenderRequest
from reliable_endo_gs.teachers import (
    A01_DATASET_IDS,
    AlignedTeacherTarget,
    RectifiedAlignment,
    TeacherPrediction,
    align_teacher_prediction,
    coerce_teacher_prediction,
)
from reliable_endo_gs.teachers.roles import DEFAULT_A01_SCARED_ROLES


class ReferenceModelError(ValueError):
    """Raised when the A04 reference boundary receives incompatible inputs."""


class ReferenceTeacher(Protocol):
    """Minimal injected teacher surface used by the A04 forward."""

    def infer(self, left: torch.Tensor, right: torch.Tensor) -> object:
        """Return an A01 ``TeacherPrediction`` or a coercible equivalent."""


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ReferenceModelError(f"provenance value {type(value).__name__} is not JSON-compatible")


def _digest(value: object) -> str:
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity_mapping(identity: object | None) -> dict[str, object]:
    if identity is None:
        return {}
    to_dict = getattr(identity, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(cast(Mapping[str, object], value))
    if isinstance(identity, Mapping):
        return dict(cast(Mapping[str, object], identity))
    if isinstance(identity, str) and identity.strip():
        return {"candidate_id": identity}
    raise ReferenceModelError("teacher identity must be serializable")


def _move_target(target: AlignedTeacherTarget, device: torch.device) -> AlignedTeacherTarget:
    """Move a cached A01 target to the deployment device without changing semantics."""

    return replace(
        target,
        # A01 targets are frozen supervision inputs at the A04 boundary.  The
        # explicit detach also protects callers that construct an analytic
        # target from a tensor carrying an accidental graph.
        disparity=target.disparity.detach().to(device=device),
        validity=target.validity.detach().to(device=device),
        support=target.support.detach().to(device=device),
        predicted_confidence=(
            None
            if target.predicted_confidence is None
            else target.predicted_confidence.detach().to(device=device)
        ),
        calibrated_risk=(
            None
            if target.calibrated_risk is None
            else target.calibrated_risk.detach().to(device=device)
        ),
    )


def _move_prediction(prediction: TeacherPrediction, device: torch.device) -> TeacherPrediction:
    if prediction.disparity.device == device:
        return prediction
    return replace(
        prediction,
        disparity=prediction.disparity.to(device=device),
        validity=None if prediction.validity is None else prediction.validity.to(device=device),
        support=None if prediction.support is None else prediction.support.to(device=device),
        predicted_confidence=(
            None
            if prediction.predicted_confidence is None
            else prediction.predicted_confidence.to(device=device)
        ),
    )


def _freeze_injected_teacher(teacher: object) -> None:
    freeze = getattr(teacher, "freeze", None)
    if callable(freeze):
        freeze()
    eval_method = getattr(teacher, "eval", None)
    if callable(eval_method):
        eval_method()
    parameters = getattr(teacher, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            if isinstance(parameter, torch.Tensor):
                parameter.requires_grad_(False)


def _validate_views(name: str, views: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(view) for view in views)
    if not normalized or any(view not in ("left", "right") for view in normalized):
        raise ReferenceModelError(f"{name} must contain only left/right and cannot be empty")
    return normalized


def _concrete_renderer_identity(renderer: object | None) -> str | None:
    """Return an explicit backend identity when the renderer exposes one."""

    if renderer is None:
        return None
    for attribute in ("identity", "backend_id"):
        value = getattr(renderer, attribute, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


@dataclass(frozen=True, slots=True)
class ReferenceProvenance:
    """Identity metadata carried by every A04 reference output/checkpoint."""

    teacher_identity: Mapping[str, object]
    target_identity: Mapping[str, object]
    alignment: Mapping[str, object]
    calibration: Mapping[str, object]
    data_role: str
    split_identity: str
    target_source: str
    renderer_identity: str
    config_hash: str
    supervision_views: tuple[str, ...] = ("left",)
    assessment_views: tuple[str, ...] = ("left", "right")
    dataset_name: str = "scared"
    active_dataset_ids: tuple[str, ...] = A01_DATASET_IDS
    excluded_dataset_names: tuple[str, ...] = ("SCARED-C",)
    schema_version: str = "a04_reference_provenance.v1"
    geometry_schema: str = "a03_geometry_prior.v1"
    composition_schema: str = "a04_bounded_composition.v1"

    def __post_init__(self) -> None:
        for name in (
            "data_role",
            "split_identity",
            "target_source",
            "renderer_identity",
            "config_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ReferenceModelError(f"provenance {name} must be non-empty")
        try:
            DEFAULT_A01_SCARED_ROLES.datasets_for_role(self.data_role)
        except ValueError as error:
            raise ReferenceModelError(str(error)) from error
        for name, value in (
            ("teacher_identity", self.teacher_identity),
            ("target_identity", self.target_identity),
            ("alignment", self.alignment),
            ("calibration", self.calibration),
        ):
            if not isinstance(value, Mapping):
                raise ReferenceModelError(f"provenance {name} must be a mapping")
            _canonical(value)
        if self.teacher_identity.get("status") == "injected_at_runtime":
            raise ReferenceModelError("provenance must contain the resolved teacher identity")
        if self.dataset_name != "scared":
            raise ReferenceModelError("A04 reference provenance is restricted to SCARED")
        if tuple(self.active_dataset_ids) != A01_DATASET_IDS:
            raise ReferenceModelError("provenance must enumerate active SCARED dataset_1 through dataset_7")
        if "SCARED-C" not in self.excluded_dataset_names:
            raise ReferenceModelError("provenance must record SCARED-C as excluded")
        supervision = _validate_views("supervision_views", self.supervision_views)
        assessment = _validate_views("assessment_views", self.assessment_views)
        if any(view not in assessment for view in supervision):
            raise ReferenceModelError("supervision views must be included in assessment views")
        if "left" not in assessment:
            raise ReferenceModelError("assessment views must include the left reconstruction view")
        if not any(view not in supervision for view in assessment):
            raise ReferenceModelError(
                "assessment views must include a view held out from supervision views"
            )
        object.__setattr__(self, "teacher_identity", MappingProxyType(dict(self.teacher_identity)))
        object.__setattr__(self, "target_identity", MappingProxyType(dict(self.target_identity)))
        object.__setattr__(self, "alignment", MappingProxyType(dict(self.alignment)))
        object.__setattr__(self, "calibration", MappingProxyType(dict(self.calibration)))
        object.__setattr__(self, "supervision_views", supervision)
        object.__setattr__(self, "assessment_views", assessment)
        object.__setattr__(self, "active_dataset_ids", tuple(self.active_dataset_ids))
        object.__setattr__(self, "excluded_dataset_names", tuple(self.excluded_dataset_names))

    def to_dict(self) -> dict[str, object]:
        """Return all identities in a JSON-compatible mapping."""

        return {
            "schema_version": self.schema_version,
            "teacher_identity": _canonical(self.teacher_identity),
            "target_identity": _canonical(self.target_identity),
            "alignment": _canonical(self.alignment),
            "calibration": _canonical(self.calibration),
            "data_role": self.data_role,
            "split_identity": self.split_identity,
            "target_source": self.target_source,
            "renderer_identity": self.renderer_identity,
            "config_hash": self.config_hash,
            "supervision_views": list(self.supervision_views),
            "assessment_views": list(self.assessment_views),
            "dataset_name": self.dataset_name,
            "active_dataset_ids": list(self.active_dataset_ids),
            "excluded_dataset_names": list(self.excluded_dataset_names),
            "geometry_schema": self.geometry_schema,
            "composition_schema": self.composition_schema,
        }

    @property
    def provenance_hash(self) -> str:
        """Return a stable identity for the complete provenance record."""

        return _digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class ReferenceReconstructionResult:
    """Terminal A04 output with first-party tensors and identity metadata."""

    stereo: StereoPrediction
    target: AlignedTeacherTarget
    prior: GeometryPrior
    gaussians: object
    render_left: RenderOutput
    render_right: RenderOutput | None
    provenance: Mapping[str, object]
    provenance_record: ReferenceProvenance
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    metrics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep the result type independent of later repair/state contracts,
        # while still requiring the canonical first-party GaussianField.
        from reliable_endo_gs.contracts.gaussians import GaussianField

        if not isinstance(self.gaussians, GaussianField):
            raise TypeError("gaussians must be a first-party GaussianField")
        if not isinstance(self.render_left, RenderOutput):
            raise TypeError("render_left must be a RenderOutput")
        if self.render_right is not None and not isinstance(self.render_right, RenderOutput):
            raise TypeError("render_right must be a RenderOutput or None")
        if not isinstance(self.provenance_record, ReferenceProvenance):
            raise TypeError("provenance_record must be ReferenceProvenance")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True, slots=True)
class ReferenceRuntimeHooks:
    """Injectable synchronization and memory hooks for measured profiling."""

    device: torch.device | str = "cpu"
    warmup: int = 0
    repetitions: int = 1
    synchronize: Callable[[], None] | None = None
    reset_peak_memory: Callable[[], None] | None = None
    peak_memory: Callable[[], int] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int) or self.warmup < 0:
            raise ValueError("warmup must be a non-negative integer")
        if (
            isinstance(self.repetitions, bool)
            or not isinstance(self.repetitions, int)
            or self.repetitions < 1
        ):
            raise ValueError("repetitions must be a positive integer")


class ReferenceReconstructionModel(nn.Module):
    """A04 quality reference composed from an injected frozen teacher."""

    def __init__(
        self,
        *,
        teacher: ReferenceTeacher | object | None,
        renderer: FieldRenderer | object | None,
        config: ReferenceConfig | None = None,
        head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        resolved_config = config or ReferenceConfig()
        concrete_renderer_identity = _concrete_renderer_identity(renderer)
        if concrete_renderer_identity is not None:
            if resolved_config.renderer_identity == "injected-field-renderer":
                resolved_config = replace(
                    resolved_config, renderer_identity=concrete_renderer_identity
                )
            elif resolved_config.renderer_identity != concrete_renderer_identity:
                raise ReferenceModelError(
                    "configured renderer_identity does not match the injected renderer: "
                    f"{resolved_config.renderer_identity!r} != {concrete_renderer_identity!r}"
                )
        self.config = resolved_config
        resolved_head = head or ReferenceAttributeHead(
            input_dim=self.config.head_input_dim,
            hidden_dims=self.config.head_hidden_dims,
            bounds=self.config.residual_bounds,
        )
        if not isinstance(resolved_head, nn.Module):
            raise TypeError("head must be a torch.nn.Module")
        if getattr(resolved_head, "input_dim", self.config.head_input_dim) != self.config.head_input_dim:
            raise ReferenceModelError("reference head input_dim does not match configuration")
        self.head = resolved_head
        # Teacher/renderer are runtime dependencies, not trainable model
        # components.  Bypass nn.Module registration so checkpoints contain
        # only the independently fitted reference head.
        object.__setattr__(self, "_teacher", teacher)
        object.__setattr__(self, "_renderer", renderer)
        if teacher is not None:
            _freeze_injected_teacher(teacher)

    @property
    def teacher(self) -> ReferenceTeacher | object | None:
        """Return the injected teacher without registering it as a submodule."""

        return cast(ReferenceTeacher | object | None, self.__dict__.get("_teacher"))

    @property
    def renderer(self) -> FieldRenderer | object | None:
        """Return the injected FieldRenderer boundary."""

        return cast(FieldRenderer | object | None, self.__dict__.get("_renderer"))

    def _teacher_identity(self) -> dict[str, object]:
        teacher = self.teacher
        identity = getattr(teacher, "identity", None) if teacher is not None else None
        if identity is not None:
            mapping = _identity_mapping(identity)
            identity_hash = getattr(identity, "identity_hash", None)
            if isinstance(identity_hash, str) and identity_hash:
                mapping.setdefault("identity_hash", identity_hash)
            return mapping
        if self.config.teacher_identity:
            identity = dict(self.config.teacher_identity)
            if identity.get("status") == "injected_at_runtime":
                raise ReferenceModelError(
                    "an explicit teacher identity is required for the A04 reference"
                )
            return identity
        raise ReferenceModelError("teacher identity is required for reference provenance")

    def _renderer_identity(self) -> str:
        """Resolve the concrete renderer identity without registering it."""

        renderer = self.renderer
        identity = getattr(renderer, "identity", None) if renderer is not None else None
        if isinstance(identity, str) and identity.strip():
            return identity
        if isinstance(identity, Mapping):
            encoded = json.dumps(_canonical(identity), sort_keys=True, separators=(",", ":"))
            return f"renderer:{hashlib.sha256(encoded.encode()).hexdigest()}"
        return self.config.renderer_identity

    @staticmethod
    def _validate_target_metadata(
        target: AlignedTeacherTarget,
        *,
        calibration_metadata: Mapping[str, object] | None,
    ) -> None:
        metadata = target.metadata
        if metadata.get("alignment_schema") != "a00_rectified_input_grid_v1":
            raise ReferenceModelError(
                "aligned target must declare the A00 rectified alignment schema"
            )
        if metadata.get("target_disparity_units") != "a00_input_grid_pixels":
            raise ReferenceModelError(
                "A04 accepts only metric-constructor input disparity in a00_input_grid_pixels"
            )
        if metadata.get("disparity_sign") != "u_left_minus_right":
            raise ReferenceModelError(
                "aligned target must declare disparity_sign=u_left_minus_right"
            )
        if calibration_metadata is None:
            calibration = metadata.get("calibration")
            if not isinstance(calibration, Mapping) or not calibration:
                raise ReferenceModelError(
                    "aligned target must carry calibration metadata or receive it explicitly"
                )

    def _identity_alignment(
        self,
        spatial_shape: tuple[int, int],
        *,
        teacher_hw: tuple[int, int] | None = None,
    ) -> RectifiedAlignment:
        return RectifiedAlignment(
            input_hw=spatial_shape,
            teacher_hw=spatial_shape if teacher_hw is None else teacher_hw,
            crop_hw=spatial_shape,
            image_transform_id=self.config.alignment_identity,
            calibration_transform_id=self.config.calibration_identity,
        )

    def _infer_target(
        self,
        batch: DeploymentStereoBatch,
        *,
        target: AlignedTeacherTarget | None,
        alignment: RectifiedAlignment | None,
        calibration_metadata: Mapping[str, object] | None,
    ) -> tuple[AlignedTeacherTarget, str]:
        if target is not None:
            if not isinstance(target, AlignedTeacherTarget):
                raise TypeError("target must be an A01 AlignedTeacherTarget")
            return _move_target(target, batch.left.device), "aligned_target_injected_or_cache"
        teacher = self.teacher
        if teacher is None:
            raise ReferenceModelError("a teacher is required when no aligned target is supplied")
        infer = getattr(teacher, "infer", None)
        if not callable(infer):
            raise TypeError("teacher must expose infer(left, right)")
        _freeze_injected_teacher(teacher)
        with torch.inference_mode():
            raw = infer(batch.left, batch.right)
        prediction = _move_prediction(coerce_teacher_prediction(raw), batch.left.device)
        identity = self._teacher_identity()
        output = identity.get("output")
        metadata = dict(prediction.metadata)
        if isinstance(output, Mapping):
            metadata.setdefault("disparity_units", output.get("disparity_units"))
            metadata.setdefault("disparity_sign", output.get("disparity_sign"))
        if metadata.get("disparity_units") != "teacher_grid_pixels":
            raise ReferenceModelError(
                "teacher output must declare disparity_units=teacher_grid_pixels"
            )
        if metadata.get("disparity_sign") != "u_left_minus_right":
            raise ReferenceModelError(
                "teacher output must declare disparity_sign=u_left_minus_right"
            )
        prediction = replace(prediction, metadata=metadata)
        spatial_shape = (int(batch.left.shape[2]), int(batch.left.shape[3]))
        resolved_alignment = alignment or self._identity_alignment(
            spatial_shape,
            teacher_hw=tuple(int(value) for value in prediction.disparity.shape[-2:]),
        )
        resolved_calibration = calibration_metadata
        if resolved_calibration is None:
            resolved_calibration = {"calibration_identity": self.config.calibration_identity}
        return (
            align_teacher_prediction(
                prediction,
                resolved_alignment,
                input_support=batch.support,
                calibration_metadata=resolved_calibration,
            ),
            "live_teacher_inference",
        )

    def _validate_target_shape(
        self, target: AlignedTeacherTarget, batch: DeploymentStereoBatch
    ) -> None:
        spatial_shape = (int(batch.left.shape[2]), int(batch.left.shape[3]))
        expected = (batch.left.shape[0], 1, *spatial_shape)
        if tuple(target.disparity.shape) != expected:
            raise ReferenceModelError(
                f"aligned target shape must be {expected}; got {tuple(target.disparity.shape)}"
            )
        if target.disparity.device != batch.left.device:
            raise ReferenceModelError("aligned target must be on the deployment device")
        if target.support.shape != expected or target.validity.shape != expected:
            raise ReferenceModelError("aligned target validity/support shape mismatch")

    def _render(
        self,
        field: GaussianField,
        camera: RectifiedStereoCamera,
        image_hw: tuple[int, int],
        view: str,
    ) -> RenderOutput:
        renderer = self.renderer
        if renderer is None:
            raise ReferenceModelError("an injected FieldRenderer is required for reconstruction")
        # A03 retains the canonical support covariance for diagnostics and
        # metrics.  The independent R/S renderer boundary consumes only the
        # equivalent rotation/scale representation, so covariance is omitted
        # from this request rather than silently selecting a second formula.
        render_field = replace(
            field,
            cov_surface=None,
            cov_center=None,
            cov_effective=None,
        )
        request = FieldRenderRequest(
            field=render_field, camera=camera, image_hw=image_hw, view=view
        )
        render_fn = getattr(renderer, "render", None)
        if callable(render_fn):
            output = render_fn(request)
        elif callable(renderer):
            output = renderer(request)
        else:
            raise TypeError("renderer must expose render(request) or be callable")
        if not isinstance(output, RenderOutput):
            raise TypeError("injected renderer must return a RenderOutput")
        expected = (render_field.means3d.shape[0], 3, *image_hw)
        if tuple(output.image.shape) != expected:
            raise ReferenceModelError(
                f"renderer image shape must be {expected}; got {tuple(output.image.shape)}"
            )
        if output.image.device != render_field.means3d.device:
            raise ReferenceModelError("renderer output must remain on the field device")
        if not bool(torch.isfinite(output.image).all()):
            raise ReferenceModelError("renderer output image must be finite")
        return output

    def forward(
        self,
        batch: DeploymentStereoBatch,
        *,
        target: AlignedTeacherTarget | None = None,
        alignment: RectifiedAlignment | None = None,
        calibration_metadata: Mapping[str, object] | None = None,
        sample_ids: Sequence[str] | None = None,
        data_role: str | None = None,
    ) -> ReferenceReconstructionResult:
        """Run one teacher-backed A04 reference reconstruction.

        ``target`` may be a verified A01 cache item.  If absent, the injected
        teacher runs under inference mode and is frozen before inference.
        Sample IDs are optional for deployment, but when supplied they are
        checked against the active A01 SCARED role policy.
        """

        if not isinstance(batch, DeploymentStereoBatch):
            raise TypeError("A04 reference forward requires DeploymentStereoBatch")
        target, target_source = self._infer_target(
            batch,
            target=target,
            alignment=alignment,
            calibration_metadata=calibration_metadata,
        )
        self._validate_target_metadata(target, calibration_metadata=calibration_metadata)
        self._validate_target_shape(target, batch)
        role = self.config.data_role if data_role is None else data_role
        try:
            expected_datasets = DEFAULT_A01_SCARED_ROLES.datasets_for_role(role)
        except ValueError as error:
            raise ReferenceModelError(str(error)) from error
        if sample_ids is not None:
            if len(sample_ids) != batch.left.shape[0]:
                raise ReferenceModelError("sample_ids must match batch size")
            for sample_id in sample_ids:
                try:
                    normalized = DEFAULT_A01_SCARED_ROLES.validate_sample_id(
                        sample_id, role=role, cache=False
                    )
                except ValueError as error:
                    raise ReferenceModelError(str(error)) from error
                dataset_id = normalized.split("/", maxsplit=1)[0]
                if dataset_id not in expected_datasets:
                    raise ReferenceModelError("sample ID dataset does not match data_role")

        support = target.support & target.validity & batch.support
        from reliable_endo_gs.gaussians import build_geometry_prior

        geometry_prior = build_geometry_prior(
            target.disparity,
            batch.camera,
            support,
            rgb=batch.left,
            policy=self.config.prior_policy,
        )
        context = context_from_geometry_prior(
            geometry_prior, context_scale_m=self.config.context_scale_m
        )
        residuals = self.head(context, geometry_prior.valid_mask)
        # The default/reference head exposes the residual semantics directly;
        # custom injected heads must return the same first-party value.
        from reliable_endo_gs.gaussians.residual_head import AttributeResiduals

        if not isinstance(residuals, AttributeResiduals):
            raise TypeError("reference head must return AttributeResiduals")
        field = compose_gaussian_field(
            geometry_prior,
            residuals,
            bounds=self.config.residual_bounds,
        )
        image_hw = (int(batch.left.shape[2]), int(batch.left.shape[3]))
        render_left = self._render(field, batch.camera, image_hw, "left")
        render_right = (
            self._render(field, batch.camera, image_hw, "right")
            if "right" in self.config.assessment_views
            else None
        )

        teacher_identity = self._teacher_identity()
        target_metadata = target.cache_metadata()
        target_identity = {
            "target_schema_hash": self.config.target_schema_hash,
            "source": target_source,
            "metadata": target_metadata,
        }
        alignment_metadata = {
            key: value
            for key, value in target.metadata.items()
            if key.endswith("_transform_id")
            or key in {
                "alignment_schema",
                "input_hw",
                "teacher_hw",
                "crop_hw",
                "left_crop_xy",
                "right_crop_xy",
                "scale_xy",
                "disparity_sign",
                "source_disparity_units",
                "target_disparity_units",
            }
        }
        if not alignment_metadata:
            alignment_metadata = {
                "alignment_identity": self.config.alignment_identity,
                "target_disparity_units": "a00_input_grid_pixels",
            }
        calibration = calibration_metadata
        if calibration is None:
            raw_calibration = target.metadata.get("calibration")
            calibration = raw_calibration if isinstance(raw_calibration, Mapping) else {}
        calibration_mapping = dict(calibration)
        calibration_mapping.setdefault("calibration_identity", self.config.calibration_identity)
        provenance_record = ReferenceProvenance(
            teacher_identity=teacher_identity,
            target_identity=target_identity,
            alignment=alignment_metadata,
            calibration=calibration_mapping,
            data_role=role,
            split_identity=self.config.split_identity,
            target_source=target_source,
            renderer_identity=self._renderer_identity(),
            config_hash=self.config.config_hash,
            supervision_views=self.config.supervision_views,
            assessment_views=self.config.assessment_views,
        )
        stereo = StereoPrediction(
            disparity=target.disparity,
            valid_mask=target.validity,
            diagnostics={
                "support_count": int(target.support.sum().item()),
                "supported_valid_count": int(support.sum().item()),
                "target_source": target_source,
            },
        )
        diagnostics: dict[str, object] = {
            "teacher_valid_count": int(target.validity.sum().item()),
            "support_count": int(target.support.sum().item()),
            "supported_valid_count": int(support.sum().item()),
            "geometry_prior": {
                "eligible_count": geometry_prior.diagnostics.eligible_count,
                "selected_count": geometry_prior.diagnostics.selected_count,
                "fallback_count": geometry_prior.diagnostics.fallback_count,
                "dropped_count": geometry_prior.diagnostics.dropped_count,
            },
            "provenance_hash": provenance_record.provenance_hash,
        }
        metrics: dict[str, object] = {
            "valid_gaussian_count": int(field.valid_mask.sum().item()),
            "gaussian_count": field.gaussian_count,
            "coverage_fraction": float(
                field.valid_mask.sum().item() / max(field.valid_mask.numel(), 1)
            ),
            "render_views": tuple(self.config.assessment_views),
            "render_left_finite": bool(torch.isfinite(render_left.image).all()),
            "render_right_finite": (
                None if render_right is None else bool(torch.isfinite(render_right.image).all())
            ),
            "quality_target_status": "not_assessed_without_declared_observation_metric",
        }
        return ReferenceReconstructionResult(
            stereo=stereo,
            target=target,
            prior=geometry_prior,
            gaussians=field,
            render_left=render_left,
            render_right=render_right,
            provenance=provenance_record.to_dict(),
            provenance_record=provenance_record,
            diagnostics=diagnostics,
            metrics=metrics,
        )

    def profile(
        self,
        batch: DeploymentStereoBatch,
        *,
        target: AlignedTeacherTarget | None = None,
        alignment: RectifiedAlignment | None = None,
        calibration_metadata: Mapping[str, object] | None = None,
        sample_ids: Sequence[str] | None = None,
        data_role: str | None = None,
        hooks: ReferenceRuntimeHooks | None = None,
    ) -> tuple[RuntimeProfile, ReferenceReconstructionResult]:
        """Measure the injected end-to-end reference path and memory hook."""

        resolved = hooks or ReferenceRuntimeHooks()
        output: ReferenceReconstructionResult | None = None

        def run() -> ReferenceReconstructionResult:
            nonlocal output
            with torch.inference_mode():
                output = self(
                    batch,
                    target=target,
                    alignment=alignment,
                    calibration_metadata=calibration_metadata,
                    sample_ids=sample_ids,
                    data_role=data_role,
                )
            return output

        profile, measured = profile_callable(
            run,
            warmup=resolved.warmup,
            repetitions=resolved.repetitions,
            batch_size=batch.left.shape[0],
            device=resolved.device,
            synchronize=resolved.synchronize,
            reset_peak_memory=resolved.reset_peak_memory,
            peak_memory=resolved.peak_memory,
        )
        if not isinstance(measured, ReferenceReconstructionResult) or output is None:
            raise ReferenceModelError("reference profiling did not return a reconstruction result")
        return profile, measured

    def save_checkpoint(
        self, path: str | object, *, provenance: ReferenceProvenance | None = None
    ) -> None:
        """Persist only the reference head plus explicit config/provenance."""

        from reliable_endo_gs.reconstruction.checkpoint import save_reference_checkpoint

        save_reference_checkpoint(cast(str, path), self, provenance=provenance)

    def load_checkpoint(self, path: str | object, *, strict: bool = True) -> ReferenceProvenance:
        """Load a reference-head checkpoint without loading a teacher."""

        from reliable_endo_gs.reconstruction.checkpoint import load_reference_checkpoint

        return load_reference_checkpoint(cast(str, path), self, strict=strict)


ReferenceModel = ReferenceReconstructionModel
TeacherBackedReference = ReferenceReconstructionModel
ReferenceResult = ReferenceReconstructionResult


__all__ = [
    "ReferenceModel",
    "ReferenceModelError",
    "ReferenceProvenance",
    "ReferenceReconstructionModel",
    "ReferenceReconstructionResult",
    "ReferenceResult",
    "ReferenceRuntimeHooks",
    "ReferenceTeacher",
    "TeacherBackedReference",
]
