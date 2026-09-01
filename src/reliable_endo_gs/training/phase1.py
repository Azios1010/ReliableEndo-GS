"""Bounded, dependency-injected Phase-I cross-view orchestration.

This module coordinates the finalized Plan 04--07 contracts.  It does not
load data, select a scientific provider, call the native renderer, or run an
epoch loop.  The local reference path is deliberately marked development-only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

import torch
from torch import nn
from torch.optim import Optimizer

from reliable_endo_gs.contracts import GaussianField, StereoBatch, StereoPrediction
from reliable_endo_gs.geometry import (
    GeometryConvention,
    center_covariance_from_calibrated_sigma,
    focal_length_x_from_intrinsics,
    pixel_rays,
)
from reliable_endo_gs.probabilistic_gs import (
    RepresentationVariant,
    build_probabilistic_representation,
)
from reliable_endo_gs.rendering.cross_view import (
    CameraRequest,
    CameraView,
    CrossViewLossResult,
    CrossViewMaskResult,
    ReferenceRenderer,
    RobustLossConfig,
    aggregate_view_losses,
    build_cross_view_mask,
    cross_view_loss,
    make_camera_request,
)
from reliable_endo_gs.rendering.protocol import CovarianceRequest, RendererRequest
from reliable_endo_gs.uncertainty.records import CalibratedDisparitySigma


class BaselinePredictor(Protocol):
    """Injection boundary for an already-audited baseline adapter."""

    def predict(self, batch: StereoBatch) -> BaselineOutput:
        """Return baseline stereo evidence and Gaussian parameters."""


class SigmaProvider(Protocol):
    """Provider-neutral Plan 04/05 calibrated uncertainty boundary."""

    def predict(self, prediction: StereoPrediction) -> CalibratedDisparitySigma:
        """Return deployable disparity standard deviation only."""


class Renderer(Protocol):
    """Renderer injection boundary for production or local reference backends."""

    def render(self, request: RendererRequest, camera_request: CameraRequest) -> Any:
        """Render one explicit Gaussian request from one explicit camera."""


@dataclass(frozen=True, slots=True)
class BaselineOutput:
    """Output of a baseline adapter at the Phase-I boundary."""

    stereo: StereoPrediction
    gaussians: GaussianField

    def __post_init__(self) -> None:
        if not isinstance(self.stereo, StereoPrediction):
            raise TypeError("baseline stereo output must be a StereoPrediction")
        if not isinstance(self.gaussians, GaussianField):
            raise TypeError("baseline Gaussian output must be a GaussianField")
        if self.gaussians.batch_size != self.stereo.batch_size:
            raise ValueError("baseline stereo and Gaussian batch dimensions must match")


@dataclass(frozen=True, slots=True)
class Phase1Config:
    """Explicit local Phase-I numerical and loss settings."""

    baseline_m: float
    representation_variant: RepresentationVariant | str = (
        RepresentationVariant.STEREO_COVARIANCE_CORRECTED
    )
    min_abs_disparity_px: float = 1.0e-3
    covariance_epsilon_m2: float = 0.0
    max_covariance_eigenvalue_m2: float | None = None
    rank_tolerance_m2: float = 1.0e-12
    seed: int = 0
    loss: RobustLossConfig = field(default_factory=RobustLossConfig)
    development_renderer_backend: str = "development_reference_renderer.v1"
    config_identity: str = "phase1_development_config"

    def __post_init__(self) -> None:
        if self.baseline_m <= 0 or not _finite_number(self.baseline_m):
            raise ValueError("baseline_m must be finite and strictly positive")
        object.__setattr__(
            self, "representation_variant", RepresentationVariant(self.representation_variant)
        )
        for name, value in (
            ("min_abs_disparity_px", self.min_abs_disparity_px),
            ("covariance_epsilon_m2", self.covariance_epsilon_m2),
            ("rank_tolerance_m2", self.rank_tolerance_m2),
        ):
            if value < 0 or not _finite_number(value):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_covariance_eigenvalue_m2 is not None and (
            self.max_covariance_eigenvalue_m2 < self.covariance_epsilon_m2
            or not _finite_number(self.max_covariance_eigenvalue_m2)
        ):
            raise ValueError("max_covariance_eigenvalue_m2 must be >= covariance_epsilon_m2")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.config_identity or not self.development_renderer_backend:
            raise ValueError("config and renderer identities must be non-empty")


PhaseIConfig = Phase1Config


@dataclass(frozen=True, slots=True)
class PhaseIForwardResult:
    """All named outputs of one injected Phase-I forward pass."""

    baseline: BaselineOutput
    sigma_d: CalibratedDisparitySigma
    sigma_geo: torch.Tensor
    geometry_valid: torch.Tensor
    representation: Any
    camera_left: CameraRequest
    camera_right: CameraRequest
    render_left: Any
    render_right: Any
    mask_left: CrossViewMaskResult
    mask_right: CrossViewMaskResult
    losses: CrossViewLossResult
    provenance: Mapping[str, object]

    @property
    def loss_dict(self) -> dict[str, torch.Tensor | None]:
        """Return named left/right/total losses for structured logging."""

        return self.losses.as_dict()


@dataclass(frozen=True, slots=True)
class Phase1LogRecord:
    """Structured development-only training diagnostics."""

    step: int
    total_loss: float | None
    component_losses: Mapping[str, float | None]
    valid_counts: Mapping[str, int]
    provider_identity: str
    representation_variant: str
    scientific_status: str = "development_only"


@dataclass(frozen=True, slots=True)
class Phase1StepResult:
    """Result of exactly one synthetic/local optimizer step."""

    step: int
    loss: float
    component_losses: Mapping[str, float | None]
    valid_counts: Mapping[str, int]
    gradient_norm: float
    parameter_update_norm: float
    log_record: Phase1LogRecord


CHECKPOINT_SCHEMA = "phase1_checkpoint.v1"


@dataclass(frozen=True, slots=True)
class Phase1CheckpointMetadata:
    """Portable checkpoint identity without dataset or machine paths."""

    step: int
    config_identity: str
    provider_identity: str
    geometry_schema: str
    representation_schema: str
    renderer_backend: str
    seed: int
    phase: str = "phase1"
    scientific_status: str = "development_only"
    schema_version: str = CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.step < 0 or self.seed < 0:
            raise ValueError("checkpoint step and seed must be non-negative")
        for name, value in (
            ("config_identity", self.config_identity),
            ("provider_identity", self.provider_identity),
            ("geometry_schema", self.geometry_schema),
            ("representation_schema", self.representation_schema),
            ("renderer_backend", self.renderer_backend),
            ("phase", self.phase),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.scientific_status != "development_only":
            raise ValueError("local Phase-I checkpoints must be development_only")
        if self.schema_version != CHECKPOINT_SCHEMA:
            raise ValueError(f"unsupported checkpoint schema {self.schema_version!r}")


class PhaseIForward:
    """Compose baseline, uncertainty, Plan 06, Plan 07, and renderer owners."""

    def __init__(
        self,
        baseline: BaselinePredictor | Callable[[StereoBatch], BaselineOutput],
        uncertainty_provider: SigmaProvider,
        renderer: Renderer | Callable[[RendererRequest, CameraRequest], Any],
        *,
        config: Phase1Config,
        convention: GeometryConvention | None = None,
    ) -> None:
        self.baseline = baseline
        self.uncertainty_provider = uncertainty_provider
        self.renderer = renderer
        self.config = config
        self.convention = convention or GeometryConvention()
        # Backend selection is dependency-injected.  A custom test double is
        # allowed, but its identity is recorded as the injected backend rather
        # than being mistaken for native production rendering.

    def train(self, mode: bool = True) -> PhaseIForward:
        """Propagate train/eval mode to injected modules when supported."""

        for component in (self.baseline, self.uncertainty_provider, self.renderer):
            _set_component_mode(component, mode)
        return self

    def eval(self) -> PhaseIForward:
        return self.train(False)

    def forward(self, batch: StereoBatch) -> PhaseIForwardResult:
        """Run one local forward pass from caller-supplied tensors only."""

        if not isinstance(batch, StereoBatch):
            raise TypeError("batch must be a StereoBatch")
        baseline = _call_baseline(self.baseline, batch)
        prediction = baseline.stereo
        sigma = self.uncertainty_provider.predict(prediction)
        if not isinstance(sigma, CalibratedDisparitySigma):
            raise TypeError("uncertainty provider must return CalibratedDisparitySigma")
        if baseline.gaussians.gaussian_count != batch.left.shape[2] * batch.left.shape[3]:
            raise ValueError("Phase-I geometry mapping requires one Gaussian per image pixel")
        variant = RepresentationVariant(self.config.representation_variant)

        height, width = batch.spatial_shape
        rays = pixel_rays(
            batch.left_camera.intrinsics,
            height=height,
            width=width,
            convention=self.convention,
        )
        focal_x = focal_length_x_from_intrinsics(batch.left_camera.intrinsics)
        baseline_tensor = torch.full(
            (batch.batch_size,),
            self.config.baseline_m,
            dtype=batch.left.dtype,
            device=batch.device,
        )
        center = center_covariance_from_calibrated_sigma(
            prediction.disparity,
            cast(Any, sigma),
            focal_x,
            baseline_tensor,
            rays.rays,
            prediction.valid_mask,
            min_abs_disparity=self.config.min_abs_disparity_px,
            epsilon=self.config.covariance_epsilon_m2,
            max_eigenvalue=self.config.max_covariance_eigenvalue_m2,
            rank_tolerance=self.config.rank_tolerance_m2,
            convention=self.convention,
        )
        geometry_valid = center.valid_mask
        center_flat = center.cov_center.reshape(batch.batch_size, height * width, 3, 3)
        geometry_flat = geometry_valid.reshape(batch.batch_size, height * width)
        gaussian_valid = baseline.gaussians.valid_mask & geometry_flat

        representation = build_probabilistic_representation(
            baseline.gaussians.means3d,
            baseline.gaussians.rotations,
            baseline.gaussians.scales,
            baseline.gaussians.opacities,
            None if variant is RepresentationVariant.BASELINE else center_flat,
            gaussian_valid,
            frame=self.convention.frame,
            length_unit=self.convention.length_unit,
            variant=variant,
            colors=baseline.gaussians.colors,
        )
        source = _covariance_source(variant)
        renderer_request = RendererRequest.from_representation(
            representation, covariance_source=source
        )
        camera_left = make_camera_request(
            batch.left_camera, view=CameraView.LEFT, image_size=batch.spatial_shape
        )
        camera_right = make_camera_request(
            batch.right_camera, view=CameraView.RIGHT, image_size=batch.spatial_shape
        )
        render_left = _call_renderer(self.renderer, renderer_request, camera_left)
        render_right = _call_renderer(self.renderer, renderer_request, camera_right)
        _validate_render_output("left", render_left, batch)
        _validate_render_output("right", render_right, batch)
        evidence = _batch_mask_evidence(batch)
        left_evidence = dict(evidence)
        right_evidence = dict(evidence)
        left_evidence["geometry_valid"] = geometry_valid
        right_evidence["geometry_valid"] = geometry_valid
        left_evidence["renderer_visibility"] = _visibility_mask(render_left, geometry_valid)
        right_evidence["renderer_visibility"] = _visibility_mask(render_right, geometry_valid)
        mask_left = build_cross_view_mask(geometry_valid, left_evidence)
        mask_right = build_cross_view_mask(geometry_valid, right_evidence)
        left_loss = cross_view_loss(
            render_left, batch.left, mask_left.combined_mask, self.config.loss
        )
        right_loss = cross_view_loss(
            render_right, batch.right, mask_right.combined_mask, self.config.loss
        )
        losses = aggregate_view_losses(
            left_loss,
            right_loss,
            left_weight=self.config.loss.left_weight,
            right_weight=self.config.loss.right_weight,
        )
        provider_id = str(getattr(sigma, "provider_id", "unknown_provider"))
        provenance: dict[str, object] = {
            "phase": "phase1",
            "scientific_status": "development_only",
            "provider_id": provider_id,
            "calibration_id": sigma.calibration_id,
            "geometry_schema": "geometry_center_covariance.v1",
            "representation_schema": getattr(representation, "SCHEMA_VERSION", "unknown"),
            "representation_variant": variant.value,
            "renderer_backend": getattr(self.renderer, "backend_id", "injected_renderer"),
            "camera_contract": "world_from_camera_camera_to_world",
        }
        return PhaseIForwardResult(
            baseline=baseline,
            sigma_d=sigma,
            sigma_geo=center_flat,
            geometry_valid=geometry_valid,
            representation=representation,
            camera_left=camera_left,
            camera_right=camera_right,
            render_left=render_left,
            render_right=render_right,
            mask_left=mask_left,
            mask_right=mask_right,
            losses=losses,
            provenance=MappingProxyType(provenance),
        )

    __call__ = forward


class Phase1Trainer:
    """One-step optimizer/checkpoint owner for local differentiable tests."""

    def __init__(
        self,
        pipeline: PhaseIForward,
        optimizer: Optimizer,
        *,
        trainable_modules: Mapping[str, nn.Module] | None = None,
        frozen_modules: Mapping[str, nn.Module] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.optimizer = optimizer
        self.trainable_modules = dict(trainable_modules or {})
        self.frozen_modules = dict(frozen_modules or {})
        if any(not name for name in self.trainable_modules | self.frozen_modules):
            raise ValueError("module names must be non-empty")
        if set(self.trainable_modules) & set(self.frozen_modules):
            raise ValueError("a module cannot be both trainable and frozen")
        for module in self.frozen_modules.values():
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.step = 0
        self.records: list[Phase1LogRecord] = []

    def train_step(self, batch: StereoBatch) -> Phase1StepResult:
        """Run exactly one forward/backward/optimizer step."""

        self.pipeline.train(True)
        before_frozen = _snapshot_modules(self.frozen_modules)
        before_trainable = _snapshot_modules(self.trainable_modules)
        self.optimizer.zero_grad(set_to_none=True)
        result = self.pipeline.forward(batch)
        if result.losses.total is None:
            raise ValueError("Phase-I training requires at least one valid supervised view")
        loss = result.losses.total
        if not bool(torch.isfinite(loss)):
            raise ValueError("Phase-I loss is non-finite")
        loss.backward()
        trainable_parameters = [
            parameter
            for module in self.trainable_modules.values()
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        gradients = [parameter.grad for parameter in trainable_parameters]
        if trainable_parameters and any(gradient is None for gradient in gradients):
            raise ValueError("all trainable parameters must receive gradients")
        nonnull_gradients = [gradient for gradient in gradients if gradient is not None]
        if nonnull_gradients and not all(
            bool(torch.isfinite(gradient).all()) for gradient in nonnull_gradients
        ):
            raise ValueError("Phase-I gradients are non-finite")
        gradient_norm = _gradient_norm(nonnull_gradients)
        self.optimizer.step()
        _assert_snapshot_unchanged(self.frozen_modules, before_frozen, "frozen module")
        parameter_update_norm = _update_norm(self.trainable_modules, before_trainable)
        if not _finite_number(gradient_norm) or not _finite_number(parameter_update_norm):
            raise ValueError("Phase-I optimization diagnostics are non-finite")
        self.step += 1
        component_losses = {
            "left_photometric": _float_or_none(result.losses.left.loss),
            "right_cross_view": _float_or_none(result.losses.right.loss),
        }
        valid_counts = {
            "left": result.mask_left.valid_count,
            "right": result.mask_right.valid_count,
        }
        provider_id = str(result.provenance["provider_id"])
        record = Phase1LogRecord(
            step=self.step,
            total_loss=float(loss.detach().item()),
            component_losses=component_losses,
            valid_counts=valid_counts,
            provider_identity=provider_id,
            representation_variant=result.representation.variant.value,
        )
        self.records.append(record)
        return Phase1StepResult(
            self.step,
            float(loss.detach().item()),
            component_losses,
            valid_counts,
            gradient_norm,
            parameter_update_norm,
            record,
        )

    def evaluate(self, batch: StereoBatch) -> PhaseIForwardResult:
        """Run a no-grad deterministic evaluation pass and restore modes."""

        states = _capture_component_modes(
            (self.pipeline.baseline, self.pipeline.uncertainty_provider, self.pipeline.renderer)
        )
        self.pipeline.eval()
        try:
            with torch.no_grad():
                return self.pipeline.forward(batch)
        finally:
            _restore_component_modes(states)

    def save_checkpoint(self, path: Path, metadata: Phase1CheckpointMetadata) -> None:
        """Save model/optimizer/step/provenance in a versioned local envelope."""

        if metadata.step != self.step:
            raise ValueError("checkpoint metadata step must match trainer step")
        save_phase1_checkpoint(path, self, metadata)

    def load_checkpoint(self, path: Path) -> Phase1CheckpointMetadata:
        """Restore model/optimizer/step and return validated provenance."""

        return load_phase1_checkpoint(path, self)


def save_phase1_checkpoint(
    path: Path, trainer: Phase1Trainer, metadata: Phase1CheckpointMetadata
) -> None:
    """Persist a development-only Phase-I checkpoint."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if metadata.step != trainer.step:
        raise ValueError("checkpoint metadata step must match trainer step")
    payload: dict[str, object] = {
        "metadata": asdict(metadata),
        "module_state_dicts": {
            name: module.state_dict() for name, module in trainer.trainable_modules.items()
        },
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "step": trainer.step,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_phase1_checkpoint(path: Path, trainer: Phase1Trainer) -> Phase1CheckpointMetadata:
    """Load and strictly validate a local Phase-I checkpoint."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Phase-I checkpoint must contain a mapping")
    raw_metadata = payload.get("metadata")
    if not isinstance(raw_metadata, dict):
        raise ValueError("Phase-I checkpoint metadata is missing")
    try:
        metadata = Phase1CheckpointMetadata(**raw_metadata)
    except (TypeError, ValueError) as error:
        raise ValueError("Phase-I checkpoint metadata is incompatible") from error
    module_states = payload.get("module_state_dicts")
    if not isinstance(module_states, dict):
        raise ValueError("Phase-I checkpoint module_state_dicts is missing")
    expected_names = set(trainer.trainable_modules)
    if set(module_states) != expected_names:
        raise ValueError("checkpoint trainable-module identities do not match current trainer")
    for name, module in trainer.trainable_modules.items():
        state = module_states[name]
        if not isinstance(state, dict):
            raise ValueError(f"checkpoint state for {name!r} is not a mapping")
        module.load_state_dict(state, strict=True)
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict):
        raise ValueError("Phase-I checkpoint optimizer state is missing")
    trainer.optimizer.load_state_dict(optimizer_state)
    raw_step = payload.get("step")
    if not isinstance(raw_step, int) or raw_step != metadata.step or raw_step < 0:
        raise ValueError("Phase-I checkpoint step is invalid")
    trainer.step = raw_step
    return metadata


def _call_baseline(
    baseline: BaselinePredictor | Callable[[StereoBatch], BaselineOutput], batch: StereoBatch
) -> BaselineOutput:
    output = baseline.predict(batch) if hasattr(baseline, "predict") else baseline(batch)
    if not isinstance(output, BaselineOutput):
        raise TypeError("baseline must return BaselineOutput")
    return output


def _call_renderer(
    renderer: Renderer | Callable[[RendererRequest, CameraRequest], Any],
    request: RendererRequest,
    camera: CameraRequest,
) -> Any:
    return (
        renderer.render(request, camera)
        if hasattr(renderer, "render")
        else renderer(request, camera)
    )


def _covariance_source(variant: RepresentationVariant) -> CovarianceRequest:
    if variant is RepresentationVariant.BASELINE:
        return CovarianceRequest.NONE
    return CovarianceRequest.EFFECTIVE


def _batch_mask_evidence(batch: StereoBatch) -> dict[str, torch.Tensor]:
    allowed = {
        "stereo_valid",
        "in_view",
        "left_right_valid",
        "depth_valid",
        "occlusion",
        "specularity",
        "tool",
        "saturation",
    }
    return {name: value for name, value in batch.masks.items() if name in allowed}


def _visibility_mask(render: Any, reference: torch.Tensor) -> torch.Tensor:
    visibility = getattr(render, "visibility", None)
    if visibility is None:
        return torch.ones_like(reference)
    if not isinstance(visibility, torch.Tensor):
        raise TypeError("renderer visibility must be a tensor")
    if visibility.shape == reference.shape:
        return visibility
    if visibility.shape == reference[:, 0].shape:
        return visibility.unsqueeze(1)
    raise ValueError("renderer visibility must have shape [B,1,H,W]")


def _validate_render_output(name: str, render: Any, batch: StereoBatch) -> None:
    image = getattr(render, "image", None)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} renderer output must expose image tensor")
    if image.shape != batch.left.shape:
        raise ValueError(f"{name} renderer image must match stereo image shape")
    if image.dtype != batch.left.dtype or image.device != batch.device:
        raise ValueError(f"{name} renderer image must match stereo image dtype/device")


def _set_component_mode(component: object, mode: bool) -> None:
    if isinstance(component, nn.Module):
        component.train(mode)
    elif hasattr(component, "train") and callable(component.train):
        component.train(mode)
    elif hasattr(component, "eval") and callable(component.eval) and not mode:
        component.eval()
    nested_head = getattr(component, "head", None)
    if nested_head is not None and nested_head is not component:
        _set_component_mode(nested_head, mode)


def _capture_component_modes(components: tuple[object, ...]) -> list[tuple[object, bool]]:
    modes: list[tuple[object, bool]] = []
    seen: set[int] = set()

    def visit(component: object) -> None:
        if id(component) in seen:
            return
        seen.add(id(component))
        training = getattr(component, "training", None)
        if isinstance(training, bool):
            modes.append((component, training))
        nested_head = getattr(component, "head", None)
        if nested_head is not None and nested_head is not component:
            visit(nested_head)

    for component in components:
        visit(component)
    return modes


def _restore_component_modes(states: list[tuple[object, bool]]) -> None:
    for component, mode in states:
        _set_component_mode(component, mode)


def _snapshot_modules(modules: Mapping[str, nn.Module]) -> dict[str, list[torch.Tensor]]:
    return {
        name: [parameter.detach().clone() for parameter in module.parameters()]
        for name, module in modules.items()
    }


def _assert_snapshot_unchanged(
    modules: Mapping[str, nn.Module], before: Mapping[str, list[torch.Tensor]], name: str
) -> None:
    for module_name, module in modules.items():
        for initial, current in zip(before[module_name], module.parameters(), strict=True):
            if not torch.equal(initial, current.detach()):
                raise ValueError(f"{name} {module_name!r} changed during optimizer step")


def _update_norm(
    modules: Mapping[str, nn.Module], before: Mapping[str, list[torch.Tensor]]
) -> float:
    differences = [
        (current.detach() - initial).float().square().sum()
        for module_name, module in modules.items()
        for initial, current in zip(before[module_name], module.parameters(), strict=True)
    ]
    return float(torch.stack(differences).sum().sqrt().item()) if differences else 0.0


def _gradient_norm(gradients: list[torch.Tensor]) -> float:
    if not gradients:
        return 0.0
    return float(
        torch.stack([gradient.detach().float().square().sum() for gradient in gradients])
        .sum()
        .sqrt()
        .item()
    )


def _float_or_none(value: torch.Tensor | None) -> float | None:
    return None if value is None else float(value.detach().item())


def _finite_number(value: float) -> bool:
    return bool(torch.isfinite(torch.tensor(value)))


__all__ = [
    "BaselineOutput",
    "BaselinePredictor",
    "CHECKPOINT_SCHEMA",
    "Phase1CheckpointMetadata",
    "Phase1Config",
    "PhaseIConfig",
    "Phase1LogRecord",
    "Phase1StepResult",
    "Phase1Trainer",
    "PhaseIForward",
    "PhaseIForwardResult",
    "ReferenceRenderer",
    "SigmaProvider",
    "load_phase1_checkpoint",
    "save_phase1_checkpoint",
]
