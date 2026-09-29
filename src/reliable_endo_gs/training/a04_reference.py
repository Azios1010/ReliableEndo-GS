"""Small executable A04 reference trainer/evaluator.

The trainer owns orchestration only. Geometry, composition, rendering and
teacher target semantics remain in their canonical packages. The optimizer is
constructed from the A04 head exclusively; injected teachers and renderers are
never registered as trainable model components.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera
from reliable_endo_gs.data.scared_a00 import (
    ScaredA00Adapter,
    ScaredA00Dataset,
    ScaredA00Sample,
    collate_scared_a00,
)
from reliable_endo_gs.profiling.runtime import RuntimeProfile
from reliable_endo_gs.reconstruction import (
    ReferenceConfig,
    ReferenceReconstructionModel,
    ReferenceReconstructionResult,
    ReferenceRuntimeHooks,
    load_reference_checkpoint,
    reference_checkpoint_path,
    save_reference_checkpoint,
)
from reliable_endo_gs.rendering.cross_view import RobustLossConfig, cross_view_loss
from reliable_endo_gs.teachers import AlignedTeacherTarget, TargetCache, Teacher
from reliable_endo_gs.teachers.workflow import (
    A01WorkflowError,
    infer_and_align,
    target_cache_key,
)
from reliable_endo_gs.utils.seed import set_seed


class A04TrainingError(RuntimeError):
    """Raised when an A04 command cannot produce a valid result."""


@dataclass(frozen=True, slots=True)
class A04StepResult:
    sample_id: str
    loss: float | None
    valid_pixels: int
    skipped: bool
    target_source: str


@dataclass(frozen=True, slots=True)
class A04RunResult:
    steps: tuple[A04StepResult, ...]
    checkpoint: Path | None = None

    @property
    def completed_steps(self) -> int:
        return sum(not step.skipped for step in self.steps)


def _move_batch(batch: DeploymentStereoBatch, device: torch.device) -> DeploymentStereoBatch:
    camera = batch.camera
    moved_camera = RectifiedStereoCamera(
        camera.K_L.to(device=device),
        camera.K_R.to(device=device),
        camera.T_R_from_L.to(device=device),
        True,
    )
    return DeploymentStereoBatch(
        batch.left.to(device=device),
        batch.right.to(device=device),
        moved_camera,
        batch.support.to(device=device),
    )


def _move_target(target: AlignedTeacherTarget, device: torch.device) -> AlignedTeacherTarget:
    return replace(
        target,
        disparity=target.disparity.detach().to(device=device),
        validity=target.validity.detach().to(device=device),
        support=target.support.detach().to(device=device),
        predicted_confidence=(
            None
            if target.predicted_confidence is None
            else target.predicted_confidence.detach().to(device=device)
        ),
    )


class A04ReferenceTrainer:
    """DataLoader-driven A04 smoke/train/eval/profile orchestration."""

    def __init__(
        self,
        *,
        config: ReferenceConfig,
        adapter: ScaredA00Adapter,
        teacher: Teacher | None,
        renderer: object,
        cache: TargetCache | None = None,
        device: torch.device | str | None = None,
        learning_rate: float = 1.0e-3,
    ) -> None:
        if not isinstance(config, ReferenceConfig):
            raise TypeError("config must be ReferenceConfig")
        if not isinstance(adapter, ScaredA00Adapter):
            raise TypeError("adapter must be ScaredA00Adapter")
        if not callable(getattr(renderer, "render", None)):
            raise A04TrainingError("renderer must expose render(FieldRenderRequest)")
        resolved_device = torch.device(device or config.runtime.device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise A04TrainingError(
                f"A04 requested {resolved_device}, but CUDA is unavailable; use --device cpu for CPU smoke"
            )
        self.config = config
        self.adapter = adapter
        self.teacher = teacher
        self.renderer = renderer
        self.cache = cache
        self.device = resolved_device
        self.model = ReferenceReconstructionModel(
            teacher=teacher,
            renderer=renderer,
            config=config,
        ).to(resolved_device)
        # The model resolves the concrete renderer identity before hashing its
        # config. Keep trainer checkpoint paths and provenance on that same
        # resolved config.
        self.config = self.model.config
        if isinstance(learning_rate, bool) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.optimizer = torch.optim.Adam(self.model.head.parameters(), lr=float(learning_rate))
        scaler_enabled = self.device.type == "cuda" and self.config.runtime.precision == "fp16"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (AttributeError, TypeError):  # pragma: no cover - older supported Torch
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self._optimizer_steps = 0
        self.loader = DataLoader(
            # A04 consumes deployment RGB/calibration and the frozen A01
            # target. GT is kept out of this executable path and is loaded by
            # an explicit assessment protocol when required.
            ScaredA00Dataset(
                adapter,
                include_supervision=False,
                data_role=self.config.data_role,
                output_hw=self.config.runtime.resolution_hw,
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_scared_a00,
        )

    def _target(
        self, sample: ScaredA00Sample, batch: DeploymentStereoBatch
    ) -> tuple[AlignedTeacherTarget, str]:
        if self.cache is not None:
            try:
                cache_identity: Teacher = self.teacher or cast(
                    Teacher, _CacheIdentityTeacher(self.config.teacher_identity)
                )
                key = target_cache_key(
                    sample,
                    cache_identity,
                    split_identity=self.config.split_identity,
                )
                cached = self.cache.get(key)
            except (ValueError, A01WorkflowError) as error:
                raise A04TrainingError(
                    f"invalid A01 cache identity for {sample.sample_id}: {error}"
                ) from error
            if cached is not None:
                return _move_target(cached, self.device), "a01_cache"
        if self.teacher is None:
            raise A04TrainingError(
                f"no cached target for {sample.sample_id} and no injected frozen teacher is available"
            )
        target = infer_and_align(replace(sample, batch=batch), self.teacher)
        return _move_target(target, self.device), "live_teacher_inference"

    def _forward_step(
        self, sample: ScaredA00Sample
    ) -> tuple[ReferenceReconstructionResult, torch.Tensor | None, int, str]:
        batch = _move_batch(sample.batch, self.device)
        target, target_source = self._target(sample, batch)
        with self._autocast_context():
            result = self.model(
                batch,
                target=target,
                sample_ids=(sample.sample_id,),
                data_role=sample.data_role,
                calibration_metadata=sample.calibration_metadata,
            )
        observation = batch.left.to(dtype=result.render_left.image.dtype)
        mask = target.support & target.validity & batch.support
        loss_result = cross_view_loss(
            result.render_left,
            observation,
            mask,
            RobustLossConfig(kind="charbonnier"),
        )
        return result, loss_result.loss, loss_result.valid_count, target_source

    def _autocast_context(self) -> AbstractContextManager[None]:
        """Use AMP only on CUDA, where the declared precision policy applies."""

        if self.device.type != "cuda":
            return nullcontext()
        dtype = torch.float16 if self.config.runtime.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def smoke(self) -> A04RunResult:
        """Run one bounded batch and require a non-empty finite reconstruction."""

        set_seed(self.config.seed)
        self.model.eval()
        try:
            sample = next(iter(self.loader))
        except StopIteration as error:
            raise A04TrainingError("A04 smoke received an empty DataLoader") from error
        with torch.inference_mode():
            result, loss, valid_count, target_source = self._forward_step(sample)
        if result.gaussians.gaussian_count < 1:
            raise A04TrainingError("A04 smoke produced an empty Gaussian field")
        if not bool(torch.isfinite(result.render_left.image).all()):
            raise A04TrainingError("A04 smoke renderer returned non-finite pixels")
        return A04RunResult(
            (
                A04StepResult(
                    sample.sample_id,
                    None if loss is None else float(loss.item()),
                    valid_count,
                    loss is None,
                    target_source,
                ),
            )
        )

    def train(self, *, epochs: int = 1, max_steps: int | None = None) -> A04RunResult:
        """Fit only the independent A04 head for a bounded number of steps."""

        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 1):
            raise ValueError("max_steps must be None or a positive integer")
        set_seed(self.config.seed)
        steps: list[A04StepResult] = []
        self.model.train()
        completed = 0
        for _ in range(epochs):
            for sample in self.loader:
                self.optimizer.zero_grad(set_to_none=True)
                result, loss, valid_count, target_source = self._forward_step(sample)
                if loss is None or valid_count == 0:
                    steps.append(
                        A04StepResult(sample.sample_id, None, valid_count, True, target_source)
                    )
                    continue
                if not bool(torch.isfinite(loss)):
                    raise A04TrainingError(f"non-finite A04 loss for {sample.sample_id}")
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self._optimizer_steps += 1
                steps.append(
                    A04StepResult(
                        sample.sample_id,
                        float(loss.detach().item()),
                        valid_count,
                        False,
                        target_source,
                    )
                )
                completed += 1
                if max_steps is not None and completed >= max_steps:
                    return A04RunResult(tuple(steps))
        return A04RunResult(tuple(steps))

    def evaluate(self) -> A04RunResult:
        """Evaluate without optimizer/state mutation."""

        set_seed(self.config.seed)
        self.model.eval()
        steps: list[A04StepResult] = []
        with torch.inference_mode():
            for sample in self.loader:
                result, loss, valid_count, target_source = self._forward_step(sample)
                steps.append(
                    A04StepResult(
                        sample.sample_id,
                        None if loss is None else float(loss.item()),
                        valid_count,
                        loss is None,
                        target_source,
                    )
                )
        return A04RunResult(tuple(steps))

    def profile(
        self, *, warmup: int = 0, repetitions: int = 1
    ) -> tuple[RuntimeProfile, ReferenceReconstructionResult]:
        """Profile one bounded DataLoader sample using the declared device."""

        self.model.eval()
        try:
            sample = next(iter(self.loader))
        except StopIteration as error:
            raise A04TrainingError("A04 profile received an empty DataLoader") from error
        batch = _move_batch(sample.batch, self.device)
        target, _ = self._target(sample, batch)
        hooks = ReferenceRuntimeHooks(
            device=self.device,
            warmup=warmup,
            repetitions=repetitions,
            reset_peak_memory=(
                torch.cuda.reset_peak_memory_stats if self.device.type == "cuda" else None
            ),
            peak_memory=(torch.cuda.max_memory_allocated if self.device.type == "cuda" else None),
        )
        return self.model.profile(
            batch,
            target=target,
            calibration_metadata=sample.calibration_metadata,
            sample_ids=(sample.sample_id,),
            data_role=sample.data_role,
            hooks=hooks,
        )

    def save(
        self, path: Path | None = None, *, result: ReferenceReconstructionResult | None = None
    ) -> Path:
        """Save a provenance-bound head checkpoint without overwriting."""

        if result is None:
            sample = next(iter(self.loader), None)
            if sample is None:
                raise A04TrainingError("cannot save an A04 checkpoint without a sample")
            self.model.eval()
            with torch.inference_mode():
                result, _, _, _ = self._forward_step(sample)
        destination = Path(path) if path is not None else reference_checkpoint_path(self.config)
        save_reference_checkpoint(
            destination,
            self.model,
            provenance=result.provenance_record,
            optimizer_state=self.optimizer.state_dict(),
            scaler_state=self.scaler.state_dict() if self.scaler.is_enabled() else None,
            training_state={"optimizer_steps": self._optimizer_steps},
        )
        return destination

    def resume(self, path: Path) -> None:
        """Load only the A04 head and validate external teacher identity."""

        training_state: dict[str, object] = {}
        load_reference_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            training_state=training_state,
        )
        raw_steps = training_state.get("optimizer_steps", 0)
        if isinstance(raw_steps, int) and not isinstance(raw_steps, bool) and raw_steps >= 0:
            self._optimizer_steps = raw_steps


@dataclass(frozen=True, slots=True)
class _CacheIdentityTeacher:
    """Identity-only helper used when an A04 cache is validated without a teacher."""

    identity_mapping: Mapping[str, object]

    @property
    def identity(self) -> object:
        import hashlib
        import json
        from types import SimpleNamespace

        encoded = json.dumps(
            dict(self.identity_mapping), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return SimpleNamespace(identity_hash=hashlib.sha256(encoded).hexdigest())


__all__ = [
    "A04RunResult",
    "A04StepResult",
    "A04ReferenceTrainer",
    "A04TrainingError",
]
