"""Small executable A04 reference trainer/evaluator.

The trainer owns orchestration only. Geometry, composition, rendering and
teacher target semantics remain in their canonical packages. The optimizer is
constructed from the A04 head exclusively; injected teachers and renderers are
never registered as trainable model components.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from math import isfinite
from pathlib import Path
from typing import cast

import numpy as np
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
    ReferenceProvenance,
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
from reliable_endo_gs.training.harness import (
    CursorSampler,
    capture_rng,
    checkpoint_rng,
    restore_rng,
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
    skip_reason: str | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class A04RunResult:
    steps: tuple[A04StepResult, ...]
    checkpoint: Path | None = None
    stop_reason: str | None = None

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
        # Initialization must be reproducible too; resume restores RNG after
        # construction. Do not reseed at every train() invocation.
        set_seed(config.seed)
        np.random.seed(config.seed)
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
        self._epoch = 0
        self._next_sample = 0
        self._last_provenance: ReferenceProvenance | None = None
        self._run_metadata: dict[str, object] = {}
        self._skip_counts: dict[str, int] = {}
        self._consecutive_skips = 0
        self._step_skip_reason: str | None = None
        self._step_diagnostics: dict[str, object] = {}
        self._checkpoint_safe = True
        self._boundary_rng = capture_rng(self.device)
        self._dataset = ScaredA00Dataset(
            adapter,
            include_supervision=False,
            data_role=self.config.data_role,
            output_hw=self.config.runtime.resolution_hw,
        )
        self._sampler = CursorSampler(len(self._dataset))
        self.loader = DataLoader(
            # A04 consumes deployment RGB/calibration and the frozen A01
            # target. GT is kept out of this executable path and is loaded by
            # an explicit assessment protocol when required.
            self._dataset,
            batch_size=1,
            sampler=self._sampler,
            num_workers=0,
            collate_fn=collate_scared_a00,
            # DataLoader iterator creation must not consume the model RNG.
            generator=torch.Generator().manual_seed(config.seed),
        )

    def _target(
        self, sample: ScaredA00Sample, batch: DeploymentStereoBatch
    ) -> tuple[AlignedTeacherTarget, str]:
        if self.cache is not None and sample.data_role in self.cache.roles.cache_roles:
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
        # A fresh resume lazily constructs the frozen teacher again. Its
        # initialization/inference must not advance the trainable head's RNG.
        rng = capture_rng(self.device)
        try:
            target = infer_and_align(replace(sample, batch=batch), self.teacher)
        finally:
            restore_rng(rng, self.device)
        return _move_target(target, self.device), "live_teacher_inference"

    def _forward_step(
        self, sample: ScaredA00Sample
    ) -> tuple[ReferenceReconstructionResult, torch.Tensor | None, int, str]:
        self._step_skip_reason = None
        self._step_diagnostics = {}
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
        visibility = result.render_left.visibility
        self._step_diagnostics = self._render_diagnostics(result)
        if visibility is not None and not bool((visibility > 0).any()):
            valid_count = int(result.gaussians.valid_mask.sum().item())
            near_count = self._step_diagnostics["centres_before_near"]
            if valid_count == 0:
                self._step_skip_reason = "empty_geometry_support"
            elif near_count == valid_count:
                self._step_skip_reason = "all_centres_before_near"
            if self.model.training and self._step_skip_reason is not None:
                # The renderer has no supported observation here. Retain the
                # manifest ID, consume it explicitly, and perform no update.
                return result, None, 0, target_source
            raise A04TrainingError(
                f"A04 renderer has no visible primitives for {sample.sample_id}; "
                f"diagnostics={self._step_diagnostics}"
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

    def _render_diagnostics(self, result: ReferenceReconstructionResult) -> dict[str, object]:
        """Summarize left-camera metres and renderer culling without changing geometry."""

        valid = result.gaussians.valid_mask
        depths = result.gaussians.means3d[..., 2][valid]
        finite = depths[torch.isfinite(depths)]
        near = getattr(self.renderer, "metric_near_plane_m", None)
        if near is not None and (
            not isinstance(near, (int, float)) or not isfinite(near) or near <= 0
        ):
            raise A04TrainingError("renderer declares an invalid metric near plane")
        visibility = result.render_left.visibility
        return {
            "supported_valid_pixels": result.diagnostics["supported_valid_count"],
            "valid_gaussians": int(valid.sum().item()),
            "visible_gaussians": None if visibility is None else int((visibility > 0).sum().item()),
            "depth_min_m": None if not finite.numel() else float(finite.min().item()),
            "depth_max_m": None if not finite.numel() else float(finite.max().item()),
            "metric_near_plane_m": near,
            "centres_before_near": None if near is None else int((depths <= near).sum().item()),
        }

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
        self._sampler.start = 0
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

    @property
    def training_progress(self) -> dict[str, object]:
        """Zero-based cursor: completed epochs and next sample within an epoch."""

        return {
            "epoch": self._epoch,
            "next_sample_index": self._next_sample,
            "samples_per_epoch": len(self._dataset),
            "samples_seen": self._epoch * len(self._dataset) + self._next_sample,
            "optimizer_steps": self._optimizer_steps,
            "manifest_hash": self.adapter.manifest.manifest_hash,
            "skip_counts": dict(self._skip_counts),
            "consecutive_skips": self._consecutive_skips,
        }

    def train(
        self,
        *,
        epochs: int = 1,
        max_steps: int | None = None,
        log_every: int = 25,
        checkpoint_every: int = 500,
        checkpoint_dir: Path | None = None,
        progress: Callable[[Mapping[str, object]], None] | None = None,
        run_metadata: Mapping[str, object] | None = None,
        max_consecutive_skips: int = 100,
        stop_requested: Callable[[], str | None] | None = None,
    ) -> A04RunResult:
        """Train to a total epoch budget; max_steps caps updates in this call.

        Data order is the manifest's sequential final_train order, batch size
        one and no workers. Periodic files are immutable and save the cursor
        after the logged update. No skipped samples are replayed on resume.
        """

        if self.config.data_role != "final_train":
            raise A04TrainingError("A04 training is restricted to the final_train role")
        if type(epochs) is not int or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be None or a positive integer")
        for name, interval in (
            ("log_every", log_every),
            ("checkpoint_every", checkpoint_every),
            ("max_consecutive_skips", max_consecutive_skips),
        ):
            if type(interval) is not int or interval < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._check_loader()
        if not len(self._dataset):
            raise A04TrainingError("A04 training received an empty dataset")
        if epochs <= self._epoch:
            raise A04TrainingError("epoch budget already completed; increase --epochs to continue")
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if not checkpoint_dir.is_dir():
                raise A04TrainingError("checkpoint_dir must be a directory")
            for existing in checkpoint_dir.glob("step-*.pt"):
                suffix = existing.stem.removeprefix("step-")
                if suffix.isdigit() and int(suffix) > self._optimizer_steps:
                    raise A04TrainingError(
                        "checkpoint directory contains later steps; resume its latest checkpoint "
                        "or choose a new --checkpoint-dir"
                    )
        if run_metadata is not None:
            self._run_metadata = dict(run_metadata)
        steps: list[A04StepResult] = []
        self.model.train()
        completed = 0
        started = time.monotonic()
        self._checkpoint_safe = True
        self._boundary_rng = capture_rng(self.device)

        def emit(event: str, **values: object) -> None:
            if progress is not None:
                progress({"event": event, **self.training_progress, **values})

        emit("train_start", epochs=epochs, max_steps=max_steps, metadata=self._run_metadata)
        latest_checkpoint: Path | None = None
        while self._epoch < epochs:
            self._sampler.start = self._next_sample
            for sample in self.loader:
                reason = None if stop_requested is None else stop_requested()
                if reason is not None:
                    emit("train_stop", reason=reason, completed_steps=completed)
                    return A04RunResult(tuple(steps), latest_checkpoint, reason)
                sample_epoch, sample_index = self._epoch, self._next_sample
                self.optimizer.zero_grad(set_to_none=True)
                self._step_skip_reason = None
                self._step_diagnostics = {}
                result, loss, valid_count, target_source = self._forward_step(sample)
                skipped = loss is None or valid_count == 0
                skip_reason = (self._step_skip_reason or "empty_supervision") if skipped else None
                if not skipped:
                    assert loss is not None
                    if not bool(torch.isfinite(loss)):
                        raise A04TrainingError(f"non-finite A04 loss for {sample.sample_id}")
                    # Do not serialize an update that was interrupted halfway
                    # through backward/Adam. Periodic checkpoints remain valid.
                    self._checkpoint_safe = False
                    if self.scaler.is_enabled():
                        old_scale = self.scaler.get_scale()
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        # GradScaler reduces its scale precisely when it has
                        # skipped this optimizer update due to overflow.
                        skipped = self.scaler.get_scale() < old_scale
                        skip_reason = "amp_overflow" if skipped else None
                    else:
                        loss.backward()
                        if any(
                            p.grad is not None and not bool(torch.isfinite(p.grad).all())
                            for p in self.model.head.parameters()
                        ):
                            raise A04TrainingError(
                                f"non-finite A04 gradients for {sample.sample_id}"
                            )
                        self.optimizer.step()
                    if not skipped:
                        self._optimizer_steps += 1
                        completed += 1
                if skipped:
                    assert skip_reason is not None
                    self._skip_counts[skip_reason] = self._skip_counts.get(skip_reason, 0) + 1
                    self._consecutive_skips += 1
                else:
                    self._consecutive_skips = 0
                self._last_provenance = result.provenance_record
                self._next_sample += 1
                if self._next_sample == len(self._dataset):
                    self._epoch += 1
                    self._next_sample = 0
                self._checkpoint_safe = True
                self._boundary_rng = capture_rng(self.device)
                steps.append(
                    A04StepResult(
                        sample.sample_id,
                        None if loss is None else float(loss.detach().item()),
                        valid_count,
                        skipped,
                        target_source,
                        skip_reason,
                        dict(self._step_diagnostics),
                    )
                )
                requested_reason = None if stop_requested is None else stop_requested()
                stop = (
                    max_steps is not None and completed >= max_steps
                ) or requested_reason is not None
                seen = self._epoch * len(self._dataset) + self._next_sample
                if (
                    len(steps) == 1
                    or seen % log_every == 0
                    or skipped
                    or stop
                    or self._next_sample == 0
                ):
                    emit(
                        "sample_progress",
                        sample_id=sample.sample_id,
                        sample_epoch=sample_epoch,
                        sample_index=sample_index,
                        loss=steps[-1].loss,
                        valid_pixels=valid_count,
                        skipped=skipped,
                        skip_reason=skip_reason,
                        target_source=target_source,
                        elapsed_seconds=time.monotonic() - started,
                        diagnostics=dict(self._step_diagnostics),
                    )
                if self._consecutive_skips >= max_consecutive_skips:
                    raise A04TrainingError(
                        f"{self._consecutive_skips} consecutive unsupported samples; "
                        "inspect teacher/geometry diagnostics before continuing"
                    )
                if (
                    checkpoint_dir is not None
                    and not skipped
                    and self._optimizer_steps % checkpoint_every == 0
                ):
                    latest_checkpoint = self.save(
                        checkpoint_dir / f"step-{self._optimizer_steps:09d}.pt", result=result
                    )
                    emit("checkpoint_saved", checkpoint=str(latest_checkpoint))
                # Avoid keeping a render/teacher/autograd graph across samples.
                del result, loss
                if stop:
                    reason = requested_reason or "max_steps"
                    emit("train_stop", reason=reason, completed_steps=completed)
                    return A04RunResult(tuple(steps), latest_checkpoint, reason)
        emit("train_stop", reason="epochs_completed", completed_steps=completed)
        return A04RunResult(tuple(steps), latest_checkpoint, "epochs_completed")

    def _check_loader(self) -> None:
        if (
            self.loader.dataset is not self._dataset
            or self.loader.sampler is not self._sampler
            or self.loader.batch_size != 1
            or self.loader.num_workers != 0
        ):
            raise A04TrainingError("exact resume requires the original sequential A04 DataLoader")

    def evaluate(self) -> A04RunResult:
        """Evaluate without optimizer/state mutation."""

        if self.config.data_role != "final_test":
            raise A04TrainingError("A04 final evaluation is restricted to the final_test role")
        set_seed(self.config.seed)
        self.model.eval()
        self._sampler.start = 0
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
        self._sampler.start = 0
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

        self._check_loader()
        if not self._checkpoint_safe:
            raise A04TrainingError("cannot checkpoint a partially completed optimizer update")
        destination = Path(path) if path is not None else reference_checkpoint_path(self.config)
        if destination.exists():
            raise A04TrainingError(f"refusing to overwrite existing checkpoint: {destination}")
        rng = capture_rng(self.device)
        provenance = result.provenance_record if result is not None else self._last_provenance
        if provenance is None:
            if not len(self._dataset):
                raise A04TrainingError("cannot save an A04 checkpoint without a sample")
            was_training = self.model.training
            try:
                self.model.eval()
                with torch.inference_mode():
                    first_result, _, _, _ = self._forward_step(self._dataset[0])
                provenance = first_result.provenance_record
            finally:
                self.model.train(was_training)
                restore_rng(rng, self.device)
        self._last_provenance = provenance
        save_reference_checkpoint(
            destination,
            self.model,
            provenance=provenance,
            optimizer_state=self.optimizer.state_dict(),
            scaler_state=self.scaler.state_dict() if self.scaler.is_enabled() else None,
            training_state={
                "schema": "a04_training_cursor.v1",
                **self.training_progress,
                "data_role": self.config.data_role,
                "rng": rng,
                "run_metadata": self._run_metadata,
            },
        )
        return destination

    def save_recovery(self, directory: Path, error: BaseException) -> Path:
        """Preserve the last committed cursor on a safe Python-level failure."""

        import uuid

        message = str(error).lower()
        if any(
            text in message
            for text in (
                "illegal memory access",
                "device-side assert",
                "unspecified launch failure",
            )
        ):
            raise A04TrainingError("CUDA context is unsafe; use the last periodic checkpoint")
        if not self._checkpoint_safe or self._last_provenance is None:
            raise A04TrainingError("no safe completed update boundary is available for recovery")
        previous_rng = capture_rng(self.device)
        try:
            restore_rng(self._boundary_rng, self.device)
            destination = Path(directory) / (
                f"recovery-step-{self._optimizer_steps:09d}-sample-"
                f"{self._epoch * len(self._dataset) + self._next_sample:09d}-{uuid.uuid4().hex[:8]}.pt"
            )
            return self.save(destination)
        finally:
            restore_rng(previous_rng, self.device)

    def resume(self, path: Path, *, restore_training: bool = True) -> None:
        """Restore an exact training cursor, or load model weights for assessment.

        Legacy optimizer-only checkpoints can be read for assessment. They
        cannot establish a data position and are rejected for training resume.
        """

        self._check_loader()
        state: Mapping[str, object] = {}
        skip_counts: dict[str, int] = {}
        consecutive_skips = 0
        if restore_training:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping):
                raise A04TrainingError("checkpoint must contain a mapping")
            raw_state = payload.get("training_state")
            if (
                not isinstance(raw_state, Mapping)
                or raw_state.get("schema") != "a04_training_cursor.v1"
            ):
                raise A04TrainingError(
                    "checkpoint lacks an exact data cursor; legacy checkpoints cannot resume training"
                )
            state = raw_state
            if (
                state.get("manifest_hash") != self.adapter.manifest.manifest_hash
                or state.get("samples_per_epoch") != len(self._dataset)
                or state.get("data_role") != self.config.data_role
            ):
                raise A04TrainingError(
                    "checkpoint manifest/order/role does not match this DataLoader"
                )
            counts: dict[str, int] = {}
            for name in ("epoch", "next_sample_index", "optimizer_steps", "samples_seen"):
                value = state.get(name)
                if type(value) is not int or value < 0:
                    raise A04TrainingError(f"checkpoint has an invalid {name}")
                counts[name] = value
            epoch = counts["epoch"]
            next_sample = counts["next_sample_index"]
            raw_skips = state.get("skip_counts")
            if raw_skips is None:
                historical_skips = counts["samples_seen"] - counts["optimizer_steps"]
                skip_counts = (
                    {"historical_unclassified": historical_skips} if historical_skips else {}
                )
            elif isinstance(raw_skips, Mapping) and all(
                isinstance(key, str) and type(value) is int and value >= 0
                for key, value in raw_skips.items()
            ):
                skip_counts = dict(raw_skips)
            else:
                raise A04TrainingError("checkpoint has invalid skip counts")
            raw_consecutive = state.get("consecutive_skips", 0)
            if type(raw_consecutive) is not int or raw_consecutive < 0:
                raise A04TrainingError("checkpoint has invalid consecutive skip count")
            consecutive_skips = raw_consecutive
            if (
                not 0 <= next_sample < len(self._dataset)
                or state["samples_seen"] != epoch * len(self._dataset) + next_sample
                or counts["optimizer_steps"] > counts["samples_seen"]
                or not isinstance(payload.get("optimizer_state"), Mapping)
                or (
                    self.scaler.is_enabled()
                    and not isinstance(payload.get("scaler_state"), Mapping)
                )
                or not isinstance(state.get("run_metadata", {}), Mapping)
                or sum(skip_counts.values()) != counts["samples_seen"] - counts["optimizer_steps"]
                or consecutive_skips > sum(skip_counts.values())
            ):
                raise A04TrainingError("checkpoint has inconsistent cursor/optimizer/scaler state")
            try:
                rng = checkpoint_rng(state.get("rng"))
                # Validate restoration before mutating the model/optimizer.
                previous_rng = capture_rng(self.device)
                try:
                    restore_rng(rng, self.device)
                finally:
                    restore_rng(previous_rng, self.device)
            except (TypeError, ValueError, KeyError, RuntimeError) as error:
                raise A04TrainingError(f"invalid checkpoint RNG: {error}") from error
        provenance = load_reference_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer if restore_training else None,
            scaler=self.scaler if restore_training else None,
        )
        self._last_provenance = provenance
        if restore_training:
            self._epoch = cast(int, state["epoch"])
            self._next_sample = cast(int, state["next_sample_index"])
            self._optimizer_steps = cast(int, state["optimizer_steps"])
            self._skip_counts = skip_counts
            self._consecutive_skips = consecutive_skips
            self._run_metadata = dict(cast(Mapping[str, object], state.get("run_metadata", {})))
            restore_rng(checkpoint_rng(state["rng"]), self.device)
            self._boundary_rng = capture_rng(self.device)
            self._checkpoint_safe = True


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
