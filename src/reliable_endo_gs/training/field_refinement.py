"""Full-frame, both-view training of the Gaussian field refinement (student frozen).

Two arms share one objective (the canonical ``reconstruction_loss``: mean of the left and right Charbonnier losses on
the deployment supports, over the WHOLE frame):

* ``refiner``: train a :class:`FieldRefiner` applied to the composed field; the student and its attribute head are
  frozen and run under ``no_grad``.
* ``head``: control, fine-tune the student's own attribute head with the identical objective (zero added latency),
  stereo network frozen. Weights are returned/saved separately and never overwrite a student checkpoint.

The trainer takes an ordered list of sample ids (the caller owns the seeded permutation and the split), a loader,
and records per-step losses; it asserts that every frozen parameter is bitwise unchanged when it finishes.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from reliable_endo_gs.actions.refinement import full_frame_region, refine_gaussian_field
from reliable_endo_gs.actions.repair import Action, RepairState, _base_result
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch
from reliable_endo_gs.evaluation.rendering import psnr, windowed_ssim
from reliable_endo_gs.gaussians.refinement import FieldRefiner
from reliable_endo_gs.reconstruction.model import ReferenceReconstructionModel
from reliable_endo_gs.training.interventions import reconstruction_loss

Arm = Literal["refiner", "head"]


@dataclass(frozen=True, slots=True)
class FieldRefinementTrainConfig:
    steps: int
    arm: Arm = "refiner"
    learning_rate: float = 1.0e-3
    grad_clip: float = 1.0
    log_every: int = 25

    def __post_init__(self) -> None:
        if self.steps < 1 or self.log_every < 1:
            raise ValueError("steps and log_every must be positive")
        if not 0 < self.learning_rate < 1 or self.grad_clip <= 0:
            raise ValueError("learning_rate must lie in (0,1) and grad_clip must be positive")
        if self.arm not in ("refiner", "head"):
            raise ValueError("arm must be 'refiner' or 'head'")


Loaded = tuple[DeploymentStereoBatch, dict[str, Any]]  # (batch on the device, calibration metadata)


def state_digest(module: nn.Module) -> str:
    """SHA-256 over parameter and buffer bytes in name order (used to prove frozen weights did not move)."""
    digest = hashlib.sha256()
    for name, tensor in sorted(
        {**dict(module.named_parameters()), **dict(module.named_buffers())}.items()
    ):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _stop_outcome(state: RepairState) -> Any:
    return _base_result(state, Action.STOP)


def train_field_refinement(
    model: ReferenceReconstructionModel,
    refiner: FieldRefiner | None,
    load: Callable[[str], Loaded],
    sample_ids: Sequence[str],
    config: FieldRefinementTrainConfig,
    *,
    composition_alpha_epsilon: float,
    record: Callable[[dict[str, Any]], None] = lambda row: None,
) -> dict[str, Any]:
    """Run ``config.steps`` optimiser steps over ``sample_ids`` (cycled); return a summary dict."""
    if config.arm == "refiner" and refiner is None:
        raise ValueError("the refiner arm needs a FieldRefiner")
    if config.arm == "head" and refiner is not None:
        raise ValueError("the head arm trains model.head and takes no refiner")
    if not sample_ids:
        raise ValueError("sample_ids is empty")
    model.eval().requires_grad_(False)
    if config.arm == "head":
        trainable = list(model.head.parameters())
        for parameter in trainable:
            parameter.requires_grad_(True)
        model.head.train()
        frozen_before = {"stereo_student": state_digest(model.stereo_student)}
    else:
        assert refiner is not None
        trainable = list(refiner.parameters())
        refiner.train()
        frozen_before = {"model": state_digest(model)}
    optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
    window: list[float] = []
    started = time.perf_counter()
    for step in range(config.steps):
        sample_id = sample_ids[step % len(sample_ids)]
        batch, calibration = load(sample_id)
        if config.arm == "refiner":
            with torch.no_grad():
                base = model(
                    batch,
                    sample_ids=(sample_id,),
                    data_role="final_train",
                    calibration_metadata=calibration,
                )
            state = RepairState(batch, base)
            assert refiner is not None
            outcome = refine_gaussian_field(
                state, refiner, model, composition_alpha_epsilon=composition_alpha_epsilon
            )
            with torch.no_grad():
                reference_loss = float(reconstruction_loss(state, _stop_outcome(state)))
        else:
            base = model(
                batch,
                sample_ids=(sample_id,),
                data_role="final_train",
                calibration_metadata=calibration,
            )
            state = RepairState(batch, base)
            outcome = _stop_outcome(state)
            reference_loss = float("nan")  # the head arm has no frozen reference inside the step
        loss = reconstruction_loss(state, outcome)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step} ({sample_id})")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip))
        optimizer.step()
        window.append(float(loss))
        if (step + 1) % config.log_every == 0 or step == 0:
            record(
                {
                    "step": step + 1,
                    "loss": float(loss),
                    "loss_window_mean": sum(window) / len(window),
                    "loss_stop_reference": reference_loss,
                    "grad_norm": grad_norm,
                    "seconds": time.perf_counter() - started,
                    "sample_id": sample_id,
                }
            )
            window.clear()
        del state, base, outcome, loss
    if config.arm == "head":
        after = {"stereo_student": state_digest(model.stereo_student)}
    else:
        after = {"model": state_digest(model)}
    if after != frozen_before:
        raise RuntimeError("a frozen parameter changed during training")
    return {
        "steps": config.steps,
        "arm": config.arm,
        "seconds": time.perf_counter() - started,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "frozen_digest": frozen_before,
        "frozen_unchanged": True,
    }


def frame_metrics(
    left: torch.Tensor, right: torch.Tensor, batch: DeploymentStereoBatch
) -> dict[str, float | None]:
    """Both-view PSNR / SSIM with the shared metric functions (full frame, no mask)."""
    return {
        "left_psnr": psnr(left, batch.left).value,
        "right_psnr": psnr(right, batch.right).value,
        "left_ssim": windowed_ssim(left, batch.left).value,
        "right_ssim": windowed_ssim(right, batch.right).value,
    }


def column_error_profile(
    image: torch.Tensor, target: torch.Tensor, *, bins: int = 32
) -> list[float]:
    """Mean squared error per vertical band of the image (width must be divisible by ``bins``)."""
    squared = ((image - target) ** 2).mean(1)[0]
    if squared.shape[-1] % bins:
        raise ValueError("image width must be divisible by bins")
    return [float(v) for v in squared.mean(0).reshape(bins, -1).mean(1)]


__all__ = [
    "FieldRefinementTrainConfig",
    "column_error_profile",
    "frame_metrics",
    "full_frame_region",
    "state_digest",
    "train_field_refinement",
]
