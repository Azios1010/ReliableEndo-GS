"""Bounded optimizer and artifact helpers for the learned uncertainty head.

This module deliberately operates on caller-supplied tensors.  It does not
load datasets, initialize devices, run a Phase-I loop, or choose experiment
splits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.optim import Optimizer

from reliable_endo_gs.uncertainty.head import LaplaceUncertaintyHead
from reliable_endo_gs.uncertainty.laplace import laplace_nll
from reliable_endo_gs.uncertainty.providers import UncertaintyProviderProvenance

CHECKPOINT_SCHEMA = "uncertainty_checkpoint.v1"


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    """Finite diagnostics from one bounded optimization step."""

    loss: float
    gradient_norm: float
    parameter_update_norm: float
    valid_count: int


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Portable checkpoint envelope with explicit provenance identities."""

    provider_id: str
    architecture_id: str
    config_identity: str
    checkpoint_identity: str
    schema_version: str
    calibration_id: str
    feature_schema: str | None = None
    checkpoint_schema: str = CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.checkpoint_schema != CHECKPOINT_SCHEMA:
            raise ValueError(f"unsupported checkpoint schema {self.checkpoint_schema!r}")
        values = (
            self.provider_id,
            self.architecture_id,
            self.config_identity,
            self.checkpoint_identity,
            self.schema_version,
            self.calibration_id,
        )
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("checkpoint metadata identities must be non-empty strings")
        if self.feature_schema is not None and not self.feature_schema:
            raise ValueError("feature_schema must be non-empty when supplied")

    @classmethod
    def from_provenance(cls, provenance: UncertaintyProviderProvenance) -> CheckpointMetadata:
        """Build checkpoint metadata without fabricating a hash."""

        return cls(**asdict(provenance))

    def as_provenance(self) -> UncertaintyProviderProvenance:
        """Recover provider provenance from a validated checkpoint envelope."""

        return UncertaintyProviderProvenance(
            provider_id=self.provider_id,
            architecture_id=self.architecture_id,
            config_identity=self.config_identity,
            checkpoint_identity=self.checkpoint_identity,
            schema_version=self.schema_version,
            calibration_id=self.calibration_id,
            feature_schema=self.feature_schema,
        )


def _parameter_norm(parameters: list[torch.Tensor]) -> float:
    """Return an L2 norm on detached parameters for finite diagnostics."""

    if not parameters:
        raise ValueError("uncertainty head must expose at least one parameter")
    squared = torch.stack([parameter.detach().float().pow(2).sum() for parameter in parameters])
    return float(squared.sum().sqrt().item())


def train_uncertainty_step(
    head: LaplaceUncertaintyHead,
    optimizer: Optimizer,
    features: torch.Tensor,
    predicted_disparity: torch.Tensor,
    target_disparity: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_grad_norm: float | None = None,
) -> TrainingStepResult:
    """Run exactly one optimizer step using the canonical Laplace NLL.

    Tensor shapes are ``[B, C, H, W]`` for ``features`` and ``[B, 1, H, W]``
    for disparities/masks.  Dataset loading, batching, and split policy remain
    outside this helper.
    """

    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be strictly positive when supplied")
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer.zero_grad(set_to_none=True)
    head.train()
    output = head(features)
    loss = laplace_nll(predicted_disparity, target_disparity, output.scale_b, valid_mask)
    if not bool(torch.isfinite(loss)):
        raise ValueError("uncertainty training loss is non-finite")
    loss.backward()
    gradients = [parameter.grad for parameter in parameters]
    if any(gradient is None for gradient in gradients):
        raise ValueError("all trainable uncertainty-head parameters must receive gradients")
    finite_gradients = [gradient for gradient in gradients if gradient is not None]
    if not all(bool(torch.isfinite(gradient).all()) for gradient in finite_gradients):
        raise ValueError("uncertainty-head gradients are non-finite")
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    gradient_norm = _parameter_norm(finite_gradients)
    optimizer.step()
    update_squared = torch.stack(
        [
            (parameter.detach() - initial).float().pow(2).sum()
            for parameter, initial in zip(parameters, before, strict=True)
        ]
    )
    update_norm = float(update_squared.sum().sqrt().item())
    if not torch.isfinite(torch.tensor((gradient_norm, update_norm))).all():
        raise ValueError("uncertainty-head optimization diagnostics are non-finite")
    return TrainingStepResult(
        loss=float(loss.detach().item()),
        gradient_norm=gradient_norm,
        parameter_update_norm=update_norm,
        valid_count=int(valid_mask.sum().item()),
    )


def evaluate_uncertainty_head(head: LaplaceUncertaintyHead, features: torch.Tensor) -> torch.Tensor:
    """Evaluate raw ``sigma_d`` deterministically without mutating train mode."""

    was_training = head.training
    head.eval()
    with torch.no_grad():
        output = cast(torch.Tensor, head(features).raw_sigma_d)
    head.train(was_training)
    return output


def save_uncertainty_checkpoint(
    path: Path,
    head: LaplaceUncertaintyHead,
    provenance: UncertaintyProviderProvenance,
    *,
    optimizer: Optimizer | None = None,
    step: int = 0,
) -> None:
    """Persist a head and explicit provenance in a versioned envelope."""

    if step < 0:
        raise ValueError("step must be non-negative")
    metadata = CheckpointMetadata.from_provenance(provenance)
    payload: dict[str, Any] = {
        "metadata": asdict(metadata),
        "model_state_dict": head.state_dict(),
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_uncertainty_checkpoint(
    path: Path,
    head: LaplaceUncertaintyHead,
    *,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> UncertaintyProviderProvenance:
    """Load a checkpoint strictly and return its auditable provider identity."""

    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("uncertainty checkpoint must contain a mapping")
    raw_metadata = payload.get("metadata")
    if not isinstance(raw_metadata, dict):
        raise ValueError("uncertainty checkpoint metadata is missing")
    try:
        metadata = CheckpointMetadata(**raw_metadata)
    except (TypeError, ValueError) as error:
        raise ValueError("uncertainty checkpoint metadata is incompatible") from error
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("uncertainty checkpoint model_state_dict is missing")
    head.load_state_dict(state_dict, strict=True)
    if optimizer is not None:
        optimizer_state = payload.get("optimizer_state_dict")
        if not isinstance(optimizer_state, dict):
            raise ValueError("optimizer was supplied but checkpoint has no optimizer state")
        optimizer.load_state_dict(optimizer_state)
    return metadata.as_provenance()
