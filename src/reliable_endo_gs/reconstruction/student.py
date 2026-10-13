"""Trainable rectified stereo student, reusing canonical A03/A04 construction.

The inherited result's ``target`` slot is an internal disparity adapter, NOT
teacher supervision. Its source/identity explicitly say student prediction.
No teacher object, package, weights, or cache is needed for forward/inference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from math import isfinite
from types import MappingProxyType

import torch
from torch import nn
from torch.nn import functional as F

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch
from reliable_endo_gs.reconstruction.config import ReferenceConfig
from reliable_endo_gs.reconstruction.model import (
    ReferenceProvenance,
    ReferenceReconstructionModel,
    ReferenceReconstructionResult,
)
from reliable_endo_gs.teachers import AlignedTeacherTarget, RectifiedAlignment


class StereoStudent(nn.Module):
    """Shared stride-4 features, horizontal correlation, and RGB residual decoder.

    RGB inputs: [B,3,H,W]. Output disparity: [B,1,H,W], input-grid pixels,
    u_left-u_right. Cost candidates are 0..bins-1 at feature-grid resolution.
    Invalid horizontal matches are masked before softmax. Soft expectation
    and bilinear upsampling are numerical approximations, not calibrated risk.
    Positive effective disparity uses softplus after principal-point offset.
    """

    def __init__(self, width: int = 32, bins: int = 32, minimum_effective_px: float = 0.25) -> None:
        super().__init__()
        if type(width) is not int or width < 1 or type(bins) is not int or bins < 1:
            raise ValueError("student width/bins must be positive integers")
        if (
            isinstance(minimum_effective_px, bool)
            or not isfinite(minimum_effective_px)
            or minimum_effective_px <= 0
        ):
            raise ValueError("minimum effective disparity must be finite and positive")
        self.bins = bins
        self.minimum_effective_px = minimum_effective_px
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(7, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def forward(self, batch: DeploymentStereoBatch) -> torch.Tensor:
        """Predict without GT/teacher inputs; discrete support is not differentiated."""
        return self.predict_with_evidence(batch, include_evidence=False)[0]

    def predict_with_evidence(
        self,
        batch: DeploymentStereoBatch,
        *,
        include_evidence: bool = True,
        include_feature_tokens: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return disparity and detached B1HW evidence from one matching pass."""
        if include_feature_tokens and not include_evidence:
            raise ValueError("feature tokens require explicit matching evidence")
        coarse, probabilities, features = self._matching(batch)
        residual = self.decoder(
            torch.cat((batch.left.float(), batch.right.float(), coarse / batch.left.shape[-1]), 1)
        )
        offset = (batch.camera.K_L[:, 0, 2] - batch.camera.K_R[:, 0, 2])[:, None, None, None]
        effective = self.minimum_effective_px + F.softplus(coarse + residual - offset)
        evidence = self._evidence(batch, probabilities, features) if include_evidence else {}
        if include_feature_tokens:
            evidence["student_features"] = features.detach()
        return offset + effective, evidence

    def _matching(
        self, batch: DeploymentStereoBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Canonical matching computation; also serves detached routing evidence."""
        left = self.features(batch.left.float())
        right = self.features(batch.right.float())
        width = left.shape[-1]
        costs = []
        for shift in range(min(self.bins, width)):
            shifted = F.pad(right[..., : width - shift], (shift, 0))
            score = -(left - shifted).square().mean(1)
            valid = torch.arange(width, device=left.device) >= shift
            costs.append(score.masked_fill(~valid, float("-inf")))
        probabilities = torch.stack(costs, 1).softmax(1)
        candidates = torch.arange(len(costs), device=left.device, dtype=left.dtype)
        coarse = (probabilities * candidates[None, :, None, None]).sum(1, keepdim=True) * 4
        coarse = F.interpolate(coarse, batch.left.shape[-2:], mode="bilinear", align_corners=False)
        return coarse, probabilities, left

    @torch.no_grad()
    def matching_evidence(self, batch: DeploymentStereoBatch) -> dict[str, torch.Tensor]:
        """Detached B1HW cost/feature proxies; not calibrated sigma or confidence.

        Standalone diagnostics perform matching; model forward instead shares
        its existing computation through predict_with_evidence.
        """
        _, probabilities, features = self._matching(batch)
        return self._evidence(batch, probabilities, features)

    @torch.no_grad()
    def _evidence(
        self, batch: DeploymentStereoBatch, probabilities: torch.Tensor, features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Summarize existing matching tensors, without retaining their graph."""
        peaks = probabilities.topk(min(2, probabilities.shape[1]), dim=1).values
        margin = peaks[:, :1] - peaks[:, 1:2] if peaks.shape[1] > 1 else torch.zeros_like(peaks)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1, keepdim=True)
        import math

        entropy = entropy / max(1.0, math.log(probabilities.shape[1]))
        energy = features.square().mean(1, keepdim=True).sqrt()
        return {
            name: F.interpolate(
                value, batch.left.shape[-2:], mode="bilinear", align_corners=False
            ).detach()
            for name, value in {
                "cost_margin": margin,
                "cost_entropy": entropy,
                "feature_energy": energy,
            }.items()
        }


@dataclass(frozen=True, slots=True)
class StudentProvenance(ReferenceProvenance):
    """Student record: BOTH views supervised; no held-out-view claim.

    Inherited identity slots are internal adapters. Serialized fields distinguish
    prediction identity from external teacher supervision used only by training.
    """

    schema_version: str = "student_reconstruction_provenance.v1"
    supervision_views: tuple[str, ...] = ("left", "right")

    def __post_init__(self) -> None:
        if self.supervision_views != ("left", "right") or self.assessment_views != (
            "left",
            "right",
        ):
            raise ValueError("student reconstruction declares both supervised views")
        # Reuse identity/data validation. The historical reference's held-out
        # view condition applies only to that mode, not to the student record.
        validation = ReferenceProvenance(
            **{
                item.name: getattr(self, item.name)
                for item in fields(ReferenceProvenance)
                if item.name != "supervision_views"
            },
            supervision_views=("left",),
        )
        for name in ("teacher_identity", "target_identity", "alignment", "calibration"):
            object.__setattr__(self, name, getattr(validation, name))

    def to_dict(self) -> dict[str, object]:
        record = ReferenceProvenance.to_dict(self)
        record["prediction_identity"] = record.pop("teacher_identity")
        record["prediction_adapter"] = record.pop("target_identity")
        record["held_out_supervision_view"] = False
        return record


@dataclass(frozen=True, slots=True)
class StudentReconstructionResult(ReferenceReconstructionResult):
    """Student output: disparity, prior, Gaussian field, two renders, provenance."""

    matching_evidence: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ReferenceReconstructionResult.__post_init__(self)
        if not isinstance(self.provenance_record, StudentProvenance):
            raise TypeError("student result requires StudentProvenance")
        keys = set(self.matching_evidence)
        required = {"cost_margin", "cost_entropy", "feature_energy"}
        if keys and (not required <= keys or keys - required - {"student_features"}):
            raise ValueError("unsupported student matching evidence schema")
        for name, value in self.matching_evidence.items():
            shape = self.target.disparity.shape
            feature_shape_ok = (
                (
                    value.ndim == 4
                    and value.shape[0] == shape[0]
                    and value.shape[1] > 0
                    and value.shape[2:] == tuple((n + 3) // 4 for n in shape[2:])
                )
                if isinstance(value, torch.Tensor)
                else False
            )
            if (
                not isinstance(value, torch.Tensor)
                or (not feature_shape_ok if name == "student_features" else value.shape != shape)
                or value.device != self.target.disparity.device
                or value.requires_grad
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    "matching evidence must be detached finite B1HW on prediction device"
                )
        object.__setattr__(
            self, "matching_evidence", MappingProxyType(dict(self.matching_evidence))
        )


class StudentReconstructionModel(ReferenceReconstructionModel):
    """Independent trainable stereo/head; reference teacher is never registered.

    Geometry, composition, render and output validation stay with their existing
    owners. Overriding target acquisition avoids the frozen reference's detach.
    """

    def __init__(
        self,
        *,
        renderer: object,
        config: ReferenceConfig,
        width: int = 32,
        bins: int = 32,
        minimum_effective_px: float = 0.25,
    ) -> None:
        config = replace(
            config,
            teacher_identity={"candidate_id": "rectified_student.v1"},
            model_id="rectified_student_reconstruction_v1",
        )
        super().__init__(teacher=None, renderer=renderer, config=config)
        self.stereo_student = StereoStudent(width, bins, minimum_effective_px)

    def forward(
        self,
        batch: DeploymentStereoBatch,
        *,
        target: AlignedTeacherTarget | None = None,
        alignment: RectifiedAlignment | None = None,
        calibration_metadata: Mapping[str, object] | None = None,
        sample_ids: Sequence[str] | None = None,
        data_role: str | None = None,
        include_matching_evidence: bool = False,
        include_feature_tokens: bool = False,
    ) -> StudentReconstructionResult:
        """Run canonical reconstruction with explicit student output provenance."""
        if not isinstance(batch, DeploymentStereoBatch):
            raise TypeError("student forward requires DeploymentStereoBatch")
        if target is not None or alignment is not None:
            raise ValueError("student forward does not accept teacher targets/alignment")
        disparity, evidence = self.stereo_student.predict_with_evidence(
            batch,
            include_evidence=include_matching_evidence,
            include_feature_tokens=include_feature_tokens,
        )
        base = self._reconstruct_prediction(
            batch,
            self._prediction_adapter(batch, disparity, calibration_metadata),
            "trainable_student_prediction",
            calibration_metadata=calibration_metadata,
            sample_ids=sample_ids,
            data_role=data_role,
        )
        record = StudentProvenance(
            **{
                item.name: getattr(base.provenance_record, item.name)
                for item in fields(ReferenceProvenance)
                if item.name not in {"schema_version", "supervision_views"}
            }
        )
        values = {item.name: getattr(base, item.name) for item in fields(base)}
        values.update(provenance_record=record, provenance=record.to_dict())
        diagnostics = dict(base.diagnostics)
        diagnostics["student_valid_count"] = diagnostics.pop("teacher_valid_count")
        diagnostics["provenance_hash"] = record.provenance_hash
        values["diagnostics"] = diagnostics
        if include_matching_evidence:
            values["matching_evidence"] = evidence
        return StudentReconstructionResult(**values)

    def _infer_target(
        self,
        batch: DeploymentStereoBatch,
        *,
        target: AlignedTeacherTarget | None,
        alignment: RectifiedAlignment | None,
        calibration_metadata: Mapping[str, object] | None,
    ) -> tuple[AlignedTeacherTarget, str]:
        if target is not None or alignment is not None:
            raise ValueError("student forward does not accept teacher targets/alignment")
        disparity = self.stereo_student(batch)
        return self._prediction_adapter(
            batch, disparity, calibration_metadata
        ), "trainable_student_prediction"

    def _prediction_adapter(
        self,
        batch: DeploymentStereoBatch,
        disparity: torch.Tensor,
        calibration_metadata: Mapping[str, object] | None,
    ) -> AlignedTeacherTarget:
        """Internal prediction wrapper; external teacher supervision is excluded."""
        return AlignedTeacherTarget(
            disparity,
            torch.isfinite(disparity),
            batch.support,
            metadata={
                "alignment_schema": "a00_rectified_input_grid_v1",
                "target_disparity_units": "a00_input_grid_pixels",
                "disparity_sign": "u_left_minus_right",
                "prediction_source": "student_not_teacher_target",
                "calibration": dict(
                    calibration_metadata
                    or {"calibration_identity": self.config.calibration_identity}
                ),
            },
        )
