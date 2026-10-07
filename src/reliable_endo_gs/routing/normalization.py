"""Training-only robust normalization, with fixed checkpoint-sized buffers."""

from __future__ import annotations

import hashlib

import torch
from torch import nn

from reliable_endo_gs.config.intervention import InterventionConfig


class TrainingNormalization(nn.Module):
    """Fit first N training snapshots; freeze statistics for deployment/resume.

    Features [R,F]; local/global utilities [R,3]. Utility scaling has no mean
    subtraction, is common across actions and preserves STOP=0. Median/IQR
    scaling is an optimization transform, not uncertainty calibration.
    """

    center: torch.Tensor
    scale: torch.Tensor
    active: torch.Tensor
    feature_rows: torch.Tensor
    utility_rows: torch.Tensor
    counts: torch.Tensor
    fitted: torch.Tensor
    gain_scales: torch.Tensor
    sample_chain: torch.Tensor

    def __init__(self, features: int, flags: tuple[int, ...], config: InterventionConfig) -> None:
        super().__init__()
        self.config, self.flags = config, flags
        self.register_buffer("center", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))
        self.register_buffer("active", torch.ones(features, dtype=torch.bool))
        self.register_buffer("feature_rows", torch.zeros(config.normalization_rows, features))
        self.register_buffer("utility_rows", torch.zeros(config.normalization_rows * 2, 2))
        # feature rows, utility rows, successful training snapshots
        self.register_buffer("counts", torch.zeros(3, dtype=torch.long))
        self.register_buffer("fitted", torch.tensor(False))
        self.register_buffer("gain_scales", torch.full((2,), config.initial_gain_scale))
        self.register_buffer("sample_chain", torch.zeros(32, dtype=torch.uint8))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if (
            values.ndim != 2
            or values.shape[1] != self.center.numel()
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("normalization requires finite [R,F] features")
        if not bool(self.fitted):
            return values.float()
        transformed = (values.float() - self.center) / self.scale
        transformed = torch.where(self.active, transformed, torch.zeros_like(transformed))
        return transformed.clamp(-self.config.feature_clip, self.config.feature_clip)

    @torch.no_grad()
    def observe(
        self,
        features: torch.Tensor,
        global_gain: torch.Tensor,
        local_gain: torch.Tensor,
        *,
        sample_id: str,
        role: str,
    ) -> None:
        """Commit AFTER a successful joint update; never called by inference."""
        if role != "final_train" or sample_id.split("/")[0] not in {
            f"dataset_{i}" for i in range(1, 8)
        }:
            raise ValueError("normalizer may only observe original SCARED training inputs")
        if bool(self.fitted):
            return
        if global_gain.shape != (features.shape[0], 3) or local_gain.shape != global_gain.shape:
            raise ValueError("normalizer utility labels must be [R,3]")
        if not bool(
            torch.isfinite(features).all()
            & torch.isfinite(global_gain).all()
            & torch.isfinite(local_gain).all()
        ):
            raise ValueError("normalizer observations must be finite")
        feature_count, utility_count = int(self.counts[0]), int(self.counts[1])
        n = min(features.shape[0], self.feature_rows.shape[0] - feature_count)
        self.feature_rows[feature_count : feature_count + n].copy_(features[:n].detach())
        utilities = torch.stack((global_gain[:, 1:], local_gain[:, 1:]), -1).reshape(-1, 2).detach()
        m = min(utilities.shape[0], self.utility_rows.shape[0] - utility_count)
        self.utility_rows[utility_count : utility_count + m].copy_(utilities[:m])
        self.counts.add_(self.counts.new_tensor([n, m, 1]))
        chain = hashlib.sha256(
            bytes(self.sample_chain.cpu().tolist()) + sample_id.encode() + b"\0"
        ).digest()
        self.sample_chain.copy_(
            torch.tensor(list(chain), dtype=torch.uint8, device=self.sample_chain.device)
        )
        # A small smoke grid can expose only one region. Keep collecting until
        # two rows exist rather than fail after an already successful update.
        if int(self.counts[2]) >= self.config.normalization_frames and int(self.counts[0]) >= 2:
            self._fit()

    @torch.no_grad()
    def _fit(self) -> None:
        values = self.feature_rows[: int(self.counts[0])]
        if values.shape[0] < 2:
            raise ValueError("normalization requires at least two observed regions")
        quantiles = values.quantile(values.new_tensor([0.25, 0.5, 0.75]), dim=0)
        spread = (quantiles[2] - quantiles[0]) / 1.349  # normal-distribution IQR conversion
        varying = values.amax(0) - values.amin(0) > self.config.normalization_floor
        self.center.copy_(quantiles[1])
        self.scale.copy_(spread.clamp_min(self.config.normalization_floor))
        self.active.copy_(varying)
        # Availability indicators retain their declared binary semantics.
        for index in self.flags:
            self.center[index] = 0
            self.scale[index] = 1
            self.active[index] = True
        utilities = self.utility_rows[: int(self.counts[1])].abs()
        magnitude = utilities.quantile(0.75, dim=0)
        self.gain_scales.copy_(
            torch.where(
                magnitude > self.config.normalization_floor,
                magnitude.clamp_min(self.config.minimum_gain_scale),
                self.gain_scales,
            )
        )
        self.fitted.fill_(True)

    def validate_state(self) -> None:
        """Reject malformed normalization buffers before checkpoint mutation."""
        if not bool(
            torch.isfinite(self.scale).all() & (self.scale > 0).all() & (self.gain_scales > 0).all()
        ):
            raise ValueError("invalid normalization scales")
        if (
            bool((self.counts < 0).any())
            or int(self.counts[0]) > self.feature_rows.shape[0]
            or int(self.counts[1]) > self.utility_rows.shape[0]
            or int(self.counts[2]) > max(2, self.config.normalization_frames)
        ):
            raise ValueError("invalid normalization observation counts")
        ready = int(self.counts[2]) >= self.config.normalization_frames and int(self.counts[0]) >= 2
        if bool(self.fitted) != ready:
            raise ValueError("inconsistent normalization calibration state")
