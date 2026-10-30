"""Offline metric labels for joint learning; GT never enters deployment state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.routing.tradeoff import (
    COMPONENT_NAMES,
    TradeoffConfig,
    feasible_changes,
    grouped_harm,
    outcome_utility,
)

if TYPE_CHECKING:
    from reliable_endo_gs.data.scared_a00 import ScaredA00Sample


@dataclass(frozen=True, slots=True)
class OfflineMetricTruth:
    """Explicit matching TRAIN identity plus disparity/mask [1,1,H,W]."""

    sample_id: str
    disparity: torch.Tensor
    valid_mask: torch.Tensor
    data_role: str = "final_train"
    resampling_identity: str | None = None

    @classmethod
    def from_a00(cls, sample: ScaredA00Sample) -> OfflineMetricTruth:
        """Bind real offline GT to the adapter's declared output-grid identity.

        Direct construction remains available for synthetic tensors. Real A00
        callers must retain preprocessing provenance instead of accepting a
        matching tensor shape as evidence of pixel correspondence.
        """
        from reliable_endo_gs.data.scared_a00 import A00_SUPERVISION_RESAMPLING

        truth = sample.supervision
        if truth is None:
            raise ValueError("missing separate offline TRAIN metric truth")
        alignment = sample.calibration_metadata.get("runtime_resampling")
        if (
            not isinstance(alignment, dict)
            or alignment.get("supervision") != A00_SUPERVISION_RESAMPLING
            or alignment.get("validity") != "same_nearest_exact_source_index_as_disparity"
        ):
            raise ValueError(
                "offline GT loader must declare the supported runtime resampling identity"
            )
        result = cls(
            sample.sample_id,
            truth.disparity,
            truth.valid_mask,
            sample.data_role,
            A00_SUPERVISION_RESAMPLING,
        )
        result.validate(sample.sample_id, sample.batch.left[:, :1].shape)
        return result

    def validate(self, sample_id: str, shape: torch.Size) -> None:
        if (
            self.sample_id != sample_id
            or self.data_role != "final_train"
            or self.sample_id.split("/")[0] not in {f"dataset_{i}" for i in range(1, 8)}
        ):
            raise ValueError("offline metric truth must match the TRAIN sample identity")
        if self.disparity.shape != shape or self.valid_mask.shape != shape:
            raise ValueError("offline metric truth must match the deployment output grid")
        if self.valid_mask.dtype != torch.bool or bool(
            (self.valid_mask & ~torch.isfinite(self.disparity)).any()
        ):
            raise ValueError("offline metric truth has invalid support")


def metric_action_policy(config: InterventionConfig, action: int) -> TradeoffConfig:
    """One raw-unit guard owner; Gaussian may tighten, never relax, tolerances."""
    if config.metric_quality is None:
        raise ValueError("physical metric policy is not configured")
    policy = TradeoffConfig(**config.metric_quality)  # type: ignore[arg-type]
    if action == 2:
        limits = {
            name: getattr(policy, name) * config.gaussian_guard_multiplier
            for name in policy.__dataclass_fields__
            if name.startswith("maximum_") and name != "maximum_predicted_violation_probability"
        }
        policy = replace(policy, **limits)
    return policy


def metric_targets(
    changes: list[float], config: InterventionConfig, action: int
) -> tuple[float, float, bool]:
    """Global decision utility, local geometry auxiliary, physical guard harm.

    Geometry mode uses the historical scaled bounded score. Render mode uses
    mean left/right PSNR gain in raw dB. The local geometry auxiliary and
    physical guards retain their canonical definitions in both modes. A stereo
    multiplier is permitted only for geometry utility. Latency is subtracted
    only in the deployment decision, never from these quality labels.
    """
    policy = metric_action_policy(config, action)
    assert config.metric_scales is not None
    utility = outcome_utility(changes, config.metric_scales, policy)
    if config.metric_utility_mode == "render_psnr":
        # Physical vectors order the two PSNR deltas before SSIM. No latency
        # subtraction here: select_repairability is the sole cost-penalty owner.
        utility = (
            sum(
                changes[COMPONENT_NAMES.index(name)]
                for name in ("left_psnr_gain_db", "right_psnr_gain_db")
            )
            / 2
        )
    local = [0.0] * len(COMPONENT_NAMES)
    index = COMPONENT_NAMES.index("local_depth_mae_gain_mm")
    local[index] = changes[index]
    local_gain = outcome_utility(local, config.metric_scales, policy)
    if action == 1:
        utility *= config.stereo_utility_multiplier
        local_gain *= config.stereo_utility_multiplier
    return utility, local_gain, not feasible_changes(changes, policy)


def metric_risk_targets(
    changes: list[float], config: InterventionConfig, action: int
) -> tuple[bool, bool, bool]:
    """Canonical action-conditioned local/global/any labels for offline supervision."""
    return grouped_harm(changes, metric_action_policy(config, action))
