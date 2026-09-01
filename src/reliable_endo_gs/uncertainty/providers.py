"""Inference-only provider contracts for the frozen Phase-I sigma boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from reliable_endo_gs.contracts import StereoPrediction
from reliable_endo_gs.uncertainty.records import CalibratedDisparitySigma, RawUncertaintyScore

UNCERTAINTY_FEATURES_KEY = "uncertainty_features"
UNCERTAINTY_FEATURE_SCHEMA = "audited_pre_render_stereo_features.v1"
CALIBRATED_SIGMA_SCHEMA = "calibrated_disparity_sigma.v1"


@dataclass(frozen=True, slots=True)
class UncertaintyProviderProvenance:
    """Auditable identity for one deployable uncertainty provider.

    Identities are supplied by the experiment/artifact system.  This class does
    not invent hashes; ``synthetic_untrained`` is the explicit checkpoint value
    for a deterministic smoke provider that has never been trained.
    """

    provider_id: str
    architecture_id: str
    config_identity: str
    checkpoint_identity: str
    schema_version: str
    calibration_id: str
    feature_schema: str | None = None

    def __post_init__(self) -> None:
        fields = (
            self.provider_id,
            self.architecture_id,
            self.config_identity,
            self.checkpoint_identity,
            self.schema_version,
            self.calibration_id,
        )
        if any(not isinstance(value, str) or not value for value in fields):
            raise ValueError("provider provenance identities must be non-empty strings")
        if self.feature_schema is not None and not self.feature_schema:
            raise ValueError("feature_schema must be non-empty when supplied")

    @property
    def architecture(self) -> str:
        """Backward-friendly alias for the architecture identity."""

        return self.architecture_id

    @property
    def config_id(self) -> str:
        """Backward-friendly alias for the config identity."""

        return self.config_identity

    @property
    def checkpoint_id(self) -> str:
        """Backward-friendly alias for the checkpoint identity."""

        return self.checkpoint_identity


class RawScoreProvider(Protocol):
    """Deployable raw-score interface: only pre-action stereo prediction enters."""

    def predict(self, prediction: StereoPrediction) -> RawUncertaintyScore:
        """Return a rank score; implementations must not consume ground truth."""


class SigmaProvider(Protocol):
    """Deployable calibrated-sigma interface for downstream geometry consumers."""

    def predict(self, prediction: StereoPrediction) -> CalibratedDisparitySigma:
        """Return canonical calibrated standard deviation in disparity pixels."""


class CalibrationApplier(Protocol):
    """Minimal frozen-calibration boundary shared by proxy providers."""

    method: str
    fit_split_identity: str

    def apply(
        self,
        raw_sigma_d: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        provider_id: str,
    ) -> CalibratedDisparitySigma:
        """Apply calibration without accepting ground-truth targets."""


class CalibratedProxySigmaProvider:
    """Convert a non-negative proxy score to sigma and apply frozen temperature.

    The score floor is explicit because zero iteration disagreement is a valid
    raw observation but cannot be passed into a probabilistic likelihood.
    """

    def __init__(
        self,
        raw_provider: RawScoreProvider,
        calibrator: CalibrationApplier,
        *,
        score_floor: float = 1e-4,
        provenance: UncertaintyProviderProvenance | None = None,
    ) -> None:
        if score_floor <= 0:
            raise ValueError("score_floor must be strictly positive")
        self.raw_provider = raw_provider
        self.calibrator = calibrator
        self.score_floor = score_floor
        estimator_id = getattr(raw_provider, "estimator_id", "proxy")
        self.provenance = provenance or UncertaintyProviderProvenance(
            provider_id=str(estimator_id),
            architecture_id=f"proxy:{estimator_id}",
            config_identity="proxy_default",
            checkpoint_identity="not_applicable",
            schema_version=CALIBRATED_SIGMA_SCHEMA,
            calibration_id=f"{calibrator.method}:{calibrator.fit_split_identity}",
        )
        if self.provenance.feature_schema is not None:
            raise ValueError("proxy provenance must not declare learned feature schema")

    def predict(self, prediction: StereoPrediction) -> CalibratedDisparitySigma:
        """Produce deployable sigma without receiving an oracle target."""

        raw = self.raw_provider.predict(prediction)
        valid_scores = raw.score[raw.valid_mask]
        if not bool((valid_scores >= 0).all()):
            raise ValueError("calibrated proxy scores must be non-negative where valid")
        # Only an exact zero is floored. Negative scores are rejected above,
        # rather than silently changing their rank or scientific meaning.
        raw_sigma = torch.where(
            raw.score == 0, torch.full_like(raw.score, self.score_floor), raw.score
        )
        return self.calibrator.apply(
            raw_sigma, raw.valid_mask, provider_id=self.provenance.provider_id
        )
