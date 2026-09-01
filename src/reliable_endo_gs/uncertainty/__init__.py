"""Deployable Phase-I disparity uncertainty estimators and calibration."""

from reliable_endo_gs.uncertainty.calibration import TemperatureCalibrator
from reliable_endo_gs.uncertainty.head import LaplaceUncertaintyHead
from reliable_endo_gs.uncertainty.laplace import laplace_nll, sigma_from_laplace_scale
from reliable_endo_gs.uncertainty.oracle import OracleUncertaintyTarget, build_oracle_target
from reliable_endo_gs.uncertainty.providers import (
    CalibratedProxySigmaProvider,
    LearnedLaplaceSigmaProvider,
    UncertaintyProviderProvenance,
)
from reliable_endo_gs.uncertainty.proxies import (
    FinalUpdateMagnitudeProxy,
    IterationDisagreementProxy,
)
from reliable_endo_gs.uncertainty.records import CalibratedDisparitySigma, RawUncertaintyScore

__all__ = [
    "CalibratedDisparitySigma",
    "CalibratedProxySigmaProvider",
    "FinalUpdateMagnitudeProxy",
    "IterationDisagreementProxy",
    "LaplaceUncertaintyHead",
    "LearnedLaplaceSigmaProvider",
    "OracleUncertaintyTarget",
    "RawUncertaintyScore",
    "TemperatureCalibrator",
    "UncertaintyProviderProvenance",
    "build_oracle_target",
    "laplace_nll",
    "sigma_from_laplace_scale",
]
