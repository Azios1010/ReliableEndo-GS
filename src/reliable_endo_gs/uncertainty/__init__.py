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
    left_right_consistency_score,
    photometric_residual_score,
)
from reliable_endo_gs.uncertainty.extraction import (
    P1_PROXY_IDS,
    ProxyOutputs,
    availability_table,
    extract_model_iterations,
    extract_model_proxies,
    extract_proxy_outputs,
)
from reliable_endo_gs.uncertainty.records import CalibratedDisparitySigma, RawUncertaintyScore

__all__ = [
    "CalibratedDisparitySigma",
    "CalibratedProxySigmaProvider",
    "FinalUpdateMagnitudeProxy",
    "IterationDisagreementProxy",
    "P1_PROXY_IDS",
    "ProxyOutputs",
    "LaplaceUncertaintyHead",
    "LearnedLaplaceSigmaProvider",
    "OracleUncertaintyTarget",
    "RawUncertaintyScore",
    "TemperatureCalibrator",
    "UncertaintyProviderProvenance",
    "build_oracle_target",
    "availability_table",
    "extract_model_iterations",
    "extract_model_proxies",
    "extract_proxy_outputs",
    "left_right_consistency_score",
    "laplace_nll",
    "photometric_residual_score",
    "sigma_from_laplace_scale",
]
