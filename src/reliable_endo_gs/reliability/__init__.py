"""P1b reliability-only diagnostics.

This namespace is intentionally separate from the historical P1 uncertainty
proxies and from the probabilistic Gaussian representation.  Its outputs are
rank-oriented diagnostics; they are not calibrated ``sigma_d`` values.
"""

from reliable_endo_gs.reliability.p1b import (
    P1B_COVERAGE_LEVELS,
    P1B_DEVELOPMENT_SEQUENCES,
    P1B_FINAL_SEQUENCES,
    P1B_FIT_SEQUENCES,
    P1B_HOLDOUT_SEQUENCES,
    CandidateSpec,
    FrameSignalRecord,
    GeometryFeatures,
    PhotometricFeatures,
    TrajectoryFeatures,
    bootstrap_mean,
    build_p1b_signals,
    candidate_specs,
    classify_candidate,
    compute_geometry_features,
    compute_photometric_features,
    compute_trajectory_features,
    evaluate_frame_signal,
    shadow_diagnostics,
    summarize_candidate_rows,
    validate_p1b_sequences,
)

__all__ = [
    "CandidateSpec",
    "FrameSignalRecord",
    "GeometryFeatures",
    "P1B_COVERAGE_LEVELS",
    "P1B_DEVELOPMENT_SEQUENCES",
    "P1B_FINAL_SEQUENCES",
    "P1B_FIT_SEQUENCES",
    "P1B_HOLDOUT_SEQUENCES",
    "PhotometricFeatures",
    "TrajectoryFeatures",
    "build_p1b_signals",
    "bootstrap_mean",
    "candidate_specs",
    "classify_candidate",
    "compute_geometry_features",
    "compute_photometric_features",
    "compute_trajectory_features",
    "evaluate_frame_signal",
    "shadow_diagnostics",
    "summarize_candidate_rows",
    "validate_p1b_sequences",
]
