"""Reciprocal calibration evidence on partial stereo surfaces; no threshold relaxation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from reliable_endo_gs.data.scared_a00_preprocess import (
    ScaredA00PreprocessError,
    _cross_view_nearest_neighbor_metrics,
    _verify_cross_view_evidence,
)


def _clouds(extra: int = 4, right_error_mm: float = 0.0) -> np.ndarray:
    common = np.column_stack((np.arange(1, 5) * 10.0, np.zeros(4), np.full(4, 100.0)))
    left_only = np.column_stack(
        (60 + np.arange(extra) * 5.0, np.zeros(extra), np.full(extra, 100.0))
    )
    right_only = np.column_stack(
        (-50 - np.arange(extra) * 5.0, np.zeros(extra), np.full(extra, 100.0))
    )
    right_common = common + np.array([-1.0, right_error_mm, 0.0])
    return np.stack(
        (np.concatenate((common, left_only)), np.concatenate((right_common, right_only)))
    )


def _check(measured: dict[str, object], expected: dict[str, object] | None = None) -> None:
    config = yaml.safe_load(Path("configs/data/a00_scared_sample_v1.yaml").read_text())
    _verify_cross_view_evidence(
        measured,
        measured if expected is None else expected,
        declared_unit="mm",
        declared_direction="right_from_left",
        tolerance_mm=0.0,
        acceptance=config["validation"]["cross_view_acceptance"],
    )


def test_partial_surfaces_need_reciprocal_correspondence() -> None:
    stacked = _clouds()
    legacy = _cross_view_nearest_neighbor_metrics(
        stacked, np.eye(3), [-1, 0, 0], declared_translation_unit="mm"
    )
    with pytest.raises(ScaredA00PreprocessError, match="absolute cross-view residual"):
        _check(legacy)
    measured = _cross_view_nearest_neighbor_metrics(
        stacked,
        np.eye(3),
        [-1, 0, 0],
        declared_translation_unit="mm",
        correspondence_policy="mutual",
    )
    _check(measured)
    assert measured["point_counts"] == {"left": 4, "right": 4}
    assert measured["correspondence_counts"] == {
        "raw_left": 8,
        "raw_right": 8,
        "mutual_pairs": 4,
        "min_fraction_per_view": 0.25,
        "min_pair_count": 3,
    }
    assert measured["metrics_mm"]["translation_as_mm"]["p99"] == 0.0
    for name in (
        "translation_as_cm",
        "translation_as_m",
        "wrong_direction_reuse_declared_transform",
    ):
        assert measured["metrics_mm"][name]["p90"] > 0.05


def test_mutual_pairing_does_not_discard_large_matching_residuals() -> None:
    measured = _cross_view_nearest_neighbor_metrics(
        _clouds(right_error_mm=2.0),
        np.eye(3),
        [-1, 0, 0],
        declared_translation_unit="mm",
        correspondence_policy="mutual",
    )
    assert measured["metrics_mm"]["translation_as_mm"]["median"] == pytest.approx(2.0)
    with pytest.raises(ScaredA00PreprocessError, match="absolute cross-view residual"):
        _check(measured)


def test_small_accidental_intersection_cannot_establish_calibration() -> None:
    with pytest.raises(ScaredA00PreprocessError, match="insufficient mutual"):
        _cross_view_nearest_neighbor_metrics(
            _clouds(extra=24),
            np.eye(3),
            [-1, 0, 0],
            declared_translation_unit="mm",
            correspondence_policy="mutual",
        )


@pytest.mark.parametrize("translation", [[-1000, 0, 0], [1, 0, 0]])
def test_wrong_scale_or_direction_stays_rejected(translation: list[int]) -> None:
    with pytest.raises(ScaredA00PreprocessError):
        measured = _cross_view_nearest_neighbor_metrics(
            _clouds(),
            np.eye(3),
            translation,
            declared_translation_unit="mm",
            correspondence_policy="mutual",
        )
        _check(measured)


def test_correspondence_counts_and_support_policy_must_reproduce() -> None:
    measured = _cross_view_nearest_neighbor_metrics(
        _clouds(),
        np.eye(3),
        [-1, 0, 0],
        declared_translation_unit="mm",
        correspondence_policy="mutual",
    )
    altered = deepcopy(measured)
    altered["correspondence_counts"]["raw_left"] = 7
    with pytest.raises(ScaredA00PreprocessError, match="does not reproduce"):
        _check(measured, altered)
    altered["correspondence_counts"]["min_fraction_per_view"] = 0.1
    with pytest.raises(ScaredA00PreprocessError, match="support policy"):
        _check(measured, altered)


def test_complete_clouds_match_historical_metrics() -> None:
    stacked = _clouds(extra=0)
    legacy = _cross_view_nearest_neighbor_metrics(
        stacked, np.eye(3), [-1, 0, 0], declared_translation_unit="mm"
    )
    mutual = _cross_view_nearest_neighbor_metrics(
        stacked,
        np.eye(3),
        [-1, 0, 0],
        declared_translation_unit="mm",
        correspondence_policy="mutual",
    )
    assert legacy["metrics_mm"] == mutual["metrics_mm"]
    assert legacy["point_counts"] == mutual["point_counts"]
