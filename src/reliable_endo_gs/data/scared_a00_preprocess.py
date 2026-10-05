"""Reproducible one-sample preprocessing for original SCARED A00 inputs.

This module consumes the original stacked video plus the matching raw
``frame_data`` and ``scene_points`` archive members. It never extracts files
into the source dataset and writes processed data only to a caller-selected
external output root. The pinned calibration YAML supplies numeric R/T values
only; transform direction and translation units are resolved through measured
paired-cloud and metric-ground-truth validation.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ScaredA00PreprocessError(ValueError):
    """Raised when raw SCARED inputs or A00 preprocessing evidence are invalid."""


_SAMPLE_PATTERN = re.compile(r"^(dataset_[1-9])/(keyframe_[1-5])/(frame_data\d{6})$")
_HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_CALIBRATION_FIELD = "frame_data.camera-calibration.R/T"
_METRIC_GT_URL = "https://endovissub2019-scared.grand-challenge.org/About/"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_bytes(value: bytes, name: str) -> dict[str, object]:
    try:
        parsed: object = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScaredA00PreprocessError(f"invalid JSON in {name}: {error}") from error
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise ScaredA00PreprocessError(f"{name} must contain a JSON object")
    return parsed


def _verify_calibration_yaml_matches_frame_data(
    path: Path,
    frame_calibration: Mapping[str, object],
    *,
    expected_sha256: str,
) -> None:
    """Hash the pinned OpenCV YAML and verify its R/T match the selected JSON."""
    import numpy as np

    if not path.is_file():
        raise ScaredA00PreprocessError(f"pinned calibration YAML is missing: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256.casefold() != expected_sha256.casefold():
        raise ScaredA00PreprocessError(
            "selected endoscope_calibration.yaml SHA-256 differs from the pinned numeric value source"
        )
    try:
        import cv2
    except ImportError as error:
        raise ScaredA00PreprocessError(
            "OpenCV is required to verify the pinned calibration YAML R/T values"
        ) from error
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        if not storage.isOpened():
            raise ScaredA00PreprocessError(f"unable to open OpenCV calibration YAML: {path}")
        yaml_rotation = storage.getNode("R").mat()
        yaml_translation = storage.getNode("T").mat()
    except cv2.error as error:
        raise ScaredA00PreprocessError(
            f"unable to read OpenCV R/T matrices from calibration YAML {path}: {error}"
        ) from error
    finally:
        storage.release()
    if yaml_rotation is None or yaml_translation is None:
        raise ScaredA00PreprocessError("calibration YAML must contain OpenCV R and T matrices")
    try:
        json_rotation = np.asarray(frame_calibration.get("R"), dtype=np.float64).reshape(3, 3)
        json_translation = np.asarray(frame_calibration.get("T"), dtype=np.float64).reshape(3)
        yaml_rotation_array = np.asarray(yaml_rotation, dtype=np.float64).reshape(3, 3)
        yaml_translation_array = np.asarray(yaml_translation, dtype=np.float64).reshape(3)
    except (TypeError, ValueError) as error:
        raise ScaredA00PreprocessError(
            "calibration YAML and selected frame_data must contain numeric 3x3 R and 3-vector T"
        ) from error
    if (
        not np.isfinite(json_rotation).all()
        or not np.isfinite(json_translation).all()
        or not np.isfinite(yaml_rotation_array).all()
        or not np.isfinite(yaml_translation_array).all()
        or not np.array_equal(yaml_rotation_array, json_rotation)
        or not np.array_equal(yaml_translation_array, json_translation)
    ):
        raise ScaredA00PreprocessError(
            "pinned calibration YAML R/T do not exactly match the selected frame_data JSON"
        )


def _required_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ScaredA00PreprocessError(f"{name} must be a string-keyed mapping")
    return dict(value)


def _read_config(path: Path) -> tuple[dict[str, object], str]:
    try:
        import yaml
    except ImportError as error:
        raise ScaredA00PreprocessError("PyYAML is required to load the A00 preprocessing config") from error
    try:
        contents = path.read_bytes()
        parsed: object = yaml.safe_load(contents)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ScaredA00PreprocessError(f"unable to load preprocessing config {path}: {error}") from error
    config = _required_mapping(parsed, "preprocessing config")
    if config.get("schema_version") != 1:
        raise ScaredA00PreprocessError("preprocessing config schema_version must be 1")
    if config.get("preprocessing_id") != "reliable_endo_gs_a00_scared_sample_v1":
        raise ScaredA00PreprocessError("unexpected A00 preprocessing_id")
    return config, _sha256_bytes(contents)


def _validate_calibration_evidence(
    evidence: Mapping[str, object],
    *,
    sample_id: str,
    trusted_value_source_sha256: str | None,
) -> dict[str, object]:
    """Validate the numeric calibration source and empirical geometry resolutions."""
    if evidence.get("schema_version") != 1 or evidence.get("sample_id") != sample_id:
        raise ScaredA00PreprocessError(
            "calibration evidence must be schema 1 and identify the exact requested sample"
        )
    calibration = _required_mapping(evidence.get("calibration"), "evidence.calibration")
    if calibration.get("source_field") != _CALIBRATION_FIELD:
        raise ScaredA00PreprocessError(
            f"calibration evidence must identify {_CALIBRATION_FIELD} exactly"
        )
    unit = calibration.get("translation_unit")
    direction = calibration.get("transform_direction")
    if unit != "mm":
        raise ScaredA00PreprocessError(
            "calibration translation unit must be resolved as mm by the metric cross-view gate"
        )
    if direction != "right_from_left":
        raise ScaredA00PreprocessError(
            "calibration direction must be empirically resolved as right_from_left"
        )
    value_source = _required_mapping(
        calibration.get("calibration_value_source"),
        "evidence.calibration.calibration_value_source",
    )
    if any(
        key in value_source
        for key in (
            "translation_unit",
            "unit",
            "unit_statement",
            "transform_direction",
            "direction_statement",
            "mapping_convention",
            "camera1",
            "camera2",
        )
    ):
        raise ScaredA00PreprocessError(
            "calibration value source must not assert unit or transform-direction semantics"
        )
    expected_source_path = f"{sample_id.rsplit('/', maxsplit=1)[0]}/endoscope_calibration.yaml"
    for key in ("field_path", "source_title", "source_uri"):
        value = value_source.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ScaredA00PreprocessError(
                f"calibration_value_source.{key} must identify the numeric calibration source"
            )
    if value_source.get("field_path") != "R/T":
        raise ScaredA00PreprocessError(
            "calibration value source must identify its R/T numeric fields"
        )
    if value_source.get("source_path") != expected_source_path:
        raise ScaredA00PreprocessError(
            "calibration value source must identify the selected keyframe endoscope_calibration.yaml"
        )
    source_digest = value_source.get("source_sha256")
    if not isinstance(source_digest, str) or _HEX_SHA256.fullmatch(source_digest) is None:
        raise ScaredA00PreprocessError(
            "calibration_value_source.source_sha256 must be a full SHA-256"
        )
    if (
        trusted_value_source_sha256 is None
        or _HEX_SHA256.fullmatch(trusted_value_source_sha256) is None
    ):
        raise ScaredA00PreprocessError(
            "no calibration value-source SHA-256 is pinned in the preprocessing config"
        )
    if source_digest.casefold() != trusted_value_source_sha256.casefold():
        raise ScaredA00PreprocessError(
            "calibration value-source SHA-256 does not match the pinned YAML"
        )

    direction_resolution = _required_mapping(
        evidence.get("direction_resolution"), "evidence.direction_resolution"
    )
    if (
        direction_resolution.get("method") != "cross_view_nearest_neighbor"
        or direction_resolution.get("resolution_basis") != "empirical_paired_cloud_residuals"
        or direction_resolution.get("selected_direction") != direction
        or direction_resolution.get("candidate_transform") != "R @ left_xyz + T -> right_xyz"
        or direction_resolution.get("direction_control") != "R @ right_xyz + T -> left_xyz"
        or direction_resolution.get("scene_points_layout") != "top_left_bottom_right_xyz"
        or direction_resolution.get("metric_ground_truth_source_uri") != _METRIC_GT_URL
    ):
        raise ScaredA00PreprocessError(
            "left/right direction must be resolved by measured paired-cloud residuals and controls"
        )

    unit_resolution = _required_mapping(evidence.get("unit_resolution"), "evidence.unit_resolution")
    if (
        unit_resolution.get("method") != "cross_view_nearest_neighbor"
        or unit_resolution.get("resolution_basis")
        != "empirical_cross_view_against_metric_ground_truth"
        or unit_resolution.get("selected_translation_unit") != "mm"
        or unit_resolution.get("metric_ground_truth_source_uri") != _METRIC_GT_URL
    ):
        raise ScaredA00PreprocessError(
            "translation unit must be resolved as mm by cross-view validation against official metric SCARED GT"
        )

    validation = _required_mapping(
        evidence.get("independent_validation"), "evidence.independent_validation"
    )
    if validation.get("method") != "cross_view_nearest_neighbor":
        raise ScaredA00PreprocessError(
            "independent validation must use the declared cross-view nearest-neighbor check"
        )
    if validation.get("valid_point_policy") not in (
        "finite_nonzero_xyz_positive_z",
        "finite_nonzero_xyz_positive_z_mutual_nearest_neighbor",
    ):
        raise ScaredA00PreprocessError(
            "cross-view evidence must declare a supported point-validity/correspondence policy"
        )
    if (
        validation.get("valid_point_policy")
        == "finite_nonzero_xyz_positive_z_mutual_nearest_neighbor"
    ):
        _validate_mutual_correspondence_counts(validation)
    point_counts = _required_mapping(validation.get("point_counts"), "validation.point_counts")
    for key in ("left", "right"):
        if type(point_counts.get(key)) is not int or int(point_counts[key]) < 1:
            raise ScaredA00PreprocessError(
                f"cross-view point_counts.{key} must be a positive integer"
            )
    metrics = _required_mapping(validation.get("metrics_mm"), "validation.metrics_mm")
    for transform_name in (
        "translation_as_mm",
        "translation_as_cm",
        "translation_as_m",
        "wrong_direction_reuse_declared_transform",
    ):
        summary = _required_mapping(metrics.get(transform_name), f"metrics_mm.{transform_name}")
        for key in ("median", "p90", "p99"):
            value = summary.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ScaredA00PreprocessError(
                    f"metrics_mm.{transform_name}.{key} must be finite and non-negative"
                )
    correspondence = _required_mapping(
        evidence.get("frame_correspondence"), "evidence.frame_correspondence"
    )
    expected_frame = sample_id.rsplit("/", maxsplit=1)[-1]
    if (
        correspondence.get("frame_data_id") != expected_frame
        or correspondence.get("scene_points_id")
        != expected_frame.replace("frame_data", "scene_points")
        or correspondence.get("matched") is not True
        or correspondence.get("ambiguous") is not False
    ):
        raise ScaredA00PreprocessError(
            "evidence must verify one unambiguous frame_data/scene_points ID match"
        )
    metric_gt = _required_mapping(
        evidence.get("metric_ground_truth"), "evidence.metric_ground_truth"
    )
    if metric_gt.get("xyz_units") != "mm" or metric_gt.get("depth_units") != "mm":
        raise ScaredA00PreprocessError(
            "metric ground-truth evidence must declare scene XYZ and depth in millimetres"
        )
    if metric_gt.get("source_uri") != _METRIC_GT_URL:
        raise ScaredA00PreprocessError(
            "metric ground-truth source must be the official SCARED challenge documentation"
        )
    member_hashes = _required_mapping(evidence.get("source_member_sha256"), "source_member_sha256")
    for key in ("frame_data", "scene_points"):
        value = member_hashes.get(key)
        if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
            raise ScaredA00PreprocessError(
                f"source_member_sha256.{key} must bind evidence to the exact selected archive member"
            )
    frame_match = _required_mapping(
        value_source.get("frame_data_match"), "calibration_value_source.frame_data_match"
    )
    if (
        frame_match.get("source_field") != _CALIBRATION_FIELD
        or frame_match.get("frame_data_member_sha256") != member_hashes.get("frame_data")
        or frame_match.get("R_matches") is not True
        or frame_match.get("T_matches") is not True
    ):
        raise ScaredA00PreprocessError(
            "calibration YAML R/T must be recorded as matching the selected frame_data member"
        )
    return {
        "calibration": calibration,
        "calibration_value_source": value_source,
        "direction_resolution": direction_resolution,
        "unit_resolution": unit_resolution,
        "independent_validation": validation,
        "frame_correspondence": correspondence,
        "metric_ground_truth": metric_gt,
        "source_member_sha256": member_hashes,
    }


def _cross_view_nearest_neighbor_metrics(
    stacked_scene_points: Any,
    rotation: Any,
    translation: Any,
    *,
    declared_translation_unit: str,
    correspondence_policy: str = "all_valid",
) -> dict[str, object]:
    """Compare absolute mm/cm/m hypotheses against both matched XYZ views.

    Uses exact single-neighbor queries from SciPy's cKDTree. ``all_valid``
    reproduces historical evidence. ``mutual`` retains reciprocal nearest
    neighbours under the declared transform, without a residual cutoff. All
    unit/direction controls query the same retained clouds; alternative units
    still get their own best nearest neighbour, not a forced correspondence.
    Partial surfaces and unequal raster sampling do not establish stereo
    correspondences by a one-way nearest query alone. The mutual policy
    requires at least 25% of each valid cloud and at least three matched pairs
    so a tiny accidental intersection cannot establish calibration semantics.
    These support gates are part of the recorded policy, not metric tolerances.
    """
    import numpy as np

    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise ScaredA00PreprocessError(
            "A00 provenance verification requires scipy.spatial.cKDTree; no dependency was installed"
        ) from error
    stacked = np.asarray(stacked_scene_points, dtype=np.float64)
    if stacked.ndim != 3 or stacked.shape[-1] != 3 or stacked.shape[0] % 2:
        raise ScaredA00PreprocessError("scene_points must have stacked [2H,W,3] XYZ layout")
    height = stacked.shape[0] // 2
    left_raw = stacked[:height].reshape(-1, 3)
    right_raw = stacked[height:].reshape(-1, 3)

    def valid_xyz(points: Any) -> Any:
        return points[
            np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1) & (points[:, 2] > 0)
        ]

    left = valid_xyz(left_raw)
    right = valid_xyz(right_raw)
    if not len(left) or not len(right):
        raise ScaredA00PreprocessError("both scene_points views must have valid metric XYZ")
    unit_scale = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
    if declared_translation_unit not in unit_scale:
        raise ScaredA00PreprocessError(
            "unsupported declared translation unit for cross-view evidence"
        )
    rotation_array = np.asarray(rotation, dtype=np.float64)
    translation_raw = np.asarray(translation, dtype=np.float64).reshape(3)
    if not np.isfinite(rotation_array).all() or not np.isfinite(translation_raw).all():
        raise ScaredA00PreprocessError("raw cross-view R/T must contain only finite values")
    if correspondence_policy not in {"all_valid", "mutual"}:
        raise ScaredA00PreprocessError("unsupported cross-view correspondence policy")
    correspondence_counts: dict[str, object] | None = None
    policy = "finite_nonzero_xyz_positive_z"
    if correspondence_policy == "mutual":
        raw_left_count, raw_right_count = len(left), len(right)
        mapped_left = (rotation_array @ left.T).T + translation_raw * unit_scale[
            declared_translation_unit
        ]
        _, nearest_right = cKDTree(right).query(mapped_left, k=1, workers=1)
        _, nearest_left = cKDTree(mapped_left).query(right, k=1, workers=1)
        mutual = nearest_left[nearest_right] == np.arange(len(left))
        left = left[mutual]
        right = right[nearest_right[mutual]]
        correspondence_counts = {
            "raw_left": raw_left_count,
            "raw_right": raw_right_count,
            "mutual_pairs": int(len(left)),
            "min_fraction_per_view": 0.25,
            "min_pair_count": 3,
        }
        policy = "finite_nonzero_xyz_positive_z_mutual_nearest_neighbor"
        _validate_mutual_correspondence_counts({"correspondence_counts": correspondence_counts})
    tree_right = cKDTree(right)
    tree_left = cKDTree(left)

    def distances(values: Any, tree: Any) -> dict[str, float]:
        result, _ = tree.query(values, k=1, workers=1)
        return {
            "median": float(np.median(result)),
            "p90": float(np.percentile(result, 90)),
            "p99": float(np.percentile(result, 99)),
        }

    rotated_left = (rotation_array @ left.T).T
    translated_left = (rotation_array @ right.T).T
    hypotheses = {
        f"translation_as_{unit}": distances(rotated_left + translation_raw * scale, tree_right)
        for unit, scale in unit_scale.items()
    }
    # Direction control is deliberately the common misuse: apply the declared
    # right-from-left R/T to points that are already in the right camera, then
    # query the left cloud. This is distinct from applying the true inverse.
    wrong_direction = translated_left + translation_raw * unit_scale[declared_translation_unit]
    hypotheses["wrong_direction_reuse_declared_transform"] = distances(wrong_direction, tree_left)
    result: dict[str, object] = {
        "valid_point_policy": policy,
        "point_counts": {"left": int(len(left)), "right": int(len(right))},
        "metrics_mm": hypotheses,
    }
    if correspondence_counts is not None:
        result["correspondence_counts"] = correspondence_counts
    return result


def _validate_mutual_correspondence_counts(value: Mapping[str, object]) -> None:
    """Reject unsupported/tiny reciprocal domains without filtering residuals."""
    counts = _required_mapping(value.get("correspondence_counts"), "correspondence_counts")
    if (
        counts.get("min_fraction_per_view") != 0.25
        or type(counts.get("min_pair_count")) is not int
        or counts.get("min_pair_count") != 3
    ):
        raise ScaredA00PreprocessError(
            "mutual correspondence support policy does not match this implementation"
        )
    for key in ("raw_left", "raw_right", "mutual_pairs"):
        if type(counts.get(key)) is not int or int(counts[key]) < 1:
            raise ScaredA00PreprocessError(f"mutual correspondence {key} must be positive")
    pairs = int(counts["mutual_pairs"])
    if pairs < 3 or any(
        pairs < 0.25 * int(counts[key]) or pairs > int(counts[key])
        for key in ("raw_left", "raw_right")
    ):
        raise ScaredA00PreprocessError("insufficient mutual cross-view correspondence support")
    if "point_counts" in value and value["point_counts"] != {"left": pairs, "right": pairs}:
        raise ScaredA00PreprocessError("mutual correspondence point counts are inconsistent")


def _verify_cross_view_evidence(
    measured: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    declared_unit: str,
    declared_direction: str,
    tolerance_mm: float,
    acceptance: Mapping[str, object],
) -> None:
    import math

    if measured.get("valid_point_policy") != expected.get("valid_point_policy"):
        raise ScaredA00PreprocessError("cross-view evidence point-validity policy mismatch")
    if (
        measured.get("valid_point_policy")
        == "finite_nonzero_xyz_positive_z_mutual_nearest_neighbor"
    ):
        _validate_mutual_correspondence_counts(measured)
        _validate_mutual_correspondence_counts(expected)
        if measured.get("correspondence_counts") != expected.get("correspondence_counts"):
            raise ScaredA00PreprocessError(
                "cross-view mutual correspondence evidence does not reproduce"
            )
    measured_counts = _required_mapping(measured.get("point_counts"), "measured point_counts")
    expected_counts = _required_mapping(expected.get("point_counts"), "evidence point_counts")
    if measured_counts != expected_counts:
        raise ScaredA00PreprocessError(
            f"cross-view evidence point counts do not match selected scene_points: {measured_counts} vs {expected_counts}"
        )
    measured_metrics = _required_mapping(measured.get("metrics_mm"), "measured metrics_mm")
    expected_metrics = _required_mapping(expected.get("metrics_mm"), "evidence metrics_mm")
    expected_names = {
        "translation_as_mm",
        "translation_as_cm",
        "translation_as_m",
        "wrong_direction_reuse_declared_transform",
    }
    if set(measured_metrics) != expected_names or set(expected_metrics) != expected_names:
        raise ScaredA00PreprocessError(
            "cross-view evidence is missing an absolute unit/direction control"
        )
    if not math.isfinite(tolerance_mm) or tolerance_mm < 0:
        raise ScaredA00PreprocessError(
            "cross-view evidence tolerance must be finite and non-negative"
        )
    for transform_name in sorted(expected_names):
        measured_summary_obj = measured_metrics[transform_name]
        measured_summary = _required_mapping(measured_summary_obj, f"measured {transform_name}")
        expected_summary = _required_mapping(
            expected_metrics.get(transform_name), f"evidence {transform_name}"
        )
        for statistic in ("median", "p90", "p99"):
            actual = float(measured_summary[statistic])
            recorded = float(expected_summary[statistic])
            if (
                not math.isfinite(actual)
                or actual < 0
                or not math.isfinite(recorded)
                or recorded < 0
                or abs(actual - recorded) > tolerance_mm
            ):
                raise ScaredA00PreprocessError(
                    f"cross-view {transform_name}.{statistic} does not reproduce the evidence manifest"
                )

    if declared_direction != "right_from_left":
        raise ScaredA00PreprocessError(
            "only right_from_left may be selected when its measured paired-cloud residual beats the opposite-use control"
        )
    if declared_unit != "mm":
        raise ScaredA00PreprocessError(
            "only the mm calibration hypothesis may be selected by metric-ground-truth validation"
        )
    declared_name = "translation_as_mm"
    declared = _required_mapping(measured_metrics[declared_name], f"measured {declared_name}")
    max_median = float(acceptance.get("max_declared_median_mm", -1.0))
    max_p90 = float(acceptance.get("max_declared_p90_mm", -1.0))
    max_p99 = float(acceptance.get("max_declared_p99_mm", -1.0))
    min_control_ratio = float(acceptance.get("min_control_to_declared_p90_ratio", -1.0))
    min_control_gap = float(acceptance.get("min_control_p90_gap_mm", -1.0))
    declared_median = float(declared["median"])
    declared_p90 = float(declared["p90"])
    declared_p99 = float(declared["p99"])
    acceptance_values = (max_median, max_p90, max_p99, min_control_ratio, min_control_gap)
    if (
        not all(math.isfinite(value) for value in acceptance_values)
        or min(max_median, max_p90, max_p99, min_control_gap) <= 0
        or min_control_ratio <= 1.0
    ):
        raise ScaredA00PreprocessError(
            "cross-view acceptance thresholds must be finite, positive, and require control separation"
        )
    if declared_median > max_median or declared_p90 > max_p90 or declared_p99 > max_p99:
        raise ScaredA00PreprocessError(
            "declared calibration unit fails configured absolute cross-view residual thresholds"
        )
    controls = expected_names - {declared_name}
    for control_name in controls:
        control = _required_mapping(measured_metrics[control_name], f"measured {control_name}")
        control_p90 = float(control["p90"])
        if (
            control_p90 <= declared_p90
            or control_p90 - declared_p90 < min_control_gap
            or control_p90 < declared_p90 * min_control_ratio
        ):
            raise ScaredA00PreprocessError(
                f"declared unit/direction does not outperform cross-view control {control_name}"
            )


def _as_array(value: object, *, name: str, shape: tuple[int, ...] | None = None) -> Any:
    import numpy as np

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ScaredA00PreprocessError(f"{name} must be numeric") from error
    if shape is not None and array.shape != shape:
        raise ScaredA00PreprocessError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ScaredA00PreprocessError(f"{name} must contain only finite values")
    return array


def _unit_to_m(unit: str) -> float:
    scales = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
    try:
        return scales[unit]
    except KeyError as error:
        raise ScaredA00PreprocessError(f"unsupported explicitly verified unit {unit!r}") from error


def _rectified_disparity_from_left_points(
    left_points: Any,
    rotation_left_rectified: Any,
    projection_left: Any,
    projection_right: Any,
    image_hw: tuple[int, int],
    *,
    allow_empty_gt: bool = False,
) -> tuple[Any, Any, dict[str, object] | None]:
    """Project left-camera XYZ into P1/P2 and rasterize left-reference disparity.

    Scene-point XYZ remains offline supervision. Invalid zeros/nonfinite points,
    points behind either camera, and points outside the corresponding image
    support receive disparity zero. Collisions keep the nearest positive z.
    Batch preprocessing may retain an empty target explicitly; geometry failures
    (camera z, scanlines or disparity sign) still raise even in that mode.
    """
    import numpy as np

    height, width = image_hw
    points = np.asarray(left_points, dtype=np.float64)
    if points.shape != (height, width, 3):
        raise ScaredA00PreprocessError(
            f"left scene_points must be {(height, width, 3)}, got {points.shape}"
        )
    finite = np.isfinite(points).all(axis=-1)
    valid = finite & np.any(points != 0, axis=-1) & (points[..., 2] > 0)
    raw = points[valid]
    disparity = np.zeros((height, width), dtype=np.float32)
    metric_depth = np.zeros((height, width), dtype=np.float32)
    diagnostics: dict[str, object] = {"valid_left_xyz_count": int(len(raw))}
    if raw.size == 0:
        if allow_empty_gt:
            return disparity, metric_depth, {**diagnostics, "reason": "no_valid_left_xyz"}
        raise ScaredA00PreprocessError("selected scene_points member has no valid left-camera XYZ")

    rectified = (rotation_left_rectified @ raw.T).T
    left_h = (projection_left[:, :3] @ rectified.T).T + projection_left[:, 3]
    right_h = (projection_right[:, :3] @ rectified.T).T + projection_right[:, 3]
    left_z, right_z = left_h[:, 2], right_h[:, 2]
    positive_z = (left_z > 0) & (right_z > 0)
    u_left = left_h[:, 0] / left_z
    v_left = left_h[:, 1] / left_z
    u_right = right_h[:, 0] / right_z
    v_right = right_h[:, 1] / right_z
    disparity_values = u_left - u_right
    close_scanline = np.abs(v_left - v_right) <= 1e-3
    in_grid = (
        (u_left >= 0)
        & (u_left < width)
        & (v_left >= 0)
        & (v_left < height)
        & (u_right >= 0)
        & (u_right < width)
        & (v_right >= 0)
        & (v_right < height)
    )
    geometrically_valid = positive_z & close_scanline & (disparity_values > 0)
    usable = geometrically_valid & in_grid
    diagnostics.update(
        {
            "positive_camera_z_count": int(positive_z.sum()),
            "close_scanline_count": int(close_scanline.sum()),
            "positive_disparity_count": int((disparity_values > 0).sum()),
            "in_both_grids_count": int(in_grid.sum()),
            "usable_point_count": int(usable.sum()),
        }
    )
    if not np.any(usable):
        if allow_empty_gt and np.any(geometrically_valid):
            return (
                disparity,
                metric_depth,
                {
                    **diagnostics,
                    "reason": "no_rectified_stereo_overlap",
                },
            )
        raise ScaredA00PreprocessError("scene_points produced no positive rectified disparity")
    u = np.floor(u_left[usable] + 0.5).astype(np.int64)
    v = np.floor(v_left[usable] + 0.5).astype(np.int64)
    inside_rounded = (u < width) & (v < height)
    u, v = u[inside_rounded], v[inside_rounded]
    if len(u) == 0:
        if allow_empty_gt:
            return (
                disparity,
                metric_depth,
                {
                    **diagnostics,
                    "reason": "no_integer_pixel_support",
                },
            )
        raise ScaredA00PreprocessError("no valid metric-GT disparity survived rectification")
    disparity_values = disparity_values[usable][inside_rounded]
    depth_values = rectified[usable, 2][inside_rounded]
    flat_indices = v * width + u
    # Stable ordering makes collision resolution reproducible; choose the
    # closest positive surface point for each integer left-image pixel.
    order = np.lexsort((depth_values, flat_indices))
    ordered_flat = flat_indices[order]
    first = np.r_[True, ordered_flat[1:] != ordered_flat[:-1]]
    selected = order[first]
    out_flat_d = disparity.reshape(-1)
    out_flat_z = metric_depth.reshape(-1)
    out_flat_d[flat_indices[selected]] = disparity_values[selected].astype(np.float32)
    out_flat_z[flat_indices[selected]] = depth_values[selected].astype(np.float32)
    return disparity, metric_depth, None


def rectify_scared_sample_arrays(
    *,
    left_bgr: Any,
    right_bgr: Any,
    stacked_scene_points: Any,
    calibration: Mapping[str, object],
    translation_unit: str,
    scene_points_unit: str = "mm",
    config: Mapping[str, object],
    allow_empty_gt: bool = False,
) -> dict[str, Any]:
    """Rectify one synchronized pair and build GT disparity plus metric checks.

    Image arrays are uint8 BGR [H,W,3]; ``stacked_scene_points`` is XYZ
    [2H,W,3] with left points in its top half. The returned disparity and
    metric GT remain separate from any deployment batch. ``allow_empty_gt`` is
    for batch preparation only: missing support is reported as unavailable,
    never as a successful metric validation. Single-sample readiness is strict.
    """
    import cv2
    import numpy as np

    input_config = _required_mapping(config.get("input"), "config.input")
    rectification_config = _required_mapping(
        config.get("rectification"), "config.rectification"
    )
    validation_config = _required_mapping(config.get("validation"), "config.validation")
    left_image = np.asarray(left_bgr)
    right_image = np.asarray(right_bgr)
    if (
        left_image.ndim != 3
        or left_image.shape != right_image.shape
        or left_image.shape[2] != 3
        or left_image.dtype != np.uint8
    ):
        raise ScaredA00PreprocessError("left/right raw images must be matching uint8 BGR [H,W,3]")
    height, width = left_image.shape[:2]
    expected_hw = (input_config.get("frame_height"), input_config.get("frame_width"))
    if expected_hw != (height, width):
        raise ScaredA00PreprocessError(
            f"raw image grid {(height, width)} differs from pinned config {expected_hw}"
        )
    stacked = np.asarray(stacked_scene_points)
    if stacked.shape != (height * 2, width, 3):
        raise ScaredA00PreprocessError(
            f"stacked scene_points must have shape {(height * 2, width, 3)}, got {stacked.shape}"
        )
    left_points = stacked[:height]

    calibration_field = str(input_config.get("calibration_field"))
    calibration = _required_mapping(calibration, "frame_data.camera-calibration")
    if calibration_field != "camera-calibration":
        raise ScaredA00PreprocessError("unsupported calibration field in pinned config")
    if input_config.get("stacked_video_layout") != "top_left_bottom_right":
        raise ScaredA00PreprocessError("v1 supports only top-half left / bottom-half right video layout")
    if input_config.get("stacked_scene_points_layout") != "top_left_bottom_right_xyz":
        raise ScaredA00PreprocessError(
            "v1 supports only top-half left / bottom-half right stacked XYZ layout"
        )
    k_left = _as_array(calibration.get("KL"), name="KL", shape=(3, 3))
    k_right = _as_array(calibration.get("KR"), name="KR", shape=(3, 3))
    d_left = _as_array(calibration.get("DL"), name="DL").reshape(-1)
    d_right = _as_array(calibration.get("DR"), name="DR").reshape(-1)
    rotation = _as_array(calibration.get("R"), name="R", shape=(3, 3))
    translation = _as_array(calibration.get("T"), name="T").reshape(-1)
    if translation.shape != (3,):
        raise ScaredA00PreprocessError("T must contain exactly three translation components")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=1e-5):
        raise ScaredA00PreprocessError("raw stereo R is not a rigid rotation")
    if np.linalg.det(rotation) <= 0:
        raise ScaredA00PreprocessError("raw stereo R must be right handed")
    unit_to_m = _unit_to_m(translation_unit)

    flags_name = rectification_config.get("flags")
    if flags_name != "CALIB_ZERO_DISPARITY":
        raise ScaredA00PreprocessError("v1 requires the explicit CALIB_ZERO_DISPARITY flag")
    if rectification_config.get("alpha") != 0.0:
        raise ScaredA00PreprocessError("v1 requires explicit alpha=0.0")
    if rectification_config.get("crop") != "none" or rectification_config.get("resize") != "none":
        raise ScaredA00PreprocessError("v1 does not crop or resize the rectified image grid")

    r1, r2, p1, p2, q, roi_left, roi_right = cv2.stereoRectify(
        k_left,
        d_left,
        k_right,
        d_right,
        (width, height),
        rotation,
        translation.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
        newImageSize=(width, height),
    )
    if min(*roi_left[2:], *roi_right[2:]) < 1:
        raise ScaredA00PreprocessError("OpenCV returned an empty valid rectification ROI")
    map_type = rectification_config.get("map_type")
    if map_type != "CV_32FC1":
        raise ScaredA00PreprocessError("v1 requires explicit CV_32FC1 rectification maps")
    map_lx, map_ly = cv2.initUndistortRectifyMap(
        k_left, d_left, r1, p1, (width, height), cv2.CV_32FC1
    )
    map_rx, map_ry = cv2.initUndistortRectifyMap(
        k_right, d_right, r2, p2, (width, height), cv2.CV_32FC1
    )
    border_value = int(rectification_config.get("border_value", -1))
    if border_value != 0:
        raise ScaredA00PreprocessError("v1 requires explicit zero border value")
    interpolation_name = rectification_config.get("image_interpolation")
    if interpolation_name != "INTER_LINEAR":
        raise ScaredA00PreprocessError("v1 requires explicit INTER_LINEAR image interpolation")
    border_mode_name = rectification_config.get("border_mode")
    if border_mode_name != "BORDER_CONSTANT":
        raise ScaredA00PreprocessError("v1 requires explicit BORDER_CONSTANT remap mode")
    rectified_left = cv2.remap(
        left_image,
        map_lx,
        map_ly,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value, border_value, border_value),
    )
    rectified_right = cv2.remap(
        right_image,
        map_rx,
        map_ry,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value, border_value, border_value),
    )
    source_support = np.full((height, width), 255, dtype=np.uint8)
    support_left = cv2.remap(
        source_support,
        map_lx,
        map_ly,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    support_right = cv2.remap(
        source_support,
        map_rx,
        map_ry,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    full_support = bool(np.all(support_left == 255) and np.all(support_right == 255))
    if validation_config.get("full_image_support_required") is not True or not full_support:
        raise ScaredA00PreprocessError(
            "the A00 adapter uses all-true sensor support; rectified RGB must be fully supported"
        )

    relative_rotation = r2 @ rotation @ r1.T
    relative_translation_source = r2 @ translation
    rotation_tolerance = float(validation_config.get("rectified_rotation_atol", -1.0))
    if rotation_tolerance <= 0 or not np.allclose(
        relative_rotation, np.eye(3), atol=rotation_tolerance, rtol=rotation_tolerance
    ):
        raise ScaredA00PreprocessError(
            "rectified right-from-left rotation is not identity within the configured tolerance"
        )
    translation_yz_tolerance_mm = float(
        validation_config.get("rectified_translation_yz_atol_mm", -1.0)
    )
    relative_translation_mm = relative_translation_source * unit_to_m * 1000.0
    if (
        translation_yz_tolerance_mm <= 0
        or float(np.max(np.abs(relative_translation_mm[1:]))) > translation_yz_tolerance_mm
    ):
        raise ScaredA00PreprocessError(
            "rectified translation y/z exceeds the configured tolerance; refusing to discard it"
        )
    if relative_translation_source[0] >= 0:
        raise ScaredA00PreprocessError(
            "rectified transform has nonnegative x translation; expected right_from_left"
        )
    transform_m = np.eye(4, dtype=np.float64)
    transform_m[:3, 3] = relative_translation_source * unit_to_m
    k_left_rectified = p1[:, :3]
    k_right_rectified = p2[:, :3]
    if not np.allclose(k_left_rectified[0, 0], k_right_rectified[0, 0], atol=1e-6):
        raise ScaredA00PreprocessError("rectified camera focal lengths do not match")
    if not np.allclose(k_left_rectified[1, 2], k_right_rectified[1, 2], atol=1e-6):
        raise ScaredA00PreprocessError("rectified camera scanlines do not match")

    scene_points_to_calibration_units = _unit_to_m(scene_points_unit) / unit_to_m
    left_points_in_calibration_units = np.asarray(left_points, dtype=np.float64) * (
        scene_points_to_calibration_units
    )
    disparity, gt_depth_source, empty_gt = _rectified_disparity_from_left_points(
        left_points_in_calibration_units,
        r1,
        p1,
        p2,
        (height, width),
        allow_empty_gt=allow_empty_gt,
    )
    valid = disparity > 0
    if not np.any(valid) and empty_gt is None:
        raise ScaredA00PreprocessError("no valid metric-GT disparity survived rectification")
    q_points = cv2.reprojectImageTo3D(disparity, q, handleMissingValues=False)
    q_depth_source = q_points[..., 2].astype(np.float64)
    baseline_m = -float(transform_m[0, 3])
    effective_disparity = disparity.astype(np.float64) - (
        float(k_left_rectified[0, 2]) - float(k_right_rectified[0, 2])
    )
    a00_depth_m = (
        float(k_left_rectified[0, 0]) * baseline_m / np.where(valid, effective_disparity, 1.0)
    )
    source_units_to_mm = unit_to_m * 1000.0
    q_a00_error_mm = (
        np.abs(q_depth_source[valid] - a00_depth_m[valid] / unit_to_m) * source_units_to_mm
    )
    metric_gt_error_mm = np.abs(q_depth_source[valid] - gt_depth_source[valid]) * (
        source_units_to_mm
    )
    q_a00_tolerance = float(validation_config.get("q_vs_a00_max_abs_error_mm", -1.0))
    metric_gt_tolerance = float(validation_config.get("metric_gt_max_abs_error_mm", -1.0))
    if not all(
        math.isfinite(value) and value > 0 for value in (q_a00_tolerance, metric_gt_tolerance)
    ):
        raise ScaredA00PreprocessError("metric validation tolerances must be finite and positive")
    if empty_gt is None and float(q_a00_error_mm.max()) > q_a00_tolerance:
        raise ScaredA00PreprocessError(
            "Q-derived and A00-derived depth differ beyond the configured tolerance"
        )
    if empty_gt is None and float(metric_gt_error_mm.max()) > metric_gt_tolerance:
        raise ScaredA00PreprocessError(
            "Q-derived depth differs from corresponding metric scene_points beyond tolerance"
        )
    summary = (
        {
            "valid_pixel_count": int(valid.sum()),
            "q_vs_a00_abs_error_mm": {
                "median": float(np.median(q_a00_error_mm)),
                "p90": float(np.percentile(q_a00_error_mm, 90)),
                "p99": float(np.percentile(q_a00_error_mm, 99)),
                "max": float(q_a00_error_mm.max()),
                "tolerance": q_a00_tolerance,
            },
            "q_vs_metric_gt_abs_error_mm": {
                "median": float(np.median(metric_gt_error_mm)),
                "p90": float(np.percentile(metric_gt_error_mm, 90)),
                "p99": float(np.percentile(metric_gt_error_mm, 99)),
                "max": float(metric_gt_error_mm.max()),
                "tolerance": metric_gt_tolerance,
            },
        }
        if empty_gt is None
        else {
            "status": "unavailable",
            "valid_pixel_count": 0,
            "empty_gt_diagnostics": empty_gt,
            "q_vs_a00_abs_error_mm": None,
            "q_vs_metric_gt_abs_error_mm": None,
        }
    )
    return {
        "left_rectified_bgr": rectified_left,
        "right_rectified_bgr": rectified_right,
        "K_L": k_left_rectified,
        "K_R": k_right_rectified,
        "T_R_from_L_m": transform_m,
        "R1": r1,
        "R2": r2,
        "P1": p1,
        "P2": p2,
        "Q": q,
        "disparity": disparity,
        "metric_gt_depth_source_units": gt_depth_source,
        "metric_validation": summary,
        "full_sensor_support": full_support,
        "opencv_version": cv2.__version__,
        "image_hw": [height, width],
        "valid_roi_left": [int(value) for value in roi_left],
        "valid_roi_right": [int(value) for value in roi_right],
        "translation_unit": translation_unit,
    }


def _read_archive_member(archive_path: Path, member_basename: str) -> tuple[bytes, str]:
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            matches = [
                member
                for member in archive.getmembers()
                if Path(member.name).name == member_basename and member.isfile()
            ]
            if len(matches) != 1:
                raise ScaredA00PreprocessError(
                    f"{archive_path} contains {len(matches)} members named {member_basename}; "
                    "expected exactly one"
                )
            extracted = archive.extractfile(matches[0])
            if extracted is None:
                raise ScaredA00PreprocessError(
                    f"unable to read {member_basename} from {archive_path}"
                )
            payload = extracted.read()
    except (OSError, tarfile.TarError) as error:
        raise ScaredA00PreprocessError(f"unable to read archive {archive_path}: {error}") from error
    return payload, _sha256_bytes(payload)


def _decode_video_frame(path: Path, frame_index: int) -> Any:
    import cv2

    if frame_index < 0:
        raise ScaredA00PreprocessError("video frame index must be non-negative")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ScaredA00PreprocessError(f"unable to open SCARED video {path}")
    frame = None
    try:
        for _ in range(frame_index + 1):
            success, decoded = capture.read()
            if not success:
                raise ScaredA00PreprocessError(
                    f"SCARED video ended before requested frame index {frame_index}"
                )
            frame = decoded
    finally:
        capture.release()
    if frame is None:
        raise ScaredA00PreprocessError("video decoding returned no frame")
    return frame


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _reject_symlink_components(path: Path) -> None:
    """Reject any existing symlink along an absolute path."""
    absolute = Path(os.path.abspath(path.expanduser()))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise ScaredA00PreprocessError(
                f"refusing output path with symlink component: {current}"
            )


def _assert_safe_output_destination(
    output_root: Path,
    final_keyframe: Path,
    *,
    raw_root: Path,
    repo_root: Path,
    require_existing_parent: bool,
) -> None:
    """Check output containment, including existing path components and target."""
    _reject_symlink_components(output_root)
    _reject_symlink_components(final_keyframe)
    resolved_output = output_root.resolve(strict=require_existing_parent)
    resolved_raw = raw_root.resolve(strict=True)
    resolved_repo = repo_root.resolve(strict=True)
    if resolved_output == resolved_raw or resolved_raw in resolved_output.parents:
        raise ScaredA00PreprocessError("processed output must not be written inside the raw SCARED root")
    if resolved_output == resolved_repo or resolved_repo in resolved_output.parents:
        raise ScaredA00PreprocessError("processed SCARED data must be written outside the repository")
    if resolved_output in resolved_raw.parents:
        raise ScaredA00PreprocessError("output_root must not contain or replace the raw SCARED root")
    if require_existing_parent:
        resolved_parent = final_keyframe.parent.resolve(strict=True)
        if resolved_parent != resolved_output and resolved_output not in resolved_parent.parents:
            raise ScaredA00PreprocessError(
                "processed keyframe parent resolves outside the selected output root"
            )
        if final_keyframe.is_symlink():
            raise ScaredA00PreprocessError("refusing symlink at the final processed keyframe path")
        if final_keyframe.exists():
            raise ScaredA00PreprocessError(
                f"refusing to overwrite existing processed keyframe {final_keyframe}"
            )


def _build_invocation_record(
    *,
    data_root: Path,
    output_root: Path,
    sample_id: str,
    config_path: Path,
    evidence_path: Path,
    invocation_argv: Sequence[str] | None,
) -> dict[str, object]:
    arguments = list(invocation_argv) if invocation_argv is not None else [
        "a00",
        "prepare",
        "--data-root",
        data_root.as_posix(),
        "--sample-id",
        sample_id,
        "--output-root",
        output_root.as_posix(),
        "--config",
        config_path.as_posix(),
        "--calibration-evidence",
        evidence_path.as_posix(),
    ]
    return {
        "executable": sys.executable,
        "module": "reliable_endo_gs.cli.main",
        "argv": arguments,
        "working_directory": Path.cwd().resolve().as_posix(),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "resolved_paths": {
            "data_root": data_root.as_posix(),
            "output_root": output_root.as_posix(),
            "config": config_path.as_posix(),
            "calibration_evidence": evidence_path.as_posix(),
        },
    }


def prepare_scared_a00_sample(
    *,
    data_root: Path,
    sample_id: str,
    output_root: Path,
    config_path: Path,
    calibration_evidence_path: Path,
    invocation_argv: Sequence[str] | None = None,
) -> dict[str, object]:
    """Prepare one verified original SCARED sample without touching the repo.

    The evidence manifest identifies the pinned numeric-value source separately
    from empirical direction and mm-scale resolution. Existing outputs are
    never overwritten.
    """
    import cv2
    import numpy as np
    import tifffile

    match = _SAMPLE_PATTERN.fullmatch(sample_id)
    if match is None:
        raise ScaredA00PreprocessError(
            "sample_id must be dataset_1..dataset_9/keyframe_1..keyframe_5/frame_dataNNNNNN"
        )
    dataset_id, keyframe_id, frame_id = match.groups()
    root = Path(data_root).expanduser().resolve(strict=True)
    output_candidate = Path(os.path.abspath(Path(output_root).expanduser()))
    _reject_symlink_components(output_candidate)
    output = output_candidate.resolve()
    config_file = Path(config_path).expanduser().resolve(strict=True)
    evidence_file = Path(calibration_evidence_path).expanduser().resolve(strict=True)
    repo_root = Path(__file__).resolve().parents[3]
    _assert_safe_output_destination(
        output,
        output / dataset_id / keyframe_id,
        raw_root=root,
        repo_root=repo_root,
        require_existing_parent=False,
    )
    config, config_sha256 = _read_config(config_file)
    try:
        evidence_bytes = evidence_file.read_bytes()
        evidence_raw = _read_json_bytes(evidence_bytes, "calibration evidence manifest")
    except OSError as error:
        raise ScaredA00PreprocessError(
            f"unable to read calibration evidence manifest {evidence_file}: {error}"
        ) from error
    calibration_config = _required_mapping(config.get("calibration"), "config.calibration")
    trusted_source_obj = calibration_config.get("trusted_calibration_value_source_sha256")
    trusted_value_source_sha256 = (
        trusted_source_obj if isinstance(trusted_source_obj, str) else None
    )
    evidence = _validate_calibration_evidence(
        evidence_raw,
        sample_id=sample_id,
        trusted_value_source_sha256=trusted_value_source_sha256,
    )
    calibration_evidence_sha256 = _sha256_bytes(evidence_bytes)
    calibration = _required_mapping(evidence["calibration"], "evidence.calibration")
    translation_unit = str(calibration["translation_unit"])
    value_source = _required_mapping(
        calibration["calibration_value_source"], "evidence.calibration.calibration_value_source"
    )
    direction_resolution = _required_mapping(
        evidence["direction_resolution"], "evidence.direction_resolution"
    )
    transform_direction = str(direction_resolution["selected_direction"])
    metric_gt_evidence = _required_mapping(evidence["metric_ground_truth"], "metric_ground_truth")
    scene_points_unit = str(metric_gt_evidence["xyz_units"])

    input_config = _required_mapping(config.get("input"), "config.input")
    keyframe_root = root / dataset_id / keyframe_id
    source_file_value = calibration_config.get("calibration_value_source_file")
    if source_file_value != "endoscope_calibration.yaml":
        raise ScaredA00PreprocessError(
            "config must identify the sibling endoscope_calibration.yaml numeric value source"
        )
    calibration_value_path = keyframe_root / str(source_file_value)
    video_path = keyframe_root / str(input_config.get("video_file"))
    frame_archive = keyframe_root / str(input_config.get("frame_data_archive"))
    scene_archive = keyframe_root / str(input_config.get("scene_points_archive"))
    for source in (video_path, frame_archive, scene_archive, calibration_value_path):
        if not source.is_file():
            raise ScaredA00PreprocessError(f"required original SCARED source is missing: {source}")
    frame_bytes, frame_member_sha256 = _read_archive_member(
        frame_archive, f"{frame_id}.json"
    )
    scene_id = frame_id.replace("frame_data", "scene_points")
    scene_bytes, scene_member_sha256 = _read_archive_member(
        scene_archive, f"{scene_id}.tiff"
    )
    member_hashes = _required_mapping(evidence["source_member_sha256"], "source_member_sha256")
    if member_hashes.get("frame_data") != frame_member_sha256:
        raise ScaredA00PreprocessError(
            "calibration evidence frame_data hash does not match selected archive member"
        )
    if member_hashes.get("scene_points") != scene_member_sha256:
        raise ScaredA00PreprocessError(
            "calibration evidence scene_points hash does not match selected archive member"
        )
    frame_data = _read_json_bytes(frame_bytes, f"{frame_id}.json")
    calibration_obj = _required_mapping(
        frame_data.get(str(input_config.get("calibration_field"))),
        "frame_data.camera-calibration",
    )
    if not isinstance(trusted_value_source_sha256, str):
        raise ScaredA00PreprocessError("calibration value-source SHA-256 is not pinned")
    _verify_calibration_yaml_matches_frame_data(
        calibration_value_path,
        calibration_obj,
        expected_sha256=trusted_value_source_sha256,
    )
    try:
        scene_points_stacked = tifffile.imread(io.BytesIO(scene_bytes))
    except (OSError, ValueError) as error:
        raise ScaredA00PreprocessError(f"unable to decode selected scene_points TIFF: {error}") from error

    measured_cross_view = _cross_view_nearest_neighbor_metrics(
        scene_points_stacked,
        calibration_obj.get("R"),
        calibration_obj.get("T"),
        declared_translation_unit=translation_unit,
        correspondence_policy=(
            "mutual"
            if _required_mapping(evidence["independent_validation"], "independent_validation").get(
                "valid_point_policy"
            ) == "finite_nonzero_xyz_positive_z_mutual_nearest_neighbor"
            else "all_valid"
        ),
    )
    validation_config = _required_mapping(config.get("validation"), "config.validation")
    acceptance = _required_mapping(
        validation_config.get("cross_view_acceptance"), "validation.cross_view_acceptance"
    )
    if validation_config.get("scene_points_xyz_unit") != scene_points_unit:
        raise ScaredA00PreprocessError(
            "metric scene_points XYZ unit differs from the pinned preprocessing config"
        )
    if input_config.get("stacked_scene_points_layout") != direction_resolution.get(
        "scene_points_layout"
    ):
        raise ScaredA00PreprocessError(
            "direction evidence scene_points layout differs from the configured stacked-view layout"
        )
    _verify_cross_view_evidence(
        measured_cross_view,
        _required_mapping(evidence["independent_validation"], "independent_validation"),
        declared_unit=translation_unit,
        declared_direction=transform_direction,
        tolerance_mm=float(validation_config.get("cross_view_evidence_tolerance_mm", -1.0)),
        acceptance=acceptance,
    )

    frame_index = int(frame_id.removeprefix("frame_data"))
    stacked_video = _decode_video_frame(video_path, frame_index)
    height = int(input_config.get("frame_height", 0))
    width = int(input_config.get("frame_width", 0))
    if stacked_video.shape != (height * 2, width, 3):
        raise ScaredA00PreprocessError(
            f"stacked RGB video frame has shape {stacked_video.shape}, expected {(height * 2, width, 3)}"
        )
    left_raw = stacked_video[:height]
    right_raw = stacked_video[height:]
    arrays = rectify_scared_sample_arrays(
        left_bgr=left_raw,
        right_bgr=right_raw,
        stacked_scene_points=scene_points_stacked,
        calibration=calibration_obj,
        translation_unit=translation_unit,
        scene_points_unit=scene_points_unit,
        config=config,
    )

    source_code_path = Path(__file__).resolve()
    code_sha256 = _sha256_file(source_code_path)
    invocation = _build_invocation_record(
        data_root=root,
        output_root=output,
        sample_id=sample_id,
        config_path=config_file,
        evidence_path=evidence_file,
        invocation_argv=invocation_argv,
    )
    role = "final_test" if dataset_id in ("dataset_8", "dataset_9") else "final_train"
    newpram = {
        "intr0": arrays["K_L"].tolist(),
        "intr1": arrays["K_R"].tolist(),
        "T_R_from_L": arrays["T_R_from_L_m"].tolist(),
        "rectification": {
            "R1": arrays["R1"].tolist(),
            "R2": arrays["R2"].tolist(),
            "P1": arrays["P1"].tolist(),
            "P2": arrays["P2"].tolist(),
        },
    }
    preprocessing_metadata: dict[str, object] = {
        "images_grid": "processed",
        "processor": "reliable_endo_gs.data.scared_a00_preprocess",
        "processor_version": config.get("preprocessing_id"),
        "code_sha256": code_sha256,
        "config_path": config_file.as_posix(),
        "config_sha256": config_sha256,
        "calibration_evidence_sha256": calibration_evidence_sha256,
        "calibration_value_source_path": value_source["source_path"],
        "calibration_value_source_sha256": _required_mapping(
            _required_mapping(evidence["calibration"], "evidence.calibration").get(
                "calibration_value_source"
            ),
            "calibration_value_source",
        )["source_sha256"],
        "source_translation_unit": translation_unit,
        "source_translation_field": _CALIBRATION_FIELD,
        "scene_points_xyz_unit": scene_points_unit,
        "direction_resolution": {
            **direction_resolution,
            "measured_point_counts": measured_cross_view["point_counts"],
            "measured_metrics_mm": measured_cross_view["metrics_mm"],
        },
        "opencv_version": arrays["opencv_version"],
        "stereo_rectify_flags": "CALIB_ZERO_DISPARITY",
        "stereo_rectify_alpha": 0.0,
        "image_interpolation": "INTER_LINEAR",
        "map_type": "CV_32FC1",
        "pixel_center": "integer",
        "crop": None,
        "resize": None,
        "input_hw": arrays["image_hw"],
        "output_hw": arrays["image_hw"],
        "full_sensor_support_verified": arrays["full_sensor_support"],
        "opencv_valid_roi_left": arrays["valid_roi_left"],
        "opencv_valid_roi_right": arrays["valid_roi_right"],
        "metric_validation": arrays["metric_validation"],
        "invocation": invocation,
    }
    protocol: dict[str, object] = {
        "schema_version": 1,
        "image_range": "uint8_0_255",
        "rectified": True,
        "distortion_removed": True,
        "pixel_center": "integer",
        "disparity": {
            "units": "pixels",
            "sign": "u_left_minus_right",
            "reference": "left",
            "source": "scene_points_left_xyz_projected_through_rectified_P1_P2",
            "invalid_value": 0.0,
        },
        "extrinsics": {
            "source": "newpram.T_R_from_L",
            "direction": transform_direction,
            "units": "m",
            "raw_source_field": _CALIBRATION_FIELD,
            "raw_units": translation_unit,
            "unit_evidence_sha256": calibration_evidence_sha256,
            "calibration_value_source_sha256": preprocessing_metadata[
                "calibration_value_source_sha256"
            ],
        },
        "preprocessing": preprocessing_metadata,
        "metric_geometry_evidence_sha256": calibration_evidence_sha256,
    }

    final_keyframe = output / dataset_id / keyframe_id
    _assert_safe_output_destination(
        output,
        final_keyframe,
        raw_root=root,
        repo_root=repo_root,
        require_existing_parent=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="a00-sample-", dir=output) as temporary:
        staged_keyframe = Path(temporary) / dataset_id / keyframe_id
        data_dir = staged_keyframe / "data"
        left_path = data_dir / "left_finalpass" / f"{frame_id}.png"
        right_path = data_dir / "right_finalpass" / f"{frame_id}.png"
        frame_path = data_dir / "frame_data" / f"{frame_id}.json"
        newpram_path = data_dir / "newpram_data" / f"{frame_id}.json"
        disparity_path = data_dir / "disparity" / f"{frame_id}.tiff"
        q_path = data_dir / "reprojection_data" / f"{frame_id}.json"
        protocol_path = data_dir / "a00_protocol.json"
        for path in (left_path, right_path, frame_path, newpram_path, disparity_path, q_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(left_path), arrays["left_rectified_bgr"]):
            raise ScaredA00PreprocessError(f"failed to write rectified left RGB {left_path}")
        if not cv2.imwrite(str(right_path), arrays["right_rectified_bgr"]):
            raise ScaredA00PreprocessError(f"failed to write rectified right RGB {right_path}")
        frame_path.write_bytes(frame_bytes)
        _write_json(newpram_path, newpram)
        tifffile.imwrite(disparity_path, arrays["disparity"], dtype=np.float32)
        _write_json(q_path, {"reprojection-matrix": arrays["Q"].tolist()})
        _write_json(protocol_path, protocol)

        output_files = (
            left_path,
            right_path,
            frame_path,
            newpram_path,
            disparity_path,
            q_path,
            protocol_path,
        )
        output_hashes = {
            path.relative_to(data_dir).as_posix(): _sha256_file(path) for path in output_files
        }
        provenance = {
            "schema_version": 1,
            "dataset_name": "scared",
            "sample_id": sample_id,
            "data_role": role,
            "frame_correspondence": {
                "frame_data_id": frame_id,
                "scene_points_id": scene_id,
                "matched": True,
                "ambiguous": False,
                "frame_data_member_sha256": frame_member_sha256,
                "scene_points_member_sha256": scene_member_sha256,
            },
            "source_files": {
                "rgb_video": {"relative_path": str(input_config.get("video_file")), "sha256": _sha256_file(video_path)},
                "frame_data_archive": {"relative_path": str(input_config.get("frame_data_archive")), "member": f"{frame_id}.json", "member_sha256": frame_member_sha256},
                "scene_points_archive": {"relative_path": str(input_config.get("scene_points_archive")), "member": f"{scene_id}.tiff", "member_sha256": scene_member_sha256},
            },
            "preprocessing": preprocessing_metadata,
            "calibration_evidence_sha256": calibration_evidence_sha256,
            "invocation": invocation,
            "metric_validation": arrays["metric_validation"],
            "output_files_sha256": output_hashes,
        }
        _write_json(data_dir / "a00_provenance.json", provenance)
        (output / dataset_id).mkdir(parents=True, exist_ok=True)
        _assert_safe_output_destination(
            output,
            final_keyframe,
            raw_root=root,
            repo_root=repo_root,
            require_existing_parent=True,
        )
        os.replace(staged_keyframe, final_keyframe)
    return {
        "sample_id": sample_id,
        "data_role": role,
        "processed_root": output.as_posix(),
        "sample_path": (final_keyframe / "data").as_posix(),
        "calibration_evidence_sha256": calibration_evidence_sha256,
        "code_sha256": code_sha256,
        "config_sha256": config_sha256,
        "frame_member_sha256": frame_member_sha256,
        "scene_points_member_sha256": scene_member_sha256,
        "metric_validation": arrays["metric_validation"],
        "invocation": invocation,
    }


def a00_smoke_command(
    data_root: Path, sample_id: str, roles_path: Path, config_path: Path
) -> str:
    """Return a shell-escaped exact one-sample deployment-only smoke command."""
    import shlex

    repo_root = Path(__file__).resolve().parents[3]
    values = (
        "env",
        f"PYTHONPATH={(repo_root / 'src').as_posix()}",
        sys.executable,
        "-m",
        "reliable_endo_gs.cli.main",
        "a00",
        "smoke",
        "--data-root",
        str(Path(data_root).resolve()),
        "--sample-id",
        sample_id,
        "--roles",
        str(Path(roles_path).resolve()),
        "--config",
        str(Path(config_path).resolve()),
    )
    return shlex.join(values)


__all__ = [
    "ScaredA00PreprocessError",
    "a00_smoke_command",
    "prepare_scared_a00_sample",
    "rectify_scared_sample_arrays",
]
