"""Synthetic tests for original-SCARED A00 preprocessing and preflight."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml
from PIL import Image

from reliable_endo_gs.cli.main import build_parser, run_a00_preflight
from reliable_endo_gs.data.scared_a00 import ScaredA00ProtocolError, ScaredA00SampleError
from reliable_endo_gs.data.scared_a00_identity import (
    CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR,
    POSITIVE_GT_PROCESSOR_PAIR,
)
from reliable_endo_gs.data.scared_a00_preprocess import (
    ScaredA00PreprocessError,
    _assert_safe_output_destination,
    _build_invocation_record,
    _cross_view_nearest_neighbor_metrics,
    _rectified_disparity_from_left_points,
    _validate_calibration_evidence,
    _verify_calibration_yaml_matches_frame_data,
    _verify_cross_view_evidence,
    rectify_scared_sample_arrays,
)

DIRECTION_SOURCE_SHA256 = "597878f105e38ff2c2b7973d2c3a6af31293b904ca3b73bbe0a5f239466e6e9e"


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "preprocessing_id": "reliable_endo_gs_a00_scared_sample_v1",
        "input": {
            "frame_height": 16,
            "frame_width": 20,
            "calibration_field": "camera-calibration",
            "stacked_video_layout": "top_left_bottom_right",
            "stacked_scene_points_layout": "top_left_bottom_right_xyz",
        },
        "rectification": {
            "flags": "CALIB_ZERO_DISPARITY",
            "alpha": 0.0,
            "map_type": "CV_32FC1",
            "image_interpolation": "INTER_LINEAR",
            "border_mode": "BORDER_CONSTANT",
            "border_value": 0,
            "crop": "none",
            "resize": "none",
        },
        "validation": {
            "scene_points_xyz_unit": "mm",
            "rectified_rotation_atol": 1e-5,
            "rectified_translation_yz_atol_mm": 1e-3,
            "full_image_support_required": True,
            "q_vs_a00_max_abs_error_mm": 0.01,
            "metric_gt_max_abs_error_mm": 0.05,
            "cross_view_acceptance": {
                "max_declared_median_mm": 0.2,
                "max_declared_p90_mm": 0.5,
                "max_declared_p99_mm": 1.0,
                "min_control_to_declared_p90_ratio": 2.0,
                "min_control_p90_gap_mm": 0.05,
            },
        },
        "calibration": {
            "calibration_value_source_file": "endoscope_calibration.yaml",
            "trusted_calibration_value_source_sha256": DIRECTION_SOURCE_SHA256,
        },
    }


def _evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "sample_id": "dataset_1/keyframe_1/frame_data000000",
        "calibration": {
            "source_field": "frame_data.camera-calibration.R/T",
            "translation_unit": "mm",
            "transform_direction": "right_from_left",
            "calibration_value_source": {
                "field_path": "R/T",
                "source_title": "Original SCARED calibration value YAML",
                "source_uri": "SCARED:dataset_1/keyframe_1/endoscope_calibration.yaml",
                "source_path": "dataset_1/keyframe_1/endoscope_calibration.yaml",
                "source_sha256": DIRECTION_SOURCE_SHA256,
                "frame_data_match": {
                    "source_field": "frame_data.camera-calibration.R/T",
                    "frame_data_member_sha256": "b" * 64,
                    "R_matches": True,
                    "T_matches": True,
                },
            },
        },
        "direction_resolution": {
            "method": "cross_view_nearest_neighbor",
            "resolution_basis": "empirical_paired_cloud_residuals",
            "selected_direction": "right_from_left",
            "candidate_transform": "R @ left_xyz + T -> right_xyz",
            "direction_control": "R @ right_xyz + T -> left_xyz",
            "scene_points_layout": "top_left_bottom_right_xyz",
            "metric_ground_truth_source_uri": "https://endovissub2019-scared.grand-challenge.org/About/",
        },
        "unit_resolution": {
            "method": "cross_view_nearest_neighbor",
            "resolution_basis": "empirical_cross_view_against_metric_ground_truth",
            "selected_translation_unit": "mm",
            "metric_ground_truth_source_uri": "https://endovissub2019-scared.grand-challenge.org/About/",
        },
        "source_member_sha256": {"frame_data": "b" * 64, "scene_points": "c" * 64},
        "frame_correspondence": {
            "frame_data_id": "frame_data000000",
            "scene_points_id": "scene_points000000",
            "matched": True,
            "ambiguous": False,
        },
        "independent_validation": {
            "method": "cross_view_nearest_neighbor",
            "valid_point_policy": "finite_nonzero_xyz_positive_z",
            "point_counts": {"left": 4, "right": 4},
            "metrics_mm": {
                "translation_as_mm": {"median": 0.0, "p90": 0.0, "p99": 0.0},
                "translation_as_cm": {"median": 1.0, "p90": 1.0, "p99": 1.0},
                "translation_as_m": {"median": 3.0, "p90": 3.0, "p99": 3.0},
                "wrong_direction_reuse_declared_transform": {
                    "median": 2.0,
                    "p90": 2.0,
                    "p99": 2.0,
                },
            },
        },
        "metric_ground_truth": {
            "xyz_units": "mm",
            "depth_units": "mm",
            "source_uri": "https://endovissub2019-scared.grand-challenge.org/About/",
        },
    }


def test_a00_config_pins_only_the_numeric_calibration_value_source() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs/data/a00_scared_sample_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    calibration = config["calibration"]
    assert calibration["calibration_value_source_file"] == "endoscope_calibration.yaml"
    assert calibration["trusted_calibration_value_source_sha256"] == DIRECTION_SOURCE_SHA256
    assert "source_translation_unit_and_direction" not in calibration


def _processed_fixture(root: Path, config_path: Path) -> tuple[Path, str]:
    sample_id = "dataset_1/keyframe_1/frame_data000000"
    data_dir = root / "dataset_1" / "keyframe_1" / "data"
    frame_id = "frame_data000000"
    for name in ("left_finalpass", "right_finalpass", "frame_data", "newpram_data"):
        (data_dir / name).mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 5, 3), 80, dtype=np.uint8)).save(
        data_dir / "left_finalpass" / f"{frame_id}.png"
    )
    Image.fromarray(np.full((4, 5, 3), 120, dtype=np.uint8)).save(
        data_dir / "right_finalpass" / f"{frame_id}.png"
    )
    (data_dir / "frame_data" / f"{frame_id}.json").write_text("{}", encoding="utf-8")
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = -0.1
    (data_dir / "newpram_data" / f"{frame_id}.json").write_text(
        json.dumps(
            {
                "intr0": [[100.0, 0.0, 2.0], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]],
                "intr1": [[100.0, 0.0, 2.0], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]],
                "T_R_from_L": transform.tolist(),
            }
        ),
        encoding="utf-8",
    )
    config_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    code_path = Path(__file__).resolve().parents[2] / "src" / "reliable_endo_gs" / "data" / "scared_a00_preprocess.py"
    code_hash = hashlib.sha256(code_path.read_bytes()).hexdigest()
    evidence_path = root.parent / "evidence.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(_evidence()), encoding="utf-8")
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    preprocessing = {
        "images_grid": "processed",
        "code_sha256": code_hash,
        "config_sha256": config_hash,
        "calibration_evidence_sha256": evidence_hash,
        "calibration_value_source_sha256": DIRECTION_SOURCE_SHA256,
        "source_translation_field": "frame_data.camera-calibration.R/T",
        "source_translation_unit": "mm",
        "full_sensor_support_verified": True,
    }
    resolved_root = root.resolve()
    resolved_config = config_path.resolve()
    invocation = {
        "executable": sys.executable,
        "module": "reliable_endo_gs.cli.main",
        "argv": ["a00", "prepare", "--sample-id", sample_id],
        "working_directory": Path.cwd().resolve().as_posix(),
        "pythonpath": "src",
        "resolved_paths": {
            "data_root": (root.parent / "raw").resolve().as_posix(),
            "output_root": resolved_root.as_posix(),
            "config": resolved_config.as_posix(),
            "calibration_evidence": evidence_path.resolve().as_posix(),
        },
    }
    preprocessing["invocation"] = invocation
    protocol = {
        "schema_version": 1,
        "image_range": "uint8_0_255",
        "rectified": True,
        "distortion_removed": True,
        "pixel_center": "integer",
        "disparity": {"units": "pixels", "sign": "u_left_minus_right", "reference": "left"},
        "extrinsics": {
            "source": "newpram.T_R_from_L",
            "direction": "right_from_left",
            "units": "m",
            "raw_source_field": "frame_data.camera-calibration.R/T",
            "raw_units": "mm",
            "unit_evidence_sha256": evidence_hash,
            "calibration_value_source_sha256": DIRECTION_SOURCE_SHA256,
        },
        "preprocessing": preprocessing,
        "metric_geometry_evidence_sha256": evidence_hash,
    }
    protocol_path = data_dir / "a00_protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    disparity_dir = data_dir / "disparity"
    disparity_dir.mkdir(parents=True)
    disparity_path = disparity_dir / f"{frame_id}.tiff"
    tifffile.imwrite(disparity_path, np.ones((4, 5), dtype=np.float32))
    q_dir = data_dir / "reprojection_data"
    q_dir.mkdir(parents=True)
    q_path = q_dir / f"{frame_id}.json"
    q_path.write_text(json.dumps({"reprojection-matrix": np.eye(4).tolist()}), encoding="utf-8")
    outputs = {}
    for path in (
        data_dir / "left_finalpass" / f"{frame_id}.png",
        data_dir / "right_finalpass" / f"{frame_id}.png",
        data_dir / "frame_data" / f"{frame_id}.json",
        data_dir / "newpram_data" / f"{frame_id}.json",
        disparity_path,
        q_path,
        protocol_path,
    ):
        outputs[path.relative_to(data_dir).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "schema_version": 1,
        "sample_id": sample_id,
        "data_role": "final_train",
        "calibration_evidence_sha256": evidence_hash,
        "preprocessing": preprocessing,
        "invocation": invocation,
        "frame_correspondence": {
            "frame_data_id": frame_id,
            "scene_points_id": "scene_points000000",
            "matched": True,
            "ambiguous": False,
            "frame_data_member_sha256": "b" * 64,
            "scene_points_member_sha256": "c" * 64,
        },
        "source_files": {
            "frame_data_archive": {
                "member": f"{frame_id}.json",
                "member_sha256": "b" * 64,
            },
            "scene_points_archive": {
                "member": "scene_points000000.tiff",
                "member_sha256": "c" * 64,
            },
        },
        "output_files_sha256": outputs,
    }
    (data_dir / "a00_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return root, sample_id


@pytest.mark.parametrize("legacy", [False, True])
def test_batch_preflight_reports_exact_or_reviewed_producer_identity(
    tmp_path: Path, legacy: bool
) -> None:
    config_path = tmp_path / "processed/_provenance/configs/a00.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(_config(), sort_keys=True))
    root, sample_id = _processed_fixture(tmp_path / "processed", config_path)
    data = root / "dataset_1/keyframe_1/data"
    path = data / "a00_provenance.json"
    provenance = json.loads(path.read_text())
    pair = POSITIVE_GT_PROCESSOR_PAIR if legacy else CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR
    provenance["metric_validation"] = {"valid_pixel_count": 20}
    preprocessing = provenance["preprocessing"]
    preprocessing.update(
        code_sha256=pair[0], batch_code_sha256=pair[1], config_path=str(config_path)
    )
    protocol_path = data / "a00_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["preprocessing"] = preprocessing
    protocol_path.write_text(json.dumps(protocol))
    provenance["output_files_sha256"]["a00_protocol.json"] = hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(provenance))
    result = run_a00_preflight(
        root,
        sample_id,
        roles_path=Path("configs/teachers/a01_scared.yaml"),
        config_path=config_path,
        print_smoke_command=False,
    )
    assert result["processor_compatibility"] == (
        "reviewed_positive_gt_compatibility" if legacy else "exact_code_identity"
    )
    assert result["producer_code_sha256"] == pair[0]
    assert result["ground_truth_loaded"] is False
    assert result["metric_gt_status"] == "available"
    # Provenance cannot be relabelled as an unknown processor or a legacy empty target.
    if legacy:
        provenance["metric_validation"] = {"valid_pixel_count": 0, "status": "unavailable"}
    else:
        preprocessing["batch_code_sha256"] = "e" * 64
        protocol["preprocessing"] = preprocessing
        protocol_path.write_text(json.dumps(protocol))
        provenance["output_files_sha256"]["a00_protocol.json"] = hashlib.sha256(
            protocol_path.read_bytes()
        ).hexdigest()
    path.write_text(json.dumps(provenance))
    with pytest.raises(ScaredA00ProtocolError, match="unreviewed code identity"):
        run_a00_preflight(
            root,
            sample_id,
            roles_path=Path("configs/teachers/a01_scared.yaml"),
            config_path=config_path,
            print_smoke_command=False,
        )


def test_rectification_builds_metric_a00_geometry_from_synthetic_plane() -> None:
    height, width = 16, 20
    focal = 80.0
    depth_mm = 200.0
    baseline_mm = 10.0
    y, x = np.mgrid[:height, :width]
    scene_left = np.stack(
        (
            (x - (width - 1) / 2) * depth_mm / focal,
            (y - (height - 1) / 2) * depth_mm / focal,
            np.full((height, width), depth_mm),
        ),
        axis=-1,
    ).astype(np.float32)
    scene_right = scene_left.copy()
    scene_right[..., 0] -= baseline_mm
    stacked_scene = np.concatenate((scene_left, scene_right), axis=0)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    calibration = {
        "KL": [[focal, 0.0, (width - 1) / 2], [0.0, focal, (height - 1) / 2], [0.0, 0.0, 1.0]],
        "KR": [[focal, 0.0, (width - 1) / 2], [0.0, focal, (height - 1) / 2], [0.0, 0.0, 1.0]],
        "DL": [0.0] * 5,
        "DR": [0.0] * 5,
        "R": np.eye(3).tolist(),
        "T": [-baseline_mm, 0.0, 0.0],
    }

    result = rectify_scared_sample_arrays(
        left_bgr=image,
        right_bgr=image,
        stacked_scene_points=stacked_scene,
        calibration=calibration,
        translation_unit="mm",
        config=_config(),
    )

    assert result["full_sensor_support"] is True
    assert result["T_R_from_L_m"][0, 3] == pytest.approx(-0.01, abs=1e-8)
    assert result["metric_validation"]["valid_pixel_count"] > 0
    assert result["metric_validation"]["q_vs_a00_abs_error_mm"]["max"] <= 0.01
    assert result["metric_validation"]["q_vs_metric_gt_abs_error_mm"]["max"] <= 0.05

    calibration_cm = {**calibration, "T": [-baseline_mm / 10.0, 0.0, 0.0]}
    result_cm = rectify_scared_sample_arrays(
        left_bgr=image,
        right_bgr=image,
        stacked_scene_points=stacked_scene,
        calibration=calibration_cm,
        translation_unit="cm",
        scene_points_unit="mm",
        config=_config(),
    )
    assert result_cm["T_R_from_L_m"][0, 3] == pytest.approx(-0.01, abs=1e-8)
    assert result_cm["metric_validation"]["q_vs_metric_gt_abs_error_mm"]["max"] <= 0.05

@pytest.mark.parametrize("empty_kind", ["outside_grid", "missing_xyz", "rounds_outside_grid"])
def test_batch_empty_gt_is_explicit_and_single_sample_remains_strict(empty_kind: str) -> None:
    height, width = 16, 20
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    points = np.zeros((height * 2, width, 3), dtype=np.float32)
    if empty_kind == "outside_grid":
        points[0, 0] = [1000.0, 0.0, 200.0]
    elif empty_kind == "rounds_outside_grid":
        # Left projection u=19.75, inside the continuous grid but rounded to 20.
        points[0, 0] = [(19.75 - 9.5) * 200.0 / 80.0, 0.0, 200.0]
    calibration = {
        "KL": [[80.0, 0.0, 9.5], [0.0, 80.0, 7.5], [0.0, 0.0, 1.0]],
        "KR": [[80.0, 0.0, 9.5], [0.0, 80.0, 7.5], [0.0, 0.0, 1.0]],
        "DL": [0.0] * 5,
        "DR": [0.0] * 5,
        "R": np.eye(3).tolist(),
        "T": [-10.0, 0.0, 0.0],
    }
    kwargs = dict(
        left_bgr=image,
        right_bgr=image,
        stacked_scene_points=points,
        calibration=calibration,
        translation_unit="mm",
        config=_config(),
    )
    with pytest.raises(ScaredA00PreprocessError):
        rectify_scared_sample_arrays(**kwargs)
    result = rectify_scared_sample_arrays(**kwargs, allow_empty_gt=True)
    assert not result["disparity"].any()
    assert not result["metric_gt_depth_source_units"].any()
    assert np.array_equal(result["left_rectified_bgr"], image)
    assert result["T_R_from_L_m"][0, 3] == pytest.approx(-0.01)
    metrics = result["metric_validation"]
    assert metrics["valid_pixel_count"] == 0
    assert metrics["status"] == "unavailable"
    assert metrics["q_vs_a00_abs_error_mm"] is None
    assert metrics["q_vs_metric_gt_abs_error_mm"] is None
    assert (
        metrics["empty_gt_diagnostics"]["reason"]
        == {
            "outside_grid": "no_rectified_stereo_overlap",
            "missing_xyz": "no_valid_left_xyz",
            "rounds_outside_grid": "no_integer_pixel_support",
        }[empty_kind]
    )


@pytest.mark.parametrize("bad_geometry", ["disparity_sign", "scanline", "camera_z"])
def test_empty_gt_policy_does_not_hide_geometry_errors(bad_geometry: str) -> None:
    points = np.zeros((16, 20, 3), dtype=np.float32)
    points[0, 0] = [1000.0, 0.0, 200.0]  # Outside both images.
    projection = np.array([[80.0, 0.0, 9.5, 0.0], [0.0, 80.0, 7.5, 0.0], [0.0, 0.0, 1.0, 0.0]])
    right = projection.copy()
    right[0, 3] = -800.0
    if bad_geometry == "disparity_sign":
        right[0, 3] *= -1
    elif bad_geometry == "scanline":
        right[1, 3] = 200.0
    else:
        right[2, 3] = -400.0
    with pytest.raises(ScaredA00PreprocessError, match="no positive rectified disparity"):
        _rectified_disparity_from_left_points(
            points, np.eye(3), projection, right, (16, 20), allow_empty_gt=True
        )


def test_empty_gt_mode_still_checks_configured_metric_tolerances() -> None:
    config = _config()
    config["validation"]["q_vs_a00_max_abs_error_mm"] = float("nan")
    image = np.zeros((16, 20, 3), dtype=np.uint8)
    calibration = {
        "KL": [[80.0, 0.0, 9.5], [0.0, 80.0, 7.5], [0.0, 0.0, 1.0]],
        "KR": [[80.0, 0.0, 9.5], [0.0, 80.0, 7.5], [0.0, 0.0, 1.0]],
        "DL": [0.0] * 5,
        "DR": [0.0] * 5,
        "R": np.eye(3).tolist(),
        "T": [-10.0, 0.0, 0.0],
    }
    with pytest.raises(ScaredA00PreprocessError, match="tolerances must be finite and positive"):
        rectify_scared_sample_arrays(
            left_bgr=image,
            right_bgr=image,
            stacked_scene_points=np.zeros((32, 20, 3), dtype=np.float32),
            calibration=calibration,
            translation_unit="mm",
            config=config,
            allow_empty_gt=True,
        )


def test_calibration_evidence_fails_when_no_value_source_is_pinned() -> None:
    with pytest.raises(ScaredA00PreprocessError, match="no calibration value-source SHA-256"):
        _validate_calibration_evidence(
            _evidence(),
            sample_id="dataset_1/keyframe_1/frame_data000000",
            trusted_value_source_sha256=None,
        )


def test_calibration_value_source_hash_must_match_pinned_source() -> None:
    evidence = _evidence()
    evidence["calibration"]["calibration_value_source"]["source_sha256"] = "d" * 64
    with pytest.raises(ScaredA00PreprocessError, match="does not match the pinned YAML"):
        _validate_calibration_evidence(
            evidence,
            sample_id="dataset_1/keyframe_1/frame_data000000",
            trusted_value_source_sha256=DIRECTION_SOURCE_SHA256,
        )


def test_direction_resolution_rejects_unvalidated_opposite_candidate() -> None:
    evidence = _evidence()
    evidence["direction_resolution"]["selected_direction"] = "left_from_right"
    evidence["calibration"]["transform_direction"] = "left_from_right"
    with pytest.raises(ScaredA00PreprocessError, match="empirically resolved as right_from_left"):
        _validate_calibration_evidence(
            evidence,
            sample_id="dataset_1/keyframe_1/frame_data000000",
            trusted_value_source_sha256=DIRECTION_SOURCE_SHA256,
        )


def test_direction_resolution_binds_the_scene_points_layout() -> None:
    evidence = _evidence()
    evidence["direction_resolution"]["scene_points_layout"] = "bottom_left_top_right_xyz"
    with pytest.raises(ScaredA00PreprocessError, match="paired-cloud residuals and controls"):
        _validate_calibration_evidence(
            evidence,
            sample_id="dataset_1/keyframe_1/frame_data000000",
            trusted_value_source_sha256=DIRECTION_SOURCE_SHA256,
        )


def test_calibration_value_source_carries_no_unit_or_direction_claims() -> None:
    for field, value in (
        ("translation_unit", "mm"),
        ("transform_direction", "right_from_left"),
        ("mapping_convention", "camera1_to_camera2"),
    ):
        evidence = _evidence()
        evidence["calibration"]["calibration_value_source"][field] = value
        with pytest.raises(
            ScaredA00PreprocessError,
            match="must not assert unit or transform-direction semantics",
        ):
            _validate_calibration_evidence(
                evidence,
                sample_id="dataset_1/keyframe_1/frame_data000000",
                trusted_value_source_sha256=DIRECTION_SOURCE_SHA256,
            )


def test_calibration_yaml_hash_and_rt_must_match_frame_data(tmp_path: Path) -> None:
    import cv2

    calibration_yaml = tmp_path / "endoscope_calibration.yaml"
    rotation = np.eye(3, dtype=np.float64)
    translation = np.array([[-4.14], [0.0], [0.0]], dtype=np.float64)
    storage = cv2.FileStorage(str(calibration_yaml), cv2.FILE_STORAGE_WRITE)
    storage.write("R", rotation)
    storage.write("T", translation)
    storage.release()
    source_hash = hashlib.sha256(calibration_yaml.read_bytes()).hexdigest()

    _verify_calibration_yaml_matches_frame_data(
        calibration_yaml,
        {"R": rotation.tolist(), "T": translation.reshape(3).tolist()},
        expected_sha256=source_hash,
    )
    with pytest.raises(ScaredA00PreprocessError, match="do not exactly match"):
        _verify_calibration_yaml_matches_frame_data(
            calibration_yaml,
            {"R": rotation.tolist(), "T": [-4.15, 0.0, 0.0]},
            expected_sha256=source_hash,
        )


def test_cross_view_gate_checks_absolute_units_and_controls() -> None:
    left = np.array(
        [[[0.0, 0.0, 10.0], [10.0, 0.0, 10.0]], [[0.0, 10.0, 20.0], [10.0, 10.0, 20.0]]]
    )
    right = left.copy()
    right[..., 0] -= 1.0
    stacked = np.concatenate((left, right), axis=0)
    measured = _cross_view_nearest_neighbor_metrics(
        stacked,
        np.eye(3),
        [-1.0, 0.0, 0.0],
        declared_translation_unit="mm",
    )
    acceptance = _config()["validation"]["cross_view_acceptance"]
    _verify_cross_view_evidence(
        measured,
        measured,
        declared_unit="mm",
        declared_direction="right_from_left",
        tolerance_mm=1e-9,
        acceptance=acceptance,
    )
    assert measured["metrics_mm"]["translation_as_mm"]["p90"] == pytest.approx(0.0)
    assert measured["metrics_mm"]["wrong_direction_reuse_declared_transform"]["p90"] > 0.05

    wrong_declared_unit = _cross_view_nearest_neighbor_metrics(
        stacked,
        np.eye(3),
        [-1.0, 0.0, 0.0],
        declared_translation_unit="cm",
    )
    with pytest.raises(ScaredA00PreprocessError, match="only the mm calibration hypothesis"):
        _verify_cross_view_evidence(
            wrong_declared_unit,
            wrong_declared_unit,
            declared_unit="cm",
            declared_direction="right_from_left",
            tolerance_mm=1e-9,
            acceptance=acceptance,
        )
    with pytest.raises(ScaredA00PreprocessError, match="only right_from_left"):
        _verify_cross_view_evidence(
            measured,
            measured,
            declared_unit="mm",
            declared_direction="left_from_right",
            tolerance_mm=1e-9,
            acceptance=acceptance,
        )


def test_cross_view_gate_rejects_bad_geometry_even_when_evidence_matches() -> None:
    poor = {
        "valid_point_policy": "finite_nonzero_xyz_positive_z",
        "point_counts": {"left": 4, "right": 4},
        "metrics_mm": {
            "translation_as_mm": {"median": 1.0, "p90": 2.0, "p99": 3.0},
            "translation_as_cm": {"median": 4.0, "p90": 5.0, "p99": 6.0},
            "translation_as_m": {"median": 7.0, "p90": 8.0, "p99": 9.0},
            "wrong_direction_reuse_declared_transform": {
                "median": 4.0,
                "p90": 5.0,
                "p99": 6.0,
            },
        },
    }
    with pytest.raises(ScaredA00PreprocessError, match="absolute cross-view residual thresholds"):
        _verify_cross_view_evidence(
            poor,
            poor,
            declared_unit="mm",
            declared_direction="right_from_left",
            tolerance_mm=0.0,
            acceptance=_config()["validation"]["cross_view_acceptance"],
        )


def test_output_symlink_component_is_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    repo = tmp_path / "repo"
    output = tmp_path / "output"
    for path in (raw, repo, output):
        path.mkdir()
    (output / "dataset_1").symlink_to(repo, target_is_directory=True)
    final = output / "dataset_1" / "keyframe_1"
    with pytest.raises(ScaredA00PreprocessError, match="symlink component"):
        _assert_safe_output_destination(
            output,
            final,
            raw_root=raw,
            repo_root=repo,
            require_existing_parent=True,
        )


def test_invocation_record_pins_resolved_paths_and_python_process(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("raw", "processed", "config.yaml", "evidence.json")]
    invocation = _build_invocation_record(
        data_root=paths[0],
        output_root=paths[1],
        sample_id="dataset_1/keyframe_1/frame_data000000",
        config_path=paths[2],
        evidence_path=paths[3],
        invocation_argv=["a00", "prepare", "--sample-id", "dataset_1/keyframe_1/frame_data000000"],
    )
    assert invocation["executable"] == sys.executable
    assert invocation["module"] == "reliable_endo_gs.cli.main"
    assert invocation["resolved_paths"]["output_root"] == paths[1].resolve().as_posix()
    assert invocation["resolved_paths"]["config"] == paths[2].resolve().as_posix()


def test_gt_free_a00_preflight_reports_runnable_smoke_command_without_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    config_path = tmp_path / "a00.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    data_root, sample_id = _processed_fixture(tmp_path / "processed", config_path)

    def fail_if_loaded(*args: object, **kwargs: object) -> None:
        pytest.fail("metadata-only preflight called the A00 sample loader")

    monkeypatch.setattr("reliable_endo_gs.cli.main.ScaredA00Adapter.load", fail_if_loaded)

    report = run_a00_preflight(
        data_root,
        sample_id,
        roles_path=Path("configs/teachers/a01_scared.yaml"),
        config_path=config_path,
        print_smoke_command=True,
    )

    assert report["status"] == "ready_for_a00_smoke"
    assert report["ground_truth_loaded"] is False
    assert report["deployment_batch_loaded"] is False
    command = report["a00_smoke_command"]
    assert command.startswith("env PYTHONPATH=")
    assert sys.executable in command
    assert "-m reliable_endo_gs.cli.main a00 smoke" in command
    assert "--data-root" in command


def test_a00_preflight_rejects_nonhex_provenance_digest(tmp_path: Path) -> None:
    config = _config()
    config_path = tmp_path / "a00.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    data_root, sample_id = _processed_fixture(tmp_path / "processed", config_path)
    provenance_path = (
        data_root / "dataset_1" / "keyframe_1" / "data" / "a00_provenance.json"
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["calibration_evidence_sha256"] = "g" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ScaredA00SampleError, match="SHA-256 hex digest"):
        run_a00_preflight(
            data_root,
            sample_id,
            roles_path=Path("configs/teachers/a01_scared.yaml"),
            config_path=config_path,
            print_smoke_command=False,
        )


def test_a00_preflight_rejects_valid_but_mismatched_evidence_identity(tmp_path: Path) -> None:
    config = _config()
    config_path = tmp_path / "a00.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    data_root, sample_id = _processed_fixture(tmp_path / "processed", config_path)
    data_dir = data_root / "dataset_1" / "keyframe_1" / "data"
    provenance_path = data_dir / "a00_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    protocol_path = data_dir / "a00_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["metric_geometry_evidence_sha256"] = "e" * 64
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    provenance["output_files_sha256"]["a00_protocol.json"] = hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ScaredA00ProtocolError, match="metric-geometry evidence identity mismatch"):
        run_a00_preflight(
            data_root,
            sample_id,
            roles_path=Path("configs/teachers/a01_scared.yaml"),
            config_path=config_path,
            print_smoke_command=False,
        )


def test_a00_preflight_rehashes_recorded_external_evidence_file(tmp_path: Path) -> None:
    config = _config()
    config_path = tmp_path / "a00.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    data_root, sample_id = _processed_fixture(tmp_path / "processed", config_path)
    provenance_path = data_root / "dataset_1" / "keyframe_1" / "data" / "a00_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    evidence_path = Path(provenance["invocation"]["resolved_paths"]["calibration_evidence"])
    evidence_path.write_text(evidence_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ScaredA00SampleError, match="evidence file SHA-256 differs"):
        run_a00_preflight(
            data_root,
            sample_id,
            roles_path=Path("configs/teachers/a01_scared.yaml"),
            config_path=config_path,
            print_smoke_command=False,
        )


def test_a00_preflight_rejects_protocol_calibration_evidence_mismatch(tmp_path: Path) -> None:
    config = _config()
    config_path = tmp_path / "a00.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    data_root, sample_id = _processed_fixture(tmp_path / "processed", config_path)
    data_dir = data_root / "dataset_1" / "keyframe_1" / "data"
    protocol_path = data_dir / "a00_protocol.json"
    provenance_path = data_dir / "a00_provenance.json"
    original_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    mismatches = (
        ("extrinsics", "raw_source_field", "frame_data.unverified.R/T"),
        ("extrinsics", "raw_units", "cm"),
        ("extrinsics", "unit_evidence_sha256", "d" * 64),
        ("preprocessing", "source_translation_field", "frame_data.unverified.R/T"),
        ("preprocessing", "source_translation_unit", "cm"),
    )
    for section, field, invalid_value in mismatches:
        protocol = json.loads(json.dumps(original_protocol))
        protocol[section][field] = invalid_value
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["output_files_sha256"]["a00_protocol.json"] = hashlib.sha256(
            protocol_path.read_bytes()
        ).hexdigest()
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        with pytest.raises(ScaredA00ProtocolError, match="calibration identity differs"):
            run_a00_preflight(
                data_root,
                sample_id,
                roles_path=Path("configs/teachers/a01_scared.yaml"),
                config_path=config_path,
                print_smoke_command=False,
            )


def test_a00_preflight_rejects_evidence_member_identity_mismatch(tmp_path: Path) -> None:
    config = _config()
    config_path = tmp_path / "a00.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    data_root, sample_id = _processed_fixture(tmp_path / "processed", config_path)
    provenance_path = data_root / "dataset_1" / "keyframe_1" / "data" / "a00_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["frame_correspondence"]["scene_points_member_sha256"] = "d" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ScaredA00SampleError, match="member hash correspondence is inconsistent"):
        run_a00_preflight(
            data_root,
            sample_id,
            roles_path=Path("configs/teachers/a01_scared.yaml"),
            config_path=config_path,
            print_smoke_command=False,
        )


def test_a00_preflight_validates_complete_protocol_semantics(tmp_path: Path) -> None:
    config = _config()
    config_path = tmp_path / "a00.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    data_root, sample_id = _processed_fixture(tmp_path / "processed", config_path)
    data_dir = data_root / "dataset_1" / "keyframe_1" / "data"
    protocol_path = data_dir / "a00_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["disparity"]["sign"] = "u_right_minus_left"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    provenance_path = data_dir / "a00_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["output_files_sha256"]["a00_protocol.json"] = hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ScaredA00ProtocolError, match="A00 disparity must declare"):
        run_a00_preflight(
            data_root,
            sample_id,
            roles_path=Path("configs/teachers/a01_scared.yaml"),
            config_path=config_path,
            print_smoke_command=False,
        )


def test_a00_preflight_requires_explicit_protocol_images_grid(tmp_path: Path) -> None:
    config = _config()
    config_path = tmp_path / "a00.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    data_root, sample_id = _processed_fixture(tmp_path / "processed", config_path)
    data_dir = data_root / "dataset_1" / "keyframe_1" / "data"
    protocol_path = data_dir / "a00_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["preprocessing"].pop("images_grid")
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    provenance_path = data_dir / "a00_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["output_files_sha256"]["a00_protocol.json"] = hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ScaredA00ProtocolError, match="images_grid must be explicit"):
        run_a00_preflight(
            data_root,
            sample_id,
            roles_path=Path("configs/teachers/a01_scared.yaml"),
            config_path=config_path,
            print_smoke_command=False,
        )


def test_a00_cli_parses_preflight_and_smoke(tmp_path: Path) -> None:
    common = [
        "--data-root",
        str(tmp_path),
        "--sample-id",
        "dataset_1/keyframe_1/frame_data000000",
        "--config",
        str(tmp_path / "a00.yaml"),
    ]
    preflight = build_parser().parse_args(["a00", "preflight", *common])
    smoke = build_parser().parse_args(["a00", "smoke", *common])
    assert preflight.a00_command == "preflight"
    assert smoke.a00_command == "smoke"
