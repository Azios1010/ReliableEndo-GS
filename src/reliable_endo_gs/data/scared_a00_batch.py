"""Sequential full-dataset A00 preprocessing for original SCARED.

Raw video and archive members are streamed once per keyframe. Rectified RGB,
metric GT disparity, calibration, protocol and provenance are written outside
the repository; deployment loaders continue to keep GT out of their batch.
"""

from __future__ import annotations

import io
import json
import os
import re
import tarfile
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

from reliable_endo_gs.data.scared_a00_identity import a00_processor_compatibility
from reliable_endo_gs.data.scared_a00_preprocess import (
    ScaredA00PreprocessError,
    _assert_safe_output_destination,
    _build_invocation_record,
    _cross_view_nearest_neighbor_metrics,
    _read_config,
    _read_json_bytes,
    _required_mapping,
    _sha256_bytes,
    _sha256_file,
    _validate_calibration_evidence,
    _verify_calibration_yaml_matches_frame_data,
    _verify_cross_view_evidence,
    _write_json,
    rectify_scared_sample_arrays,
)
from reliable_endo_gs.teachers.roles import DEFAULT_A01_SCARED_ROLES, A01ScaredDataRoles

_KEYFRAME_PATTERN = re.compile(r"^keyframe_[0-5]$")
_OFFICIAL_METRIC_GT_URL = "https://endovissub2019-scared.grand-challenge.org/About/"
_SENTINEL = object()


def _archive_payloads(archive_path: Path, suffix: str) -> Iterator[tuple[str, bytes]]:
    """Stream matching regular-file members from one gzip tar archive."""
    try:
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.casefold().endswith(suffix):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ScaredA00PreprocessError(
                        f"unable to read {member.name!r} from {archive_path}"
                    )
                yield Path(member.name).name, extracted.read()
    except (OSError, tarfile.TarError) as error:
        raise ScaredA00PreprocessError(f"unable to stream archive {archive_path}: {error}") from error


def _paired_archive_payloads(
    frame_archive: Path, scene_archive: Path
) -> Iterator[tuple[str, bytes, str, bytes, str, str]]:
    """Require exact ordered frame_data/scene_points ID correspondence."""
    frame_members = _archive_payloads(frame_archive, ".json")
    scene_members = _archive_payloads(scene_archive, ".tiff")
    for index, pair in enumerate(
        zip_longest(frame_members, scene_members, fillvalue=_SENTINEL)
    ):
        frame_item, scene_item = pair
        if frame_item is _SENTINEL or scene_item is _SENTINEL:
            raise ScaredA00PreprocessError(
                f"frame_data and scene_points archive counts differ at index {index}"
            )
        frame_name, frame_bytes = frame_item
        scene_name, scene_bytes = scene_item
        expected_frame = f"frame_data{index:06d}.json"
        expected_scene = f"scene_points{index:06d}.tiff"
        if frame_name != expected_frame or scene_name != expected_scene:
            raise ScaredA00PreprocessError(
                "archive member correspondence is not one-to-one in numeric order: "
                f"expected {expected_frame}/{expected_scene}, got {frame_name}/{scene_name}"
            )
        yield (
            frame_name.removesuffix(".json"),
            frame_bytes,
            scene_name.removesuffix(".tiff"),
            scene_bytes,
            _sha256_bytes(frame_bytes),
            _sha256_bytes(scene_bytes),
        )


def _video_archive_samples(
    video_path: Path, frame_archive: Path, scene_archive: Path
) -> Iterator[tuple[str, bytes, str, bytes, str, str, Any]]:
    """Pair the two archives with sequential video decoding and reject length drift."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ScaredA00PreprocessError(f"unable to open SCARED video {video_path}")
    try:
        for frame_id, frame_bytes, scene_id, scene_bytes, frame_hash, scene_hash in (
            _paired_archive_payloads(frame_archive, scene_archive)
        ):
            success, stacked_video = capture.read()
            if not success:
                raise ScaredA00PreprocessError(
                    f"SCARED video ended before archive sample {frame_id}"
                )
            yield frame_id, frame_bytes, scene_id, scene_bytes, frame_hash, scene_hash, stacked_video
        success, _extra_frame = capture.read()
        if success:
            raise ScaredA00PreprocessError(
                "SCARED video has frames beyond the matched frame_data/scene_points archives"
            )
    finally:
        capture.release()


def _config_for_calibration_source(
    base_config: Mapping[str, object],
    calibration_sha256: str,
    output_root: Path,
) -> Path:
    """Write an external config variant pinned to one exact calibration YAML."""
    import yaml

    calibration = _required_mapping(base_config.get("calibration"), "config.calibration")
    if calibration.get("calibration_value_source_file") != "endoscope_calibration.yaml":
        raise ScaredA00PreprocessError(
            "batch preprocessing requires the sibling endoscope_calibration.yaml value source"
        )
    variant = json.loads(json.dumps(base_config))
    variant["calibration"]["trusted_calibration_value_source_sha256"] = calibration_sha256
    config_bytes = yaml.safe_dump(variant, sort_keys=False, allow_unicode=True).encode("utf-8")
    config_path = (
        output_root
        / "_provenance"
        / "configs"
        / f"a00_scared_{calibration_sha256[:16]}.yaml"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        if config_path.read_bytes() != config_bytes:
            raise ScaredA00PreprocessError(
                f"refusing to overwrite a different calibration config at {config_path}"
            )
    else:
        temporary_path = config_path.with_suffix(".yaml.partial")
        temporary_path.write_bytes(config_bytes)
        os.replace(temporary_path, config_path)
    return config_path


def _make_calibration_evidence(
    *,
    data_root: Path,
    sample_id: str,
    frame_bytes: bytes,
    frame_hash: str,
    scene_bytes: bytes,
    scene_hash: str,
    calibration_path: Path,
    calibration_sha256: str,
    config: Mapping[str, object],
    config_path: Path,
    roles: A01ScaredDataRoles,
    failed_candidates: Sequence[Mapping[str, str]],
    measured: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Build one role-scoped empirical calibration evidence record."""
    frame_data = _read_json_bytes(frame_bytes, f"{sample_id.rsplit('/', 1)[-1]}.json")
    calibration = _required_mapping(
        frame_data.get("camera-calibration"), "frame_data.camera-calibration"
    )
    _verify_calibration_yaml_matches_frame_data(
        calibration_path, calibration, expected_sha256=calibration_sha256
    )
    validation_config = _required_mapping(config.get("validation"), "config.validation")
    acceptance = _required_mapping(
        validation_config.get("cross_view_acceptance"), "validation.cross_view_acceptance"
    )
    _verify_cross_view_evidence(
        measured,
        measured,
        declared_unit="mm",
        declared_direction="right_from_left",
        tolerance_mm=float(validation_config.get("cross_view_evidence_tolerance_mm", -1.0)),
        acceptance=acceptance,
    )
    if validation_config.get("scene_points_xyz_unit") != "mm":
        raise ScaredA00PreprocessError("batch SCARED geometry evidence requires scene_points in mm")
    input_config = _required_mapping(config.get("input"), "config.input")
    if input_config.get("stacked_scene_points_layout") != "top_left_bottom_right_xyz":
        raise ScaredA00PreprocessError("unsupported stacked scene_points layout in A00 config")
    source_relative = f"{sample_id.rsplit('/', 1)[0]}/endoscope_calibration.yaml"
    source = {
        "field_path": "R/T",
        "source_title": "SCARED keyframe endoscope calibration YAML",
        "source_uri": calibration_path.as_uri(),
        "source_path": source_relative,
        "source_sha256": calibration_sha256,
        "frame_data_match": {
            "source_field": "frame_data.camera-calibration.R/T",
            "frame_data_member_sha256": frame_hash,
            "R_matches": True,
            "T_matches": True,
        },
    }
    evidence: dict[str, object] = {
        "schema_version": 1,
        "scope": "role_scoped_calibration_source",
        "sample_id": sample_id,
        "data_role": roles.role_for_dataset(sample_id.split("/", 1)[0]),
        "calibration": {
            "source_field": "frame_data.camera-calibration.R/T",
            "translation_unit": "mm",
            "transform_direction": "right_from_left",
            "calibration_value_source": source,
        },
        "direction_resolution": {
            "method": "cross_view_nearest_neighbor",
            "resolution_basis": "empirical_paired_cloud_residuals",
            "selected_direction": "right_from_left",
            "candidate_transform": "R @ left_xyz + T -> right_xyz",
            "direction_control": "R @ right_xyz + T -> left_xyz",
            "scene_points_layout": "top_left_bottom_right_xyz",
            "metric_ground_truth_source_uri": _OFFICIAL_METRIC_GT_URL,
        },
        "unit_resolution": {
            "method": "cross_view_nearest_neighbor",
            "resolution_basis": "empirical_cross_view_against_metric_ground_truth",
            "selected_translation_unit": "mm",
            "metric_ground_truth_source_uri": _OFFICIAL_METRIC_GT_URL,
        },
        "source_member_sha256": {"frame_data": frame_hash, "scene_points": scene_hash},
        "frame_correspondence": {
            "frame_data_id": sample_id.rsplit("/", 1)[-1],
            "scene_points_id": sample_id.rsplit("/", 1)[-1].replace("frame_data", "scene_points"),
            "matched": True,
            "ambiguous": False,
        },
        "independent_validation": {
            "method": "cross_view_nearest_neighbor",
            **dict(measured),
        },
        "metric_ground_truth": {
            "xyz_units": "mm",
            "depth_units": "mm",
            "source_uri": _OFFICIAL_METRIC_GT_URL,
        },
        "evidence_generation": {
            "script_path": Path(__file__).resolve().as_posix(),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "command": "reliable_endo_gs.cli.main a00 prepare-dataset",
            "role_scoped_candidate_search": [dict(item) for item in failed_candidates],
            "config_path": config_path.as_posix(),
            "data_root": data_root.as_posix(),
        },
    }
    validated = _validate_calibration_evidence(
        evidence,
        sample_id=sample_id,
        trusted_value_source_sha256=calibration_sha256,
    )
    return evidence, validated


def _calibration_evidence_for_role_source(
    *,
    data_root: Path,
    output_root: Path,
    dataset_id: str,
    keyframe_id: str,
    calibration_path: Path,
    calibration_sha256: str,
    frame_archive: Path,
    scene_archive: Path,
    base_config: Mapping[str, object],
    roles: A01ScaredDataRoles,
) -> tuple[Path, Path, dict[str, object], dict[str, object], str]:
    """Return evidence/config for a calibration source, scoped by data role."""
    import tifffile

    role = roles.role_for_dataset(dataset_id)
    config_path = _config_for_calibration_source(base_config, calibration_sha256, output_root)
    config, _config_sha = _read_config(config_path)
    evidence_dir = output_root / "_provenance" / "calibration_evidence" / role
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / f"{calibration_sha256}.json"
    if evidence_path.exists():
        try:
            evidence_value = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ScaredA00PreprocessError(
                f"existing calibration evidence is invalid at {evidence_path}: {error}"
            ) from error
        if not isinstance(evidence_value, dict) or evidence_value.get("data_role") != role:
            raise ScaredA00PreprocessError(
                f"existing calibration evidence role mismatch at {evidence_path}"
            )
        evidence_sample_id = evidence_value.get("sample_id")
        if not isinstance(evidence_sample_id, str):
            raise ScaredA00PreprocessError("calibration evidence sample_id is missing")
        validated = _validate_calibration_evidence(
            evidence_value,
            sample_id=evidence_sample_id,
            trusted_value_source_sha256=calibration_sha256,
        )
        calibration = _required_mapping(validated.get("calibration"), "evidence.calibration")
        source = _required_mapping(
            calibration.get("calibration_value_source"), "evidence calibration value source"
        )
        if source.get("source_sha256") != calibration_sha256:
            raise ScaredA00PreprocessError("cached calibration evidence numeric source hash changed")
        evidence_hash = _sha256_file(evidence_path)
        return evidence_path, config_path, config, evidence_value, evidence_hash

    attempts: list[dict[str, str]] = []
    for frame_id, frame_bytes, scene_id, scene_bytes, frame_hash, scene_hash in _paired_archive_payloads(
        frame_archive, scene_archive
    ):
        if scene_id != frame_id.replace("frame_data", "scene_points"):
            raise ScaredA00PreprocessError(
                f"frame/scene correspondence mismatch while resolving calibration for {dataset_id}/{keyframe_id}"
            )
        sample_id = f"{dataset_id}/{keyframe_id}/{frame_id}"
        frame_data = _read_json_bytes(frame_bytes, f"{frame_id}.json")
        calibration_obj = _required_mapping(
            frame_data.get("camera-calibration"), "frame_data.camera-calibration"
        )
        # A mismatch between the calibration file and frame metadata is a
        # source-integrity failure, not a candidate to skip.
        _verify_calibration_yaml_matches_frame_data(
            calibration_path, calibration_obj, expected_sha256=calibration_sha256
        )
        scene_points = tifffile.imread(io.BytesIO(scene_bytes))
        try:
            measured = _cross_view_nearest_neighbor_metrics(
                scene_points,
                calibration_obj.get("R"),
                calibration_obj.get("T"),
                declared_translation_unit="mm",
                correspondence_policy="mutual",
            )
            validation_config = _required_mapping(config.get("validation"), "config.validation")
            acceptance = _required_mapping(
                validation_config.get("cross_view_acceptance"),
                "validation.cross_view_acceptance",
            )
            _verify_cross_view_evidence(
                measured,
                measured,
                declared_unit="mm",
                declared_direction="right_from_left",
                tolerance_mm=float(validation_config.get("cross_view_evidence_tolerance_mm", -1.0)),
                acceptance=acceptance,
            )
        except ScaredA00PreprocessError as error:
            attempts.append({"sample_id": sample_id, "reason": str(error)})
            continue
        evidence_value, validated = _make_calibration_evidence(
            data_root=data_root,
            sample_id=sample_id,
            frame_bytes=frame_bytes,
            frame_hash=frame_hash,
            scene_bytes=scene_bytes,
            scene_hash=scene_hash,
            calibration_path=calibration_path,
            calibration_sha256=calibration_sha256,
            config=config,
            config_path=config_path,
            roles=roles,
            failed_candidates=attempts,
            measured=measured,
        )
        _write_json(evidence_path, evidence_value)
        evidence_hash = _sha256_file(evidence_path)
        _validate_calibration_evidence(
            evidence_value,
            sample_id=sample_id,
            trusted_value_source_sha256=calibration_sha256,
        )
        return evidence_path, config_path, config, evidence_value, evidence_hash
    reasons = "; ".join(item["reason"] for item in attempts[-4:])
    raise ScaredA00PreprocessError(
        f"no frame in {dataset_id}/{keyframe_id} passed the empirical mm/right_from_left calibration gate; "
        f"attempted {len(attempts)} frames. Recent failures: {reasons}"
    )


def _resume_keyframe_if_complete(
    *,
    final_keyframe: Path,
    source_paths: Mapping[str, Path],
    role: str,
    role_hash: str,
    processor_sha256: str,
    batch_code_sha256: str,
    config_sha256: str,
    calibration_sha256: str,
) -> int | None:
    """Skip a previously completed keyframe only when its provenance identity matches."""
    manifest_path = final_keyframe / "data" / "a00_keyframe_manifest.json"
    if not final_keyframe.exists():
        return None
    if not manifest_path.is_file():
        raise ScaredA00PreprocessError(
            f"processed keyframe already exists without a batch manifest; refusing overwrite: {final_keyframe}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScaredA00PreprocessError(f"invalid existing keyframe manifest: {error}") from error
    expected = {
        "data_role": role,
        "role_contract_hash": role_hash,
        "config_sha256": config_sha256,
        "calibration_value_source_sha256": calibration_sha256,
    }
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != value for key, value in expected.items()
    ):
        raise ScaredA00PreprocessError(
            f"existing keyframe output belongs to a different role/config/code identity: {final_keyframe}"
        )
    source_stats = manifest.get("source_file_stats")
    if not isinstance(source_stats, dict):
        raise ScaredA00PreprocessError("existing keyframe manifest lacks source file stats")
    for name, path in source_paths.items():
        stat = path.stat()
        expected_stat = source_stats.get(name)
        if expected_stat != {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
            raise ScaredA00PreprocessError(
                f"source file changed since keyframe preprocessing: {path}"
            )
    sample_ids = manifest.get("sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ScaredA00PreprocessError("existing keyframe manifest has no sample IDs")
    metrics = manifest.get("sample_metric_validation")
    if (
        not isinstance(metrics, list)
        or len(metrics) != len(sample_ids)
        or any(not isinstance(item, dict) for item in metrics)
        or [item.get("sample_id") for item in metrics] != sample_ids
    ):
        raise ScaredA00PreprocessError("existing keyframe metric/sample correspondence is invalid")
    compatibility = a00_processor_compatibility(
        (manifest.get("processor_sha256"), manifest.get("batch_processor_sha256")),
        (processor_sha256, batch_code_sha256),
        metrics,
    )
    if compatibility is None:
        raise ScaredA00PreprocessError(
            f"existing keyframe output belongs to an unreviewed code identity: {final_keyframe}"
        )
    data_dir = final_keyframe / "data"
    for sample_id, metric_summary in zip(sample_ids, metrics, strict=True):
        frame_id = str(sample_id).rsplit("/", 1)[-1]
        provenance_path = data_dir / "a00_provenance" / f"{frame_id}.json"
        required = (
            data_dir / "left_finalpass" / f"{frame_id}.png",
            data_dir / "right_finalpass" / f"{frame_id}.png",
            data_dir / "frame_data" / f"{frame_id}.json",
            data_dir / "newpram_data" / f"{frame_id}.json",
            data_dir / "disparity" / f"{frame_id}.tiff",
            data_dir / "reprojection_data" / f"{frame_id}.json",
            provenance_path,
        )
        if not all(path.is_file() for path in required):
            raise ScaredA00PreprocessError(
                f"existing keyframe output is incomplete; refusing overwrite: {final_keyframe}"
            )
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ScaredA00PreprocessError(
                f"existing sample provenance is invalid at {provenance_path}: {error}"
            ) from error
        if (
            not isinstance(provenance, dict)
            or provenance.get("sample_id") != sample_id
            or provenance.get("data_role") != role
        ):
            raise ScaredA00PreprocessError(
                f"existing sample identity/role mismatch in {provenance_path}"
            )
        preprocessing = provenance.get("preprocessing")
        if (
            not isinstance(preprocessing, dict)
            or preprocessing.get("code_sha256") != manifest.get("processor_sha256")
            or preprocessing.get("batch_code_sha256") != manifest.get("batch_processor_sha256")
            or provenance.get("metric_validation")
            != {key: value for key, value in metric_summary.items() if key != "sample_id"}
        ):
            raise ScaredA00PreprocessError(
                f"existing sample processor/metric identity mismatch: {provenance_path}"
            )
        output_hashes = provenance.get("output_files_sha256")
        if not isinstance(output_hashes, dict):
            raise ScaredA00PreprocessError(
                f"existing sample provenance has no output hashes: {provenance_path}"
            )
        for relative_name, expected_hash in output_hashes.items():
            if not isinstance(relative_name, str) or not isinstance(expected_hash, str):
                raise ScaredA00PreprocessError(
                    f"existing sample output hash entry is malformed: {provenance_path}"
                )
            output_path = (data_dir / relative_name).resolve(strict=True)
            if (
                data_dir.resolve(strict=True) not in output_path.parents
                or not output_path.is_file()
            ):
                raise ScaredA00PreprocessError(
                    f"existing sample output path escapes its keyframe: {relative_name}"
                )
            if _sha256_file(output_path) != expected_hash:
                raise ScaredA00PreprocessError(
                    f"existing sample output hash mismatch: {output_path}"
                )
    return len(sample_ids)


def prepare_scared_a00_dataset(
    *,
    data_root: Path,
    output_root: Path,
    config_path: Path,
    roles: A01ScaredDataRoles = DEFAULT_A01_SCARED_ROLES,
    invocation_argv: Sequence[str] | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Prepare all available original SCARED folds 1–9 in one bounded stream.

    Calibration unit/direction evidence is generated from a sample in the
    same role as the keyframe. It is reused only for byte-identical numeric
    calibration YAML sources. Every frame verifies source correspondence and
    calibration. Available targets pass Q/A00/metric-GT checks; empty targets
    explicitly report unavailable metric validation before they are committed.
    """
    import cv2
    import numpy as np
    import tifffile

    from reliable_endo_gs.data.scared_a00 import build_a01_manifest, load_a01_manifest

    root = Path(data_root).expanduser().resolve(strict=True)
    output_candidate = Path(os.path.abspath(Path(output_root).expanduser()))
    repo_root = Path(__file__).resolve().parents[3]
    _assert_safe_output_destination(
        output_candidate,
        output_candidate / "_provenance" / "placeholder",
        raw_root=root,
        repo_root=repo_root,
        require_existing_parent=False,
    )
    config_file = Path(config_path).expanduser().resolve(strict=True)
    base_config, _base_config_sha256 = _read_config(config_file)
    input_config = _required_mapping(base_config.get("input"), "config.input")
    height = int(input_config.get("frame_height", 0))
    width = int(input_config.get("frame_width", 0))
    if height < 1 or width < 1:
        raise ScaredA00PreprocessError("A00 config frame dimensions must be positive")
    output_candidate.mkdir(parents=True, exist_ok=True)
    output = output_candidate.resolve(strict=True)
    code_sha256 = _sha256_file(Path(__file__).resolve())
    processor_path = Path(__file__).resolve().parent / "scared_a00_preprocess.py"
    processor_sha256 = _sha256_file(processor_path)
    summary: dict[str, object] = {
        "schema_version": 1,
        "dataset_name": "scared",
        "data_root": root.as_posix(),
        "processed_root": output.as_posix(),
        "role_contract_hash": roles.role_hash,
        "processor_sha256": processor_sha256,
        "batch_processor_sha256": code_sha256,
        "config_source_path": config_file.as_posix(),
        "config_source_sha256": _base_config_sha256,
        "active_roles": {role: list(ids) for role, ids in roles.roles.items()},
        "folds": {},
        "keyframes": [],
        "skipped_static_keyframes": [],
        "sample_count": 0,
        "resumed_sample_count": 0,
        "samples_without_gt": 0,
    }
    progress_path = output / "_provenance" / "progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    source_keyframes: list[tuple[str, str, Path, Path, Path, Path, Path]] = []
    for dataset_id in roles.dataset_ids:
        role = roles.role_for_dataset(dataset_id)
        fold_count = 0
        for keyframe_root in sorted(
            path for path in (root / dataset_id).glob("keyframe_*") if path.is_dir()
        ):
            if _KEYFRAME_PATTERN.fullmatch(keyframe_root.name) is None:
                raise ScaredA00PreprocessError(
                    f"unsupported keyframe identifier in active fold: {keyframe_root}"
                )
            video = keyframe_root / str(input_config.get("video_file"))
            frame_archive = keyframe_root / str(input_config.get("frame_data_archive"))
            scene_archive = keyframe_root / str(input_config.get("scene_points_archive"))
            calibration = keyframe_root / str(
                _required_mapping(base_config.get("calibration"), "config.calibration").get(
                    "calibration_value_source_file"
                )
            )
            sequence_sources = (video, frame_archive, scene_archive)
            present = tuple(path.is_file() for path in sequence_sources)
            if not any(present):
                summary["skipped_static_keyframes"].append(
                    f"{dataset_id}/{keyframe_root.name}"
                )
                continue
            if not all(present) or not calibration.is_file():
                missing = [path.as_posix() for path in (*sequence_sources, calibration) if not path.is_file()]
                raise ScaredA00PreprocessError(
                    f"incomplete sequence sources for {dataset_id}/{keyframe_root.name}: {missing}"
                )
            source_keyframes.append(
                (dataset_id, role, keyframe_root, video, frame_archive, scene_archive, calibration)
            )
            fold_count += 1
        if fold_count == 0:
            raise ScaredA00PreprocessError(f"no video/archive keyframes found for active fold {dataset_id}")
        summary["folds"][dataset_id] = {"data_role": role, "keyframe_count": fold_count, "sample_count": 0}

    calibration_evidence_cache: dict[tuple[str, str], tuple[Path, Path, dict[str, object], dict[str, object], str]] = {}
    for (
        dataset_id,
        role,
        keyframe_root,
        video_path,
        frame_archive,
        scene_archive,
        calibration_path,
    ) in source_keyframes:
        keyframe_id = keyframe_root.name
        calibration_sha256 = _sha256_file(calibration_path)
        cache_key = (role, calibration_sha256)
        evidence_record = calibration_evidence_cache.get(cache_key)
        if evidence_record is None:
            evidence_record = _calibration_evidence_for_role_source(
                data_root=root,
                output_root=output,
                dataset_id=dataset_id,
                keyframe_id=keyframe_id,
                calibration_path=calibration_path,
                calibration_sha256=calibration_sha256,
                frame_archive=frame_archive,
                scene_archive=scene_archive,
                base_config=base_config,
                roles=roles,
            )
            calibration_evidence_cache[cache_key] = evidence_record
        evidence_path, config_variant_path, config, validated_evidence, evidence_sha256 = evidence_record
        _config_mapping, config_sha256 = _read_config(config_variant_path)
        _required_mapping(
            _required_mapping(validated_evidence.get("calibration"), "evidence.calibration").get(
                "calibration_value_source"
            ),
            "evidence calibration value source",
        )
        evidence_sample_id = str(validated_evidence.get("sample_id", ""))
        if not evidence_sample_id:
            raise ScaredA00PreprocessError("validated calibration evidence has no representative sample ID")
        # Final-test geometry can never establish train calibration semantics,
        # and train evidence is not used to inspect final-test targets.
        if roles.role_for_dataset(evidence_sample_id.split("/", 1)[0]) != role:
            raise ScaredA00PreprocessError("calibration evidence crosses the active train/test role boundary")

        final_keyframe = output / dataset_id / keyframe_id
        source_paths = {
            "rgb_video": video_path,
            "frame_data_archive": frame_archive,
            "scene_points_archive": scene_archive,
            "calibration_value_source": calibration_path,
        }
        source_file_stats = {
            name: {"size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for name, path in source_paths.items()
        }
        resumed_count = _resume_keyframe_if_complete(
            final_keyframe=final_keyframe,
            source_paths=source_paths,
            role=role,
            role_hash=roles.role_hash,
            processor_sha256=processor_sha256,
            batch_code_sha256=code_sha256,
            config_sha256=config_sha256,
            calibration_sha256=calibration_sha256,
        )
        if resumed_count is not None:
            summary["sample_count"] = int(summary["sample_count"]) + resumed_count
            summary["resumed_sample_count"] = int(summary["resumed_sample_count"]) + resumed_count
            summary["folds"][dataset_id]["sample_count"] += resumed_count
            resumed_manifest = json.loads(
                (final_keyframe / "data" / "a00_keyframe_manifest.json").read_text(encoding="utf-8")
            )
            empty_count = sum(
                item["valid_pixel_count"] == 0
                for item in resumed_manifest["sample_metric_validation"]
            )
            summary["samples_without_gt"] += empty_count
            summary["keyframes"].append(
                {
                    "keyframe_id": f"{dataset_id}/{keyframe_id}",
                    "sample_count": resumed_count,
                    "status": "resumed",
                    "samples_without_gt": empty_count,
                    "processor_sha256": resumed_manifest["processor_sha256"],
                    "batch_processor_sha256": resumed_manifest["batch_processor_sha256"],
                }
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "keyframe_resumed",
                        "keyframe": f"{dataset_id}/{keyframe_id}",
                        "sample_count": resumed_count,
                    }
                )
            continue

        video_sha256 = _sha256_file(video_path)
        invocation = _build_invocation_record(
            data_root=root,
            output_root=output,
            sample_id=evidence_sample_id,
            config_path=config_variant_path,
            evidence_path=evidence_path,
            invocation_argv=invocation_argv,
        )
        invocation["operation"] = "a00 prepare-dataset"
        invocation["data_role"] = role
        direction_resolution = _required_mapping(
            validated_evidence.get("direction_resolution"), "evidence.direction_resolution"
        )
        metric_gt = _required_mapping(validated_evidence.get("metric_ground_truth"), "evidence.metric_ground_truth")
        translation_unit = "mm"
        scene_points_unit = str(metric_gt.get("xyz_units"))
        if scene_points_unit != "mm":
            raise ScaredA00PreprocessError("active SCARED scene_points geometry must be in mm")
        code_path = Path(__file__).resolve()
        processor_path = code_path.parent / "scared_a00_preprocess.py"

        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"a00-{dataset_id}-{keyframe_id}-", dir=output
        ) as temporary:
            staged_keyframe = Path(temporary) / dataset_id / keyframe_id
            data_dir = staged_keyframe / "data"
            provenance_dir = data_dir / "a00_provenance"
            protocol_path = data_dir / "a00_protocol.json"
            provenance_dir.mkdir(parents=True, exist_ok=True)
            protocol_value: dict[str, object] | None = None
            protocol_sha256: str | None = None
            sample_ids: list[str] = []
            sample_metric_summaries: list[dict[str, object]] = []
            calibration_reference: dict[str, object] | None = None

            for (
                frame_id,
                frame_bytes,
                scene_id,
                scene_bytes,
                frame_member_sha256,
                scene_member_sha256,
                stacked_video,
            ) in _video_archive_samples(video_path, frame_archive, scene_archive):
                sample_id = f"{dataset_id}/{keyframe_id}/{frame_id}"
                if scene_id != frame_id.replace("frame_data", "scene_points"):
                    raise ScaredA00PreprocessError(f"ambiguous source correspondence for {sample_id}")
                frame_data = _read_json_bytes(frame_bytes, f"{frame_id}.json")
                calibration_obj = _required_mapping(
                    frame_data.get(str(input_config.get("calibration_field"))),
                    "frame_data.camera-calibration",
                )
                if calibration_reference is None:
                    calibration_reference = calibration_obj
                    _verify_calibration_yaml_matches_frame_data(
                        calibration_path, calibration_obj, expected_sha256=calibration_sha256
                    )
                else:
                    for field_name in ("KL", "KR", "DL", "DR", "R", "T"):
                        current = np.asarray(calibration_obj.get(field_name), dtype=np.float64)
                        reference = np.asarray(calibration_reference.get(field_name), dtype=np.float64)
                        if current.shape != reference.shape or not np.array_equal(current, reference):
                            raise ScaredA00PreprocessError(
                                f"{sample_id} calibration {field_name} differs from the keyframe calibration source"
                            )
                if stacked_video.shape != (height * 2, width, 3):
                    raise ScaredA00PreprocessError(
                        f"{sample_id} RGB frame shape {stacked_video.shape}; expected {(height * 2, width, 3)}"
                    )
                try:
                    scene_points = tifffile.imread(io.BytesIO(scene_bytes))
                except (OSError, ValueError) as error:
                    raise ScaredA00PreprocessError(
                        f"unable to decode scene_points for {sample_id}: {error}"
                    ) from error
                try:
                    arrays = rectify_scared_sample_arrays(
                        left_bgr=stacked_video[:height],
                        right_bgr=stacked_video[height:],
                        stacked_scene_points=scene_points,
                        calibration=calibration_obj,
                        translation_unit=translation_unit,
                        scene_points_unit=scene_points_unit,
                        config=config,
                        allow_empty_gt=True,
                    )
                except ScaredA00PreprocessError as error:
                    raise ScaredA00PreprocessError(f"{sample_id}: {error}") from error
                if arrays["metric_validation"]["valid_pixel_count"] == 0:
                    record = {
                        "event": "sample_without_gt",
                        "sample_id": sample_id,
                        "metric_validation": arrays["metric_validation"],
                    }
                    with progress_path.open("a", encoding="utf-8") as progress:
                        progress.write(json.dumps(record, sort_keys=True) + "\n")
                    if progress_callback is not None:
                        progress_callback(record)
                sample_ids.append(sample_id)
                sample_metric_summaries.append(
                    {"sample_id": sample_id, **arrays["metric_validation"]}
                )
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
                current_preprocessing: dict[str, object] = {
                    "images_grid": "processed",
                    "processor": "reliable_endo_gs.data.scared_a00_batch",
                    "processor_version": config.get("preprocessing_id"),
                    "code_sha256": _sha256_file(processor_path),
                    "batch_code_sha256": code_sha256,
                    "config_path": config_variant_path.as_posix(),
                    "config_sha256": config_sha256,
                    "calibration_evidence_sha256": evidence_sha256,
                    "calibration_evidence_sample_id": evidence_sample_id,
                    "calibration_value_source_path": f"{dataset_id}/{keyframe_id}/endoscope_calibration.yaml",
                    "calibration_value_source_sha256": calibration_sha256,
                    "source_translation_unit": translation_unit,
                    "source_translation_field": "frame_data.camera-calibration.R/T",
                    "scene_points_xyz_unit": scene_points_unit,
                    "empty_gt_policy": "retain_invalid_zero_target_and_report_unavailable",
                    "direction_resolution": dict(direction_resolution),
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
                    "invocation": invocation,
                    "metric_validation": arrays["metric_validation"],
                    "frame_data_member_sha256": frame_member_sha256,
                    "scene_points_member_sha256": scene_member_sha256,
                }
                if protocol_value is None:
                    protocol_preprocessing = {
                        key: value
                        for key, value in current_preprocessing.items()
                        if key not in ("metric_validation", "frame_data_member_sha256", "scene_points_member_sha256")
                    }
                    protocol_value = {
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
                            "direction": str(direction_resolution["selected_direction"]),
                            "units": "m",
                            "raw_source_field": "frame_data.camera-calibration.R/T",
                            "raw_units": translation_unit,
                            "unit_evidence_sha256": evidence_sha256,
                            "calibration_value_source_sha256": calibration_sha256,
                        },
                        "preprocessing": protocol_preprocessing,
                        "metric_geometry_evidence_sha256": evidence_sha256,
                    }
                    _write_json(protocol_path, protocol_value)
                    protocol_sha256 = _sha256_file(protocol_path)

                frame_path = data_dir / "frame_data" / f"{frame_id}.json"
                newpram_path = data_dir / "newpram_data" / f"{frame_id}.json"
                left_path = data_dir / "left_finalpass" / f"{frame_id}.png"
                right_path = data_dir / "right_finalpass" / f"{frame_id}.png"
                disparity_path = data_dir / "disparity" / f"{frame_id}.tiff"
                q_path = data_dir / "reprojection_data" / f"{frame_id}.json"
                for path in (frame_path, newpram_path, left_path, right_path, disparity_path, q_path):
                    path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(left_path), arrays["left_rectified_bgr"]):
                    raise ScaredA00PreprocessError(f"failed to write rectified left RGB {left_path}")
                if not cv2.imwrite(str(right_path), arrays["right_rectified_bgr"]):
                    raise ScaredA00PreprocessError(f"failed to write rectified right RGB {right_path}")
                frame_path.write_bytes(frame_bytes)
                _write_json(newpram_path, newpram)
                tifffile.imwrite(disparity_path, arrays["disparity"], dtype=np.float32)
                _write_json(q_path, {"reprojection-matrix": arrays["Q"].tolist()})
                output_paths = (
                    left_path,
                    right_path,
                    frame_path,
                    newpram_path,
                    disparity_path,
                    q_path,
                    protocol_path,
                )
                output_hashes = {
                    path.relative_to(data_dir).as_posix(): _sha256_file(path)
                    for path in output_paths
                }
                source_files = {
                    "rgb_video": {
                        "relative_path": str(input_config.get("video_file")),
                        "sha256": video_sha256,
                    },
                    "frame_data_archive": {
                        "relative_path": str(input_config.get("frame_data_archive")),
                        "member": f"{frame_id}.json",
                        "member_sha256": frame_member_sha256,
                    },
                    "scene_points_archive": {
                        "relative_path": str(input_config.get("scene_points_archive")),
                        "member": f"{scene_id}.tiff",
                        "member_sha256": scene_member_sha256,
                    },
                    "calibration_value_source": {
                        "relative_path": "endoscope_calibration.yaml",
                        "sha256": calibration_sha256,
                    },
                }
                provenance = {
                    "schema_version": 1,
                    "dataset_name": "scared",
                    "sample_id": sample_id,
                    "data_role": role,
                    "calibration_evidence_sha256": evidence_sha256,
                    "calibration_evidence_sample_id": evidence_sample_id,
                    "frame_correspondence": {
                        "frame_data_id": frame_id,
                        "scene_points_id": scene_id,
                        "matched": True,
                        "ambiguous": False,
                        "frame_data_member_sha256": frame_member_sha256,
                        "scene_points_member_sha256": scene_member_sha256,
                    },
                    "source_files": source_files,
                    "preprocessing": current_preprocessing,
                    "invocation": invocation,
                    "metric_validation": arrays["metric_validation"],
                    "output_files_sha256": output_hashes,
                }
                _write_json(provenance_dir / f"{frame_id}.json", provenance)
                if len(sample_ids) % 25 == 0:
                    progress_record = {
                        "event": "sample_progress",
                        "keyframe": f"{dataset_id}/{keyframe_id}",
                        "processed_in_keyframe": len(sample_ids),
                    }
                    with progress_path.open("a", encoding="utf-8") as progress:
                        progress.write(json.dumps(progress_record, sort_keys=True) + "\n")
                        progress.flush()
                    if progress_callback is not None:
                        progress_callback(progress_record)

            if not sample_ids or protocol_sha256 is None:
                raise ScaredA00PreprocessError(f"no matched frames found for {dataset_id}/{keyframe_id}")
            keyframe_manifest = {
                "schema_version": 1,
                "dataset_name": "scared",
                "keyframe_id": f"{dataset_id}/{keyframe_id}",
                "data_role": role,
                "role_contract_hash": roles.role_hash,
                "processor_sha256": processor_sha256,
                "batch_processor_sha256": code_sha256,
                "config_sha256": config_sha256,
                "calibration_value_source_sha256": calibration_sha256,
                "calibration_evidence_sha256": evidence_sha256,
                "calibration_evidence_sample_id": evidence_sample_id,
                "protocol_sha256": protocol_sha256,
                "source_file_stats": source_file_stats,
                "source_video_sha256": video_sha256,
                "sample_ids": sample_ids,
                "sample_metric_validation": sample_metric_summaries,
                "samples_without_gt": sum(
                    item["valid_pixel_count"] == 0 for item in sample_metric_summaries
                ),
            }
            _write_json(data_dir / "a00_keyframe_manifest.json", keyframe_manifest)
            (output / dataset_id).mkdir(parents=True, exist_ok=True)
            _assert_safe_output_destination(
                output,
                final_keyframe,
                raw_root=root,
                repo_root=repo_root,
                require_existing_parent=True,
            )
            os.replace(staged_keyframe, final_keyframe)

        count = len(sample_ids)
        empty_count = sum(item["valid_pixel_count"] == 0 for item in sample_metric_summaries)
        summary["samples_without_gt"] += empty_count
        summary["sample_count"] = int(summary["sample_count"]) + count
        summary["folds"][dataset_id]["sample_count"] += count
        keyframe_summary = {
            "keyframe_id": f"{dataset_id}/{keyframe_id}",
            "data_role": role,
            "sample_count": count,
            "samples_without_gt": empty_count,
            "processor_sha256": processor_sha256,
            "batch_processor_sha256": code_sha256,
            "calibration_value_source_sha256": calibration_sha256,
            "calibration_evidence_sha256": evidence_sha256,
            "calibration_evidence_sample_id": evidence_sample_id,
            "config_sha256": config_sha256,
            "status": "processed",
        }
        summary["keyframes"].append(keyframe_summary)
        with progress_path.open("a", encoding="utf-8") as progress:
            progress.write(json.dumps(keyframe_summary, sort_keys=True) + "\n")
            progress.flush()
        if progress_callback is not None:
            progress_callback({"event": "keyframe_complete", **keyframe_summary})

    manifest_path = output / "a01_manifest.json"
    temporary_manifest = output / ".a01_manifest.pending.json"
    manifest = build_a01_manifest(output, temporary_manifest, roles=roles, require_all_folds=True)
    if manifest_path.exists():
        existing = load_a01_manifest(manifest_path, roles=roles)
        if existing.manifest_hash != manifest.manifest_hash:
            temporary_manifest.unlink(missing_ok=True)
            raise ScaredA00PreprocessError(
                "existing A01 manifest differs from the completed preprocessing output"
            )
        temporary_manifest.unlink(missing_ok=True)
    else:
        os.replace(temporary_manifest, manifest_path)
    summary["manifest_path"] = manifest_path.as_posix()
    summary["manifest_hash"] = manifest.manifest_hash
    summary_path = output / "preprocessing_summary.json"
    if summary_path.exists():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ScaredA00PreprocessError(f"invalid existing preprocessing summary: {error}") from error
        if previous.get("manifest_hash") != manifest.manifest_hash:
            raise ScaredA00PreprocessError("existing summary belongs to a different manifest")
    else:
        _write_json(summary_path, summary)
    return summary


__all__ = ["prepare_scared_a00_dataset"]
