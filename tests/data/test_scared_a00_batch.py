"""Focused resume and provenance checks for the empty-GT remediation."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
import yaml

from reliable_endo_gs.cli.main import run_a00_preflight
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, load_a01_manifest
from reliable_endo_gs.data.scared_a00_batch import (
    _paired_archive_payloads,
    _resume_keyframe_if_complete,
    prepare_scared_a00_dataset,
)
from reliable_endo_gs.data.scared_a00_identity import (
    CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR,
    EMPTY_GT_PROCESSOR_PAIR,
    POSITIVE_GT_PROCESSOR_PAIR,
    TYPED_FORMATTED_PROCESSOR_PAIR,
    a00_processor_compatibility,
)
from reliable_endo_gs.data.scared_a00_preprocess import ScaredA00PreprocessError


def test_compatibility_accepts_only_reviewed_pair_with_positive_gt() -> None:
    assert (
        a00_processor_compatibility(
            POSITIVE_GT_PROCESSOR_PAIR, EMPTY_GT_PROCESSOR_PAIR, [{"valid_pixel_count": 12}]
        )
        == "reviewed_positive_gt_compatibility"
    )
    for metrics in (
        [],
        [{"valid_pixel_count": 0}],
        [{"valid_pixel_count": True}],
        [{"valid_pixel_count": 12, "status": "unavailable"}],
    ):
        assert (
            a00_processor_compatibility(
                POSITIVE_GT_PROCESSOR_PAIR, EMPTY_GT_PROCESSOR_PAIR, metrics
            )
            is None
        )
    assert (
        a00_processor_compatibility(
            ("a" * 64, "b" * 64), EMPTY_GT_PROCESSOR_PAIR, [{"valid_pixel_count": 12}]
        )
        is None
    )
    assert (
        a00_processor_compatibility(
            POSITIVE_GT_PROCESSOR_PAIR, ("c" * 64, "d" * 64), [{"valid_pixel_count": 12}]
        )
        is None
    )


def test_compatibility_registry_is_pinned_to_this_exact_remediation() -> None:
    root = Path(__file__).resolve().parents[2] / "src/reliable_endo_gs/data"
    current = tuple(
        hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("scared_a00_preprocess.py", "scared_a00_batch.py")
    )
    assert current == TYPED_FORMATTED_PROCESSOR_PAIR


@pytest.mark.parametrize("count", [0, 12])
def test_typing_format_review_preserves_correspondence_producer(count: int) -> None:
    summary = {"valid_pixel_count": count}
    assert (
        a00_processor_compatibility(
            CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR,
            TYPED_FORMATTED_PROCESSOR_PAIR,
            [summary],
        )
        == "reviewed_typing_format_compatibility"
    )
    assert (
        a00_processor_compatibility(("a" * 64, "b" * 64), TYPED_FORMATTED_PROCESSOR_PAIR, [summary])
        is None
    )
    assert (
        a00_processor_compatibility(
            TYPED_FORMATTED_PROCESSOR_PAIR, CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR, [summary]
        )
        is None
    )


def test_typing_format_review_keeps_historical_metric_gates() -> None:
    assert (
        a00_processor_compatibility(
            POSITIVE_GT_PROCESSOR_PAIR,
            TYPED_FORMATTED_PROCESSOR_PAIR,
            [{"valid_pixel_count": 12}],
        )
        == "reviewed_positive_gt_compatibility"
    )
    for invalid in ([], [{"valid_pixel_count": 0}], [{"valid_pixel_count": True}]):
        assert (
            a00_processor_compatibility(
                POSITIVE_GT_PROCESSOR_PAIR, TYPED_FORMATTED_PROCESSOR_PAIR, invalid
            )
            is None
        )
    empty = {
        "valid_pixel_count": 0,
        "status": "unavailable",
        "q_vs_a00_abs_error_mm": None,
        "q_vs_metric_gt_abs_error_mm": None,
    }
    assert (
        a00_processor_compatibility(
            EMPTY_GT_PROCESSOR_PAIR, TYPED_FORMATTED_PROCESSOR_PAIR, [empty]
        )
        == "reviewed_calibration_evidence_compatibility"
    )
    assert (
        a00_processor_compatibility(
            EMPTY_GT_PROCESSOR_PAIR,
            TYPED_FORMATTED_PROCESSOR_PAIR,
            [{"valid_pixel_count": 0}],
        )
        is None
    )


def _resume_fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    keyframe = tmp_path / "processed/dataset_1/keyframe_1"
    data = keyframe / "data"
    sample = "dataset_1/keyframe_1/frame_data000000"
    frame = "frame_data000000"
    paths = [
        f"left_finalpass/{frame}.png",
        f"right_finalpass/{frame}.png",
        f"frame_data/{frame}.json",
        f"newpram_data/{frame}.json",
        f"disparity/{frame}.tiff",
        f"reprojection_data/{frame}.json",
        "a00_protocol.json",
    ]
    hashes = {}
    for name in paths:
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"immutable synthetic output")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "sample_id": sample,
        "data_role": "final_train",
        "preprocessing": {
            "code_sha256": POSITIVE_GT_PROCESSOR_PAIR[0],
            "batch_code_sha256": POSITIVE_GT_PROCESSOR_PAIR[1],
        },
        "metric_validation": {"valid_pixel_count": 12},
        "output_files_sha256": hashes,
    }
    provenance_path = data / "a00_provenance" / f"{frame}.json"
    provenance_path.parent.mkdir()
    provenance_path.write_text(json.dumps(provenance))
    source = tmp_path / "raw-source"
    source.write_bytes(b"raw data unchanged")
    stat = source.stat()
    manifest = {
        "data_role": "final_train",
        "role_contract_hash": "roles",
        "processor_sha256": POSITIVE_GT_PROCESSOR_PAIR[0],
        "batch_processor_sha256": POSITIVE_GT_PROCESSOR_PAIR[1],
        "config_sha256": "config",
        "calibration_value_source_sha256": "calibration",
        "source_file_stats": {
            "rgb_video": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        },
        "sample_ids": [sample],
        "sample_metric_validation": [{"sample_id": sample, "valid_pixel_count": 12}],
    }
    manifest_path = data / "a00_keyframe_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    args = dict(
        final_keyframe=keyframe,
        source_paths={"rgb_video": source},
        role="final_train",
        role_hash="roles",
        processor_sha256=CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR[0],
        batch_code_sha256=CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR[1],
        config_sha256="config",
        calibration_sha256="calibration",
    )
    return args, manifest_path, provenance_path


def test_reviewed_resume_rehashes_outputs_without_rewriting_provenance(tmp_path: Path) -> None:
    args, manifest_path, provenance_path = _resume_fixture(tmp_path)
    before = (manifest_path.read_bytes(), provenance_path.read_bytes())
    assert _resume_keyframe_if_complete(**args) == 1
    assert before == (manifest_path.read_bytes(), provenance_path.read_bytes())
    (manifest_path.parent / "disparity/frame_data000000.tiff").write_bytes(b"corrupt")
    with pytest.raises(ScaredA00PreprocessError, match="output hash mismatch"):
        _resume_keyframe_if_complete(**args)


def test_typing_format_resume_preserves_historical_provenance(tmp_path: Path) -> None:
    args, manifest_path, provenance_path = _resume_fixture(tmp_path)
    args["processor_sha256"], args["batch_code_sha256"] = TYPED_FORMATTED_PROCESSOR_PAIR
    before = (manifest_path.read_bytes(), provenance_path.read_bytes())
    assert _resume_keyframe_if_complete(**args) == 1
    assert before == (manifest_path.read_bytes(), provenance_path.read_bytes())


def test_correspondence_review_preserves_empty_gt_but_rejects_unknown_identity() -> None:
    empty = {
        "valid_pixel_count": 0,
        "status": "unavailable",
        "q_vs_a00_abs_error_mm": None,
        "q_vs_metric_gt_abs_error_mm": None,
    }
    assert (
        a00_processor_compatibility(
            EMPTY_GT_PROCESSOR_PAIR, CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR, [empty]
        )
        == "reviewed_calibration_evidence_compatibility"
    )
    for invalid in (
        {"valid_pixel_count": 0},
        dict(empty, valid_pixel_count=True),
        dict(empty, q_vs_a00_abs_error_mm={"max": 0}),
    ):
        assert (
            a00_processor_compatibility(
                EMPTY_GT_PROCESSOR_PAIR, CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR, [invalid]
            )
            is None
        )
    assert (
        a00_processor_compatibility(
            ("a" * 64, "b" * 64), CALIBRATION_CORRESPONDENCE_PROCESSOR_PAIR, [empty]
        )
        is None
    )


def test_resume_rejects_sample_metric_or_processor_mismatch(tmp_path: Path) -> None:
    args, _manifest_path, provenance_path = _resume_fixture(tmp_path)
    provenance = json.loads(provenance_path.read_text())
    provenance["metric_validation"]["valid_pixel_count"] = 0
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ScaredA00PreprocessError, match="processor/metric identity mismatch"):
        _resume_keyframe_if_complete(**args)


def test_resume_rejects_legacy_empty_targets_and_unknown_code(tmp_path: Path) -> None:
    args, manifest_path, _provenance_path = _resume_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["sample_metric_validation"][0]["valid_pixel_count"] = 0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ScaredA00PreprocessError, match="unreviewed code identity"):
        _resume_keyframe_if_complete(**args)
    manifest["sample_metric_validation"][0]["valid_pixel_count"] = 12
    manifest["processor_sha256"] = "a" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ScaredA00PreprocessError, match="unreviewed code identity"):
        _resume_keyframe_if_complete(**args)


def test_synthetic_batch_retains_empty_gt_and_resumes_all_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise archive, evidence, geometry, file, manifest and loader boundaries.

    Only video decoding is replaced by deterministic synthetic uint8 arrays.
    """
    raw, output = tmp_path / "raw", tmp_path / "processed"
    config = yaml.safe_load(Path("configs/data/a00_scared_sample_v1.yaml").read_text())
    config["input"].update(frame_height=16, frame_width=20)
    config_path = tmp_path / "a00.yaml"
    config_path.write_text(yaml.safe_dump(config))
    calibration = {
        "KL": [[80.0, 0.0, 9.5], [0.0, 80.0, 7.5], [0.0, 0.0, 1.0]],
        "KR": [[80.0, 0.0, 9.5], [0.0, 80.0, 7.5], [0.0, 0.0, 1.0]],
        "DL": [0.0] * 5,
        "DR": [0.0] * 5,
        "R": np.eye(3).tolist(),
        "T": [-10.0, 0.0, 0.0],
    }
    for fold in range(1, 10):
        keyframe = raw / f"dataset_{fold}/keyframe_1"
        data = keyframe / "data"
        data.mkdir(parents=True)
        (data / "rgb.mp4").write_bytes(b"synthetic video decoder fixture")
        storage = cv2.FileStorage(
            str(keyframe / "endoscope_calibration.yaml"), cv2.FILE_STORAGE_WRITE
        )
        storage.write("R", np.eye(3))
        storage.write("T", np.array([[-10.0], [0.0], [0.0]]))
        storage.release()
        with (
            tarfile.open(data / "frame_data.tar.gz", "w:gz") as frames,
            tarfile.open(data / "scene_points.tar.gz", "w:gz") as scenes,
        ):
            for index in range(2):
                payload = json.dumps({"camera-calibration": calibration}).encode()
                member = tarfile.TarInfo(f"frame_data{index:06d}.json")
                member.size = len(payload)
                frames.addfile(member, io.BytesIO(payload))
                points = np.zeros((32, 20, 3), dtype=np.float32)
                for point_index in range(4):
                    points[0, point_index] = [
                        (0.0 if index == 0 else 1000.0) + point_index * 10.0,
                        0.0,
                        200.0,
                    ]
                    points[16, point_index] = points[0, point_index] + [-10.0, 0.0, 0.0]
                buffer = io.BytesIO()
                tifffile.imwrite(buffer, points, photometric="rgb")
                payload = buffer.getvalue()
                member = tarfile.TarInfo(f"scene_points{index:06d}.tiff")
                member.size = len(payload)
                scenes.addfile(member, io.BytesIO(payload))

    def synthetic_video_samples(video_path: Path, frame_archive: Path, scene_archive: Path):
        for record in _paired_archive_payloads(frame_archive, scene_archive):
            yield (*record, np.full((32, 20, 3), 128, dtype=np.uint8))

    monkeypatch.setattr(
        "reliable_endo_gs.data.scared_a00_batch._video_archive_samples", synthetic_video_samples
    )
    events = []
    result = prepare_scared_a00_dataset(
        data_root=raw, output_root=output, config_path=config_path, progress_callback=events.append
    )
    assert result["sample_count"] == 18
    assert result["samples_without_gt"] == 9
    assert sum(e["event"] == "sample_without_gt" for e in events) == 9
    manifest = load_a01_manifest(output / "a01_manifest.json")
    assert len(manifest.entries) == 18
    sample_id = "dataset_1/keyframe_1/frame_data000001"
    sample = ScaredA00Adapter(output, manifest).load(sample_id, include_supervision=True)
    assert sample.supervision is not None and not sample.supervision.valid_mask.any()
    assert sample.batch.support.all()
    assert (
        ScaredA00Adapter(output, manifest).load(sample_id, include_supervision=False).supervision
        is None
    )
    preflight = run_a00_preflight(
        output,
        sample_id,
        roles_path=Path("configs/teachers/a01_scared.yaml"),
        config_path=next((output / "_provenance/configs").glob("*.yaml")),
        print_smoke_command=False,
    )
    assert preflight["metric_gt_status"] == "unavailable"
    assert preflight["metric_gt_valid_pixel_count"] == 0
    hashes = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("a00_keyframe_manifest.json")
    }
    resumed = prepare_scared_a00_dataset(data_root=raw, output_root=output, config_path=config_path)
    assert resumed["resumed_sample_count"] == 18
    assert resumed["samples_without_gt"] == 9
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in hashes.items())
