"""Stage1 GT exclusions preserve source partitions and fail closed on cache damage."""

import hashlib
import json
import threading
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from reliable_endo_gs.data.scared_c import (
    CORRECTED_VIDEO_FRAME,
    CORRECTED_VIDEO_MODE,
    SCARED_C_UPSTREAM_REVISION,
    ScaredCSampleRecord,
)
from reliable_endo_gs.data.scared_c_stage1_cache import STAGE1_CACHE_SCHEMA, Stage1CacheError
from reliable_endo_gs.data.scared_c_stage1_quality import (
    Stage1GTQualityError,
    audit_stage1_gt_cache,
    load_stage1_gt_quality_manifest,
    write_stage1_gt_quality_manifest,
)
from reliable_endo_gs.data.scared_c_stage1_split import (
    STAGE1_TRAIN_DATASET_IDS,
    Stage1FrameSplit,
    make_stage1_frame_split,
    write_stage1_frame_split,
)


@pytest.fixture
def cache_fixture(tmp_path: Path) -> tuple[Path, Stage1FrameSplit, Path]:
    records = tuple(
        ScaredCSampleRecord(
            dataset_id="scared_c",
            sequence_id="dataset_1",
            keyframe_id=f"keyframe_{keyframe}",
            frame_id=frame,
            sample_id=f"scared_c/dataset_1/keyframe_{keyframe}/{frame}",
            source_type=CORRECTED_VIDEO_FRAME,
            mode=CORRECTED_VIDEO_MODE,
            keyframe_root=Path("unused"),
            role=None,
            final_role=False,
            endoscope_calibration_path=Path("unused.yaml"),
            colmap_intrinsics_path=None,
        )
        for keyframe in (1, 2)
        for frame in range(1, 5)
    )
    split = make_stage1_frame_split(records, dataset_ids=STAGE1_TRAIN_DATASET_IDS)
    split_path = write_stage1_frame_split(tmp_path / "split.json", split)
    cache_root = tmp_path / "cache"
    entries = []
    for record in records:
        relative_path = f"gt/{record.sequence_key}/{record.frame_id:06d}.npz"
        path = cache_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            disparity=np.full((2, 3), 2.0, dtype=np.float32),
            valid_mask=np.ones((2, 3), dtype=np.bool_),
            disp_const=np.float32(1.0),
        )
        entries.append(
            {
                "sample_id": record.sample_id,
                "sequence_key": record.sequence_key,
                "dataset_id": record.dataset_id,
                "keyframe_id": record.keyframe_id,
                "frame_id": record.frame_id,
                "file": relative_path,
                "shape": [2, 3],
                "disparity_dtype": "float32",
                "mask_dtype": "bool",
                "rectification_identity": "identity",
                "disp_const": 1.0,
            }
        )
    payload = {
        "schema_version": STAGE1_CACHE_SCHEMA,
        "dataset": "scared_c",
        "dataset_revision": SCARED_C_UPSTREAM_REVISION,
        "mode": CORRECTED_VIDEO_MODE,
        "split_manifest_sha256": split.manifest_sha256,
        "dataset_allowlist": list(STAGE1_TRAIN_DATASET_IDS),
        "entries": entries,
    }
    (cache_root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return cache_root, split, split_path


def _rewrite_sample(cache_root: Path, keyframe: int, frame: int, **updates: object) -> Path:
    path = cache_root / f"gt/1_{keyframe}/{frame:06d}.npz"
    with np.load(path) as loaded:
        payload = dict(loaded)
    payload.update(updates)
    np.savez(path, **payload)
    return path


def test_audit_excludes_unusable_targets_from_both_original_partitions(
    cache_fixture: tuple[Path, Stage1FrameSplit, Path],
) -> None:
    cache_root, split, split_path = cache_fixture
    source_bytes = split_path.read_bytes()
    empty_train = _rewrite_sample(cache_root, 1, 1, valid_mask=np.zeros((2, 3), dtype=np.bool_))
    _rewrite_sample(cache_root, 1, 4, valid_mask=np.zeros((2, 3), dtype=np.bool_))
    cache_bytes = empty_train.read_bytes()
    invalid = np.full((2, 3), 2.0, dtype=np.float32)
    invalid[0, :] = [np.nan, np.inf, -1.0]
    _rewrite_sample(cache_root, 2, 2, disparity=invalid)
    # Nonfinite and nonpositive values outside supervision do not invalidate a frame.
    mask = np.ones((2, 3), dtype=np.bool_)
    mask[0, :] = False
    _rewrite_sample(cache_root, 2, 1, disparity=invalid, valid_mask=mask)

    audit = audit_stage1_gt_cache(cache_root, split, split_manifest_path=split_path)
    bad_ids = {
        "scared_c/dataset_1/keyframe_1/1": "zero_valid_mask",
        "scared_c/dataset_1/keyframe_1/4": "zero_valid_mask",
        "scared_c/dataset_1/keyframe_2/2": "invalid_supervised_disparity",
    }
    assert audit.excluded == bad_ids
    assert audit.train_sample_ids == tuple(s for s in split.train_sample_ids if s not in bad_ids)
    assert audit.validation_sample_ids == ("scared_c/dataset_1/keyframe_2/4",)
    assert audit.coverage_by_keyframe["dataset_1/keyframe_1"] == {
        "train_source": 3,
        "train_retained": 2,
        "train_excluded": 1,
        "validation_source": 1,
        "validation_retained": 0,
        "validation_excluded": 1,
    }
    by_id = {sample.sample_id: sample for sample in audit.samples}
    assert by_id["scared_c/dataset_1/keyframe_2/2"].invalid_supervised_disparity_count == 3
    assert by_id["scared_c/dataset_1/keyframe_2/1"].valid_pixel_count == 3
    assert by_id["scared_c/dataset_1/keyframe_2/1"].total_pixel_count == 6
    assert audit.input_split_manifest_sha256 == split.manifest_sha256
    assert (
        audit.cache_manifest_sha256
        == hashlib.sha256((cache_root / "manifest.json").read_bytes()).hexdigest()
    )
    assert split_path.read_bytes() == source_bytes
    assert empty_train.read_bytes() == cache_bytes
    with pytest.raises(FrozenInstanceError):
        audit.samples[0].valid_pixel_count = 10  # type: ignore[misc]
    with pytest.raises(TypeError):
        audit.excluded["other"] = "zero_valid_mask"  # type: ignore[index]


@pytest.mark.parametrize("value", [0.0, -2.0, np.nan, np.inf, -np.inf])
def test_any_invalid_supervised_disparity_excludes_the_whole_sample(
    cache_fixture: tuple[Path, Stage1FrameSplit, Path], value: float
) -> None:
    cache_root, split, split_path = cache_fixture
    disparity = np.full((2, 3), 2.0, dtype=np.float32)
    disparity[0, 0] = value
    _rewrite_sample(cache_root, 1, 1, disparity=disparity)
    result = audit_stage1_gt_cache(cache_root, split, split_manifest_path=split_path)
    assert result.excluded == {"scared_c/dataset_1/keyframe_1/1": "invalid_supervised_disparity"}


@pytest.mark.parametrize("partition", ["train", "validation"])
def test_audit_rejects_empty_usable_partition(
    cache_fixture: tuple[Path, Stage1FrameSplit, Path], partition: str
) -> None:
    cache_root, split, split_path = cache_fixture
    sample_ids = split.train_sample_ids if partition == "train" else split.validation_sample_ids
    for sample_id in sample_ids:
        _, _, keyframe, frame = sample_id.split("/")
        _rewrite_sample(
            cache_root,
            int(keyframe.removeprefix("keyframe_")),
            int(frame),
            valid_mask=np.zeros((2, 3), dtype=np.bool_),
        )
    with pytest.raises(Stage1GTQualityError, match=f"no usable {partition}"):
        audit_stage1_gt_cache(cache_root, split, split_manifest_path=split_path)


@pytest.mark.parametrize("damage", ["dtype", "shape", "disp_const", "missing", "unreadable"])
def test_structurally_corrupt_cache_is_an_error_instead_of_an_exclusion(
    cache_fixture: tuple[Path, Stage1FrameSplit, Path], damage: str
) -> None:
    cache_root, split, split_path = cache_fixture
    path = cache_root / "gt/1_1/000001.npz"
    if damage == "dtype":
        _rewrite_sample(cache_root, 1, 1, disparity=np.ones((2, 3), dtype=np.float64))
    elif damage == "shape":
        _rewrite_sample(cache_root, 1, 1, valid_mask=np.ones((1, 3), dtype=np.bool_))
    elif damage == "disp_const":
        _rewrite_sample(cache_root, 1, 1, disp_const=np.float32(0.0))
    elif damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"broken npz")
    with pytest.raises(Stage1CacheError):
        audit_stage1_gt_cache(cache_root, split, split_manifest_path=split_path)


def test_manifest_round_trip_is_deterministic_and_refuses_incompatible_overwrite(
    cache_fixture: tuple[Path, Stage1FrameSplit, Path], tmp_path: Path
) -> None:
    cache_root, split, split_path = cache_fixture
    audit = audit_stage1_gt_cache(cache_root, split, split_manifest_path=split_path)
    parallel = audit_stage1_gt_cache(cache_root, split, split_manifest_path=split_path, workers=2)
    assert parallel.to_payload() == audit.to_payload()
    path = write_stage1_gt_quality_manifest(tmp_path / "quality.json", audit)
    original = path.read_bytes()
    assert write_stage1_gt_quality_manifest(path, parallel) == path
    loaded = load_stage1_gt_quality_manifest(
        path, split=split, cache_root=cache_root, split_manifest_path=split_path
    )
    assert loaded == audit
    assert loaded.manifest_sha256 == audit.manifest_sha256
    _rewrite_sample(cache_root, 1, 1, valid_mask=np.zeros((2, 3), dtype=np.bool_))
    changed = audit_stage1_gt_cache(cache_root, split, split_manifest_path=split_path)
    assert changed.manifest_sha256 != audit.manifest_sha256
    with pytest.raises(Stage1GTQualityError, match="refusing to overwrite"):
        write_stage1_gt_quality_manifest(path, changed)
    assert path.read_bytes() == original


@pytest.mark.parametrize("workers", [1, 2])
def test_progress_is_ordered_and_runs_in_calling_thread(
    cache_fixture: tuple[Path, Stage1FrameSplit, Path], workers: int
) -> None:
    cache_root, split, split_path = cache_fixture
    calls: list[tuple[int, int, int]] = []
    calling_thread = threading.get_ident()

    def progress(completed: int, total: int) -> None:
        calls.append((completed, total, threading.get_ident()))

    audit_stage1_gt_cache(
        cache_root, split, split_manifest_path=split_path, workers=workers, progress=progress
    )
    total = len(split.source_sample_ids)
    assert calls == [(completed, total, calling_thread) for completed in range(1, total + 1)]


@pytest.mark.parametrize(
    "tamper", ["hash", "count", "partition", "omission", "coverage", "decision"]
)
def test_loading_rejects_tampered_manifest(
    cache_fixture: tuple[Path, Stage1FrameSplit, Path], tmp_path: Path, tamper: str
) -> None:
    cache_root, split, split_path = cache_fixture
    audit = audit_stage1_gt_cache(cache_root, split, split_manifest_path=split_path)
    payload = audit.to_payload()
    if tamper == "hash":
        payload["manifest_sha256"] = "0" * 64
    elif tamper == "count":
        payload["samples"][0]["valid_pixel_count"] = 999
    elif tamper == "partition":
        payload["samples"][0]["partition"] = "validation"
    elif tamper == "omission":
        payload["samples"].pop()
    elif tamper == "coverage":
        payload["coverage_by_keyframe"]["dataset_1/keyframe_1"]["train_source"] = 999
    else:
        payload["samples"][0]["exclusion_reason"] = "zero_valid_mask"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Stage1GTQualityError):
        load_stage1_gt_quality_manifest(
            path, split=split, cache_root=cache_root, split_manifest_path=split_path
        )


def test_loading_rejects_cache_manifest_byte_changes(
    cache_fixture: tuple[Path, Stage1FrameSplit, Path], tmp_path: Path
) -> None:
    cache_root, split, split_path = cache_fixture
    audit = audit_stage1_gt_cache(cache_root, split, split_manifest_path=split_path)
    path = write_stage1_gt_quality_manifest(tmp_path / "quality.json", audit)
    cache_manifest = cache_root / "manifest.json"
    cache_manifest.write_bytes(cache_manifest.read_bytes() + b"\n")
    with pytest.raises(Stage1GTQualityError, match="different cache manifest"):
        load_stage1_gt_quality_manifest(
            path, split=split, cache_root=cache_root, split_manifest_path=split_path
        )


def test_audit_rejects_split_provenance_mismatch(
    cache_fixture: tuple[Path, Stage1FrameSplit, Path],
) -> None:
    cache_root, split, split_path = cache_fixture
    changed = replace(split, provenance={"different": "source"})
    with pytest.raises(Stage1GTQualityError, match="persisted source split"):
        audit_stage1_gt_cache(cache_root, changed, split_manifest_path=split_path)


def test_audit_rejects_incomplete_cache_coverage(
    cache_fixture: tuple[Path, Stage1FrameSplit, Path],
) -> None:
    cache_root, split, split_path = cache_fixture
    path = cache_root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Stage1CacheError, match="sample IDs differ"):
        audit_stage1_gt_cache(cache_root, split, split_manifest_path=split_path)
