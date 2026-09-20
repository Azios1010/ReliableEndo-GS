"""Deterministic Stage1 full-data selector and split checks."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliable_endo_gs.data import scared_c_stage1_cache as stage1_cache
from reliable_endo_gs.data.scared_c import (
    CORRECTED_VIDEO_FRAME,
    CORRECTED_VIDEO_MODE,
    FinalDatasetGuardError,
    ScaredCSampleRecord,
    build_scared_c_index,
)
from reliable_endo_gs.data.scared_c_stage1_split import (
    STAGE1_TRAIN_DATASET_IDS,
    Stage1FrameSplitError,
    load_stage1_frame_split,
    make_stage1_frame_split,
    select_stage1_records,
    write_stage1_frame_split,
)


def _record(dataset: int, keyframe: int, frame: int, *, final: bool = False) -> ScaredCSampleRecord:
    sequence_id = f"dataset_{dataset}"
    keyframe_id = f"keyframe_{keyframe}"
    return ScaredCSampleRecord(
        dataset_id="scared_c",
        sequence_id=sequence_id,
        keyframe_id=keyframe_id,
        frame_id=frame,
        sample_id=f"scared_c/{sequence_id}/{keyframe_id}/{frame}",
        source_type=CORRECTED_VIDEO_FRAME,
        mode=CORRECTED_VIDEO_MODE,
        keyframe_root=Path("unused") / sequence_id / keyframe_id,
        role=None,
        final_role=final,
        endoscope_calibration_path=Path("calibration.yaml"),
        colmap_intrinsics_path=None,
    )


def _records() -> tuple[ScaredCSampleRecord, ...]:
    return tuple(
        _record(dataset, keyframe, frame)
        for dataset, keyframe, count in ((1, 1, 10), (2, 2, 5), (3, 1, 4))
        for frame in range(1, count + 1)
    )


def test_full_split_uses_disjoint_contiguous_tail_blocks_and_round_trips(tmp_path: Path) -> None:
    split = make_stage1_frame_split(
        _records(),
        dataset_ids=STAGE1_TRAIN_DATASET_IDS,
        provenance={"test": "deterministic"},
    )

    assert split.train_count == 16
    assert split.validation_count == 3
    assert set(split.train_sample_ids).isdisjoint(split.validation_sample_ids)
    assert split.validation_frame_counts == {"1_1": 1, "2_2": 1, "3_1": 1}
    assert split.validation_sample_ids == (
        "scared_c/dataset_1/keyframe_1/10",
        "scared_c/dataset_2/keyframe_2/5",
        "scared_c/dataset_3/keyframe_1/4",
    )

    manifest_path = write_stage1_frame_split(tmp_path / "split.json", split)
    loaded = load_stage1_frame_split(manifest_path)
    assert loaded.manifest_sha256 == split.manifest_sha256
    assert loaded.train_sample_ids == split.train_sample_ids
    assert loaded.validation_sample_ids == split.validation_sample_ids


def test_selector_supports_dataset_keyframe_and_exact_sample_modes() -> None:
    records = _records()
    assert len(select_stage1_records(records, dataset_ids=("dataset_1",))) == 10
    assert len(select_stage1_records(records, keyframe_entries=("dataset_2/keyframe_2",))) == 5
    exact = "scared_c/dataset_3/keyframe_1/2"
    assert select_stage1_records(records, sample_ids=(exact,))[0].sample_id == exact


def test_selector_rejects_dataset6_and_mixed_selector_kinds() -> None:
    records = _records()
    with pytest.raises(Stage1FrameSplitError):
        select_stage1_records(records, dataset_ids=("dataset_6",))
    with pytest.raises(Stage1FrameSplitError):
        select_stage1_records(
            records, dataset_ids=("dataset_1",), sample_ids=(records[0].sample_id,)
        )


def test_stage1_index_allowlist_does_not_iterate_dataset6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for dataset in (1, 2, 3, 6):
        (tmp_path / f"dataset_{dataset}").mkdir()
    original_iterdir = Path.iterdir

    def guarded_iterdir(path: Path):
        if path.name == "dataset_6":
            raise AssertionError("dataset6 was iterated")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    index = build_scared_c_index(
        tmp_path,
        mode=CORRECTED_VIDEO_MODE,
        strict=False,
        archive_validation="lazy",
        dataset_ids=STAGE1_TRAIN_DATASET_IDS,
    )
    assert index.dataset_selection == STAGE1_TRAIN_DATASET_IDS
    assert "dataset_6" not in index.sequence_ids


def test_explicit_dataset6_index_selection_is_guarded(tmp_path: Path) -> None:
    with pytest.raises(FinalDatasetGuardError):
        build_scared_c_index(
            tmp_path,
            mode=CORRECTED_VIDEO_MODE,
            strict=False,
            dataset_ids=("dataset_6",),
        )


def test_sample_id_cache_selection_keeps_the_index_allowlisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = replace(
        _record(3, 1, 2),
        scene_points_archive_path=Path("scene_points.tar.gz"),
        scene_points_member="scene_points/000002.tiff",
    )
    calls: dict[str, object] = {}

    def fake_index(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.update(kwargs)
        return SimpleNamespace(records=(record,))

    monkeypatch.setattr(stage1_cache, "build_scared_c_index", fake_index)
    selected = stage1_cache.stage1_cache_records(
        Path("unused"),
        split_path=Path("split.json"),
        sample_ids=(record.sample_id,),
    )

    assert selected == (record,)
    assert calls["dataset_ids"] == ("dataset_3",)
    assert calls["keyframe_entries"] is None


def test_cache_accepts_legacy_file_hash_for_the_same_persisted_split(tmp_path: Path) -> None:
    records = (_record(1, 1, 1), _record(1, 1, 2))
    split = make_stage1_frame_split(
        records,
        dataset_ids=STAGE1_TRAIN_DATASET_IDS,
        provenance={"test": "cache-provenance"},
    )
    split_path = write_stage1_frame_split(tmp_path / "split.json", split)

    entries = [
        {
            "sample_id": record.sample_id,
            "sequence_key": record.sequence_key,
            "dataset_id": record.dataset_id,
            "keyframe_id": record.keyframe_id,
            "frame_id": record.frame_id,
            "file": f"gt/{record.sequence_key}/{record.frame_id:06d}.npz",
            "shape": [1, 1],
            "disparity_dtype": "float32",
            "mask_dtype": "bool",
            "rectification_identity": "identity",
            "disp_const": 1.0,
        }
        for record in records
    ]
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache_manifest = {
        "schema_version": stage1_cache.STAGE1_CACHE_SCHEMA,
        "dataset": "scared_c",
        "dataset_revision": "44baac1187c8729c96db0d1def569bd94c9d9417",
        "mode": CORRECTED_VIDEO_MODE,
        "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "dataset_allowlist": list(STAGE1_TRAIN_DATASET_IDS),
        "entry_count": len(entries),
        "entries": entries,
    }
    (cache_root / "manifest.json").write_text(json.dumps(cache_manifest), encoding="utf-8")

    cache = stage1_cache.Stage1GTCache(
        cache_root,
        expected_sample_ids=split.source_sample_ids,
        expected_split_manifest_sha256=split.manifest_sha256,
        expected_split_manifest_path=split_path,
    )

    assert cache.split_manifest_sha256 == split.manifest_sha256
