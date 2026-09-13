"""Deterministic derived disparity cache for Stage1 SCARED-C training.

The authoritative source remains the canonical corrected-video loader and XYZ
geometry implementation.  This module only changes access: it makes one
sequential pass over each selected ``scene_points.tar.gz`` and stores the
resulting rectified disparity and validity mask outside the repository.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reliable_endo_gs.data.scared_c import (
    CORRECTED_VIDEO_MODE,
    SCARED_C_UPSTREAM_REVISION,
    ScaredCSampleRecord,
    build_scared_c_index,
    load_scared_c_role_manifest,
)
from reliable_endo_gs.data.scared_c_stage1_split import (
    STAGE1_TRAIN_DATASET_IDS,
    Stage1FrameSplitError,
    load_stage1_frame_split,
    normalize_dataset_ids,
    normalize_keyframe_entries,
    normalize_sample_ids,
    select_stage1_records,
)
from reliable_endo_gs.data.scared_c_stereo import (
    RectificationCache,
    StackedStereoVideo,
    build_rectified_geometry,
    load_stereo_calibration,
    read_xyz_tar_member,
)

STAGE1_CACHE_SCHEMA = 2
STAGE1_LEGACY_CACHE_SCHEMA = 1
STAGE1_CACHE_SPLIT_NAME = "stage1_mean_v1"
STAGE1_MEAN_TRAIN = ("1_1", "2_2", "3_2")
STAGE1_MEAN_VALIDATION = ("1_2", "3_1")
STAGE1_CACHE_SEQUENCES = STAGE1_MEAN_TRAIN + STAGE1_MEAN_VALIDATION
EXPECTED_STAGE1_FRAME_COUNTS = {
    "1_1": 197,
    "2_2": 1033,
    "3_2": 1597,
    "1_2": 280,
    "3_1": 329,
}


class Stage1CacheError(ValueError):
    """Raised when a derived Stage1 cache is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class Stage1CacheEntry:
    """Manifest metadata for one cached corrected-video sample."""

    sample_id: str
    sequence_key: str
    dataset_id: str
    keyframe_id: str
    frame_id: int
    relative_path: str
    shape: tuple[int, int]
    disparity_dtype: str
    mask_dtype: str
    rectification_identity: str
    disp_const: float


def _normalised_member_name(name: str) -> str:
    return name.replace("\\", "/")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise Stage1CacheError(f"unable to hash provenance file {path}: {error}") from error
    return digest.hexdigest()


def _split_manifest_sha256(path: Path) -> str:
    """Return the canonical hash stored by a persisted Stage1 frame split."""

    try:
        return load_stage1_frame_split(path).manifest_sha256
    except (OSError, Stage1FrameSplitError) as error:
        raise Stage1CacheError(
            f"unable to load Stage1 frame split provenance {path}: {error}"
        ) from error


def _matches_split_manifest(
    recorded_hash: object,
    expected_hash: str,
    *,
    manifest_path: Path | None,
) -> bool:
    """Accept the old file-byte hash only for this exact persisted split file."""

    if recorded_hash == expected_hash:
        return True
    return (
        manifest_path is not None
        and recorded_hash == _file_sha256(manifest_path)
        and _split_manifest_sha256(manifest_path) == expected_hash
    )


def _array_digest(digest: hashlib._Hash, name: str, array: np.ndarray) -> None:
    digest.update(name.encode("utf-8"))
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())


def _rectification_identity(
    record: ScaredCSampleRecord,
    calibration: Any,
    rectification: Any,
) -> str:
    """Hash camera-file bytes and the matrices defining the rectified camera."""

    digest = hashlib.sha256()
    digest.update(record.endoscope_calibration_path.read_bytes())
    if record.colmap_intrinsics_path is not None and record.colmap_intrinsics_path.is_file():
        digest.update(record.colmap_intrinsics_path.read_bytes())
    for name, array in (
        ("M1", calibration.M1),
        ("D1", calibration.D1),
        ("M2", calibration.M2),
        ("D2", calibration.D2),
        ("R", calibration.R),
        ("T", calibration.T),
        ("R1", rectification.R1),
        ("R2", rectification.R2),
        ("P1", rectification.P1),
        ("P2", rectification.P2),
        ("Q", rectification.Q),
    ):
        _array_digest(digest, name, array)
    return digest.hexdigest()


def _cache_path(cache_root: Path, sequence_key: str, frame_id: int) -> Path:
    return cache_root / "gt" / sequence_key / f"{frame_id:06d}.npz"


def _write_npz_atomic(
    path: Path,
    *,
    disparity: np.ndarray,
    valid_mask: np.ndarray,
    disp_const: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                disparity=np.asarray(disparity, dtype=np.float32),
                valid_mask=np.asarray(valid_mask, dtype=bool),
                disp_const=np.asarray(disp_const, dtype=np.float32),
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _manifest_entry_to_dict(entry: Stage1CacheEntry) -> dict[str, object]:
    return {
        "sample_id": entry.sample_id,
        "sequence_key": entry.sequence_key,
        "dataset_id": entry.dataset_id,
        "keyframe_id": entry.keyframe_id,
        "frame_id": entry.frame_id,
        "file": entry.relative_path,
        "shape": list(entry.shape),
        "disparity_dtype": entry.disparity_dtype,
        "mask_dtype": entry.mask_dtype,
        "rectification_identity": entry.rectification_identity,
        "disp_const": entry.disp_const,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_target_records(
    records: Sequence[ScaredCSampleRecord],
    *,
    expected_sample_ids: Sequence[str] | None = None,
    expected_frame_counts: Mapping[str, int] | None = None,
    legacy_fixed_counts: bool = False,
) -> None:
    observed: dict[str, int] = defaultdict(int)
    seen_sample_ids: set[str] = set()
    for record in records:
        if record.mode != CORRECTED_VIDEO_MODE:
            raise Stage1CacheError("Stage1 cache accepts corrected-video records only")
        if record.sequence_id not in STAGE1_TRAIN_DATASET_IDS:
            raise Stage1CacheError(
                f"cache target {record.sample_id!r} is outside the Stage1 dataset allowlist"
            )
        if record.final_role or record.sequence_id in {"dataset_6", "dataset_7"}:
            raise Stage1CacheError(
                "dataset_6/dataset_7/final-role content is forbidden in the Stage1 cache"
            )
        if record.scene_points_archive_path is None or record.scene_points_member is None:
            raise Stage1CacheError(f"record {record.sample_id} has no scene-points pointer")
        if record.sample_id in seen_sample_ids:
            raise Stage1CacheError(f"duplicate target sample ID: {record.sample_id}")
        seen_sample_ids.add(record.sample_id)
        observed[record.sequence_key] += 1
    if expected_sample_ids is not None:
        expected = set(expected_sample_ids)
        if seen_sample_ids != expected:
            missing = sorted(expected - seen_sample_ids)
            extra = sorted(seen_sample_ids - expected)
            raise Stage1CacheError(
                f"cache target IDs differ from the split; missing={missing[:3]} extra={extra[:3]}"
            )
    expected_counts = (
        EXPECTED_STAGE1_FRAME_COUNTS if legacy_fixed_counts else (expected_frame_counts or {})
    )
    if expected_frame_counts is not None:
        unknown = set(observed) - set(expected_frame_counts)
        if unknown:
            raise Stage1CacheError(f"cache contains unexpected keyframes: {sorted(unknown)}")
    for sequence_key, expected in expected_counts.items():
        if observed[sequence_key] != expected:
            raise Stage1CacheError(
                f"unexpected frame count for {sequence_key}: {observed[sequence_key]} != {expected}"
            )


def build_stage1_gt_cache(
    dataset_root: str | Path,
    cache_root: str | Path,
    *,
    split_path: Path,
    source_code_sha: str,
    sample_ids: Sequence[str] | None = None,
    dataset_ids: Sequence[str] | None = None,
    keyframe_entries: Sequence[str] | None = None,
    split_manifest_path: Path | None = None,
    expected_frame_counts: Mapping[str, int] | None = None,
) -> Path:
    """Build a derived corrected-video GT cache for one explicit selection.

    The legacy call with no selector retains the original five-sequence cache
    contract.  New callers pass exact sample IDs (or an explicit dataset or
    keyframe selector); those paths are indexed without scanning any other
    dataset directory.
    """

    root = Path(dataset_root)
    output_root = Path(cache_root)
    role_manifest = load_scared_c_role_manifest(split_path)
    if role_manifest.active:
        raise Stage1CacheError("Stage1 remediation cache requires an inactive split manifest")
    legacy_selection = sample_ids is None and dataset_ids is None and keyframe_entries is None
    if (
        not legacy_selection
        and sum(
            value is not None and len(value) > 0
            for value in (sample_ids, dataset_ids, keyframe_entries)
        )
        != 1
    ):
        raise Stage1CacheError(
            "exactly one non-empty cache selector is required: sample_ids, dataset_ids, or keyframe_entries"
        )
    indexed_dataset_ids: tuple[str, ...] | None = None
    indexed_keyframe_entries: Sequence[str] | None = keyframe_entries
    if legacy_selection:
        indexed_keyframe_entries = tuple(
            f"dataset_{sequence.split('_', maxsplit=1)[0]}/"
            f"keyframe_{sequence.split('_', maxsplit=1)[1]}"
            for sequence in STAGE1_CACHE_SEQUENCES
        )
    if sample_ids is not None:
        try:
            sample_ids = normalize_sample_ids(tuple(sample_ids))
        except Stage1FrameSplitError as error:
            raise Stage1CacheError(str(error)) from error
        sample_dataset_names = {
            sample_id.split("/", maxsplit=3)[1]
            for sample_id in sample_ids
            if sample_id.count("/") >= 3
        }
        if not sample_dataset_names:
            raise Stage1CacheError("sample_ids contains no valid dataset component")
        if not sample_dataset_names <= set(STAGE1_TRAIN_DATASET_IDS):
            raise Stage1CacheError("sample_ids contains dataset7/final or non-Stage1 content")
        indexed_dataset_ids = tuple(sorted(sample_dataset_names))
    elif dataset_ids is not None:
        try:
            dataset_ids = normalize_dataset_ids(tuple(dataset_ids))
        except Stage1FrameSplitError as error:
            raise Stage1CacheError(str(error)) from error
        if not set(dataset_ids) <= set(STAGE1_TRAIN_DATASET_IDS):
            raise Stage1CacheError("dataset_ids contains dataset7/final or non-Stage1 content")
        indexed_dataset_ids = dataset_ids
    elif keyframe_entries is not None:
        try:
            keyframe_entries = normalize_keyframe_entries(tuple(keyframe_entries))
        except Stage1FrameSplitError as error:
            raise Stage1CacheError(str(error)) from error
        if not all(
            entry.split("/", maxsplit=1)[0] in STAGE1_TRAIN_DATASET_IDS
            for entry in keyframe_entries
        ):
            raise Stage1CacheError("keyframe_entries contains dataset7/final or non-Stage1 content")
        indexed_keyframe_entries = keyframe_entries
    index = build_scared_c_index(
        root,
        mode=CORRECTED_VIDEO_MODE,
        manifest_path=split_path,
        strict=True,
        archive_validation="lazy",
        dataset_ids=indexed_dataset_ids,
        keyframe_entries=indexed_keyframe_entries,
    )
    if legacy_selection:
        target_records = tuple(
            record for record in index.records if record.sequence_key in STAGE1_CACHE_SEQUENCES
        )
        _validate_target_records(target_records, legacy_fixed_counts=True)
    else:
        target_records = select_stage1_records(
            index.records,
            dataset_ids=dataset_ids,
            keyframe_entries=keyframe_entries,
            sample_ids=sample_ids,
        )
        _validate_target_records(
            target_records,
            expected_sample_ids=sample_ids,
            expected_frame_counts=expected_frame_counts,
        )

    provenance_manifest = split_path if split_manifest_path is None else Path(split_manifest_path)
    split_manifest_sha256 = (
        _split_manifest_sha256(provenance_manifest)
        if split_manifest_path is not None
        else _file_sha256(provenance_manifest)
    )
    target_sample_ids = tuple(sorted(record.sample_id for record in target_records))
    cache_manifest_path = output_root / "manifest.json"
    if cache_manifest_path.exists():
        try:
            existing_payload = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Stage1CacheError(
                f"refusing to overwrite unreadable cache manifest {cache_manifest_path}: {error}"
            ) from error
        existing_manifest = existing_payload if isinstance(existing_payload, dict) else {}
        existing_entries = existing_manifest.get("entries")
        existing_ids = (
            tuple(
                sorted(
                    str(entry.get("sample_id"))
                    for entry in existing_entries
                    if isinstance(entry, dict)
                )
            )
            if isinstance(existing_entries, list)
            else ()
        )
        if (
            existing_manifest.get("dataset_revision") == SCARED_C_UPSTREAM_REVISION
            and _matches_split_manifest(
                existing_manifest.get("split_manifest_sha256"),
                split_manifest_sha256,
                manifest_path=provenance_manifest if split_manifest_path is not None else None,
            )
            and existing_ids == target_sample_ids
        ):
            return cache_manifest_path
        raise Stage1CacheError(
            f"refusing to overwrite a different Stage1 GT cache: {cache_manifest_path}"
        )

    grouped: dict[Path, list[ScaredCSampleRecord]] = defaultdict(list)
    for record in target_records:
        assert record.scene_points_archive_path is not None
        grouped[record.scene_points_archive_path].append(record)

    rectification_cache = RectificationCache()
    entries: list[Stage1CacheEntry] = []
    seen_sample_ids: set[str] = set()
    for archive_path in sorted(grouped, key=lambda path: path.as_posix()):
        archive_records = sorted(grouped[archive_path], key=lambda record: int(record.frame_id))
        first = archive_records[0]
        assert first.video_path is not None
        with StackedStereoVideo(first.video_path) as video:
            metadata = video.metadata()
        image_size = (metadata.width, metadata.height // 2)
        calibration = load_stereo_calibration(
            first.endoscope_calibration_path,
            image_size=image_size,
            colmap_intrinsics_path=first.colmap_intrinsics_path,
        )
        rectification = rectification_cache.get_or_create(first.keyframe_root, calibration)
        identity = _rectification_identity(first, calibration, rectification)
        exact_names = {
            _normalised_member_name(record.scene_points_member): record
            for record in archive_records
            if record.scene_points_member is not None
        }
        basename_names: dict[str, ScaredCSampleRecord] = {}
        for record in archive_records:
            assert record.scene_points_member is not None
            basename = Path(_normalised_member_name(record.scene_points_member)).name
            if basename in basename_names:
                raise Stage1CacheError(f"ambiguous XYZ member basename {basename}")
            basename_names[basename] = record
        found: set[str] = set()
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                normal_name = _normalised_member_name(member.name)
                record = exact_names.get(normal_name) or basename_names.get(Path(normal_name).name)
                if record is None:
                    continue
                if record.sample_id in seen_sample_ids:
                    raise Stage1CacheError(f"duplicate cache sample {record.sample_id}")
                xyz_left = read_xyz_tar_member(archive, member.name)
                geometry = build_rectified_geometry(xyz_left, calibration, rectification)
                if geometry.disparity_left_rect.shape != (image_size[1], image_size[0]):
                    raise Stage1CacheError(
                        f"unexpected disparity shape for {record.sample_id}: "
                        f"{geometry.disparity_left_rect.shape}"
                    )
                relative_path = (
                    _cache_path(output_root, record.sequence_key, int(record.frame_id))
                    .relative_to(output_root)
                    .as_posix()
                )
                _write_npz_atomic(
                    output_root / relative_path,
                    disparity=geometry.disparity_left_rect,
                    valid_mask=geometry.valid_disparity_mask,
                    disp_const=float(rectification.Q[2, 3] / rectification.Q[3, 2]),
                )
                entries.append(
                    Stage1CacheEntry(
                        sample_id=record.sample_id,
                        sequence_key=record.sequence_key,
                        dataset_id=record.dataset_id,
                        keyframe_id=record.keyframe_id,
                        frame_id=int(record.frame_id),
                        relative_path=relative_path,
                        shape=tuple(int(value) for value in geometry.disparity_left_rect.shape),
                        disparity_dtype="float32",
                        mask_dtype="bool",
                        rectification_identity=identity,
                        disp_const=float(rectification.Q[2, 3] / rectification.Q[3, 2]),
                    )
                )
                seen_sample_ids.add(record.sample_id)
                found.add(record.sample_id)
        missing = {record.sample_id for record in archive_records} - found
        if missing:
            raise Stage1CacheError(
                f"archive {archive_path} did not contain requested members: {sorted(missing)[:3]}"
            )

    if len(entries) != len(target_records):
        raise Stage1CacheError(
            f"cache entry count {len(entries)} != target count {len(target_records)}"
        )
    entries.sort(key=lambda entry: entry.sample_id)
    observed_frame_counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        observed_frame_counts[entry.sequence_key] += 1
    if legacy_selection:
        selection_payload: dict[str, object] = {
            "kind": "keyframe_entries",
            "keyframe_entries": [
                f"dataset_{sequence.split('_', maxsplit=1)[0]}/"
                f"keyframe_{sequence.split('_', maxsplit=1)[1]}"
                for sequence in STAGE1_CACHE_SEQUENCES
            ],
        }
    elif sample_ids is not None:
        selection_payload = {"kind": "sample_ids", "sample_ids": list(sample_ids)}
    elif dataset_ids is not None:
        selection_payload = {"kind": "dataset_ids", "dataset_ids": list(dataset_ids)}
    else:
        selection_payload = {
            "kind": "keyframe_entries",
            "keyframe_entries": list(keyframe_entries or ()),
        }
    manifest: dict[str, object] = {
        "schema_version": STAGE1_CACHE_SCHEMA,
        "dataset": "scared_c",
        "dataset_revision": SCARED_C_UPSTREAM_REVISION,
        "source_code_sha": source_code_sha,
        "split_name": STAGE1_CACHE_SPLIT_NAME,
        "split_manifest": str(split_path),
        "split_manifest_sha256": split_manifest_sha256,
        "mode": CORRECTED_VIDEO_MODE,
        "selection": selection_payload,
        "dataset_allowlist": list(STAGE1_TRAIN_DATASET_IDS),
        "frame_counts": dict(sorted(observed_frame_counts.items())),
        "entry_count": len(entries),
        "dataset6_content_accessed": False,
        "dataset7_content_accessed": False,
        "provenance": {
            "role_manifest": str(split_path),
            "selection_manifest": str(provenance_manifest),
            "selection_manifest_sha256": split_manifest_sha256,
            "source_code_sha": source_code_sha,
        },
        "gt_contract": {
            "disparity_convention": "x_left - x_right",
            "sign": "positive",
            "unit": "pixels",
            "disparity_dtype": "float32",
            "valid_mask_dtype": "bool",
            "invalid_disparity_storage": "NaN; mask is authoritative",
            "camera_rectification": "canonical SCARED-C loader calibration and OpenCV stereoRectify",
        },
        "entries": [_manifest_entry_to_dict(entry) for entry in entries],
    }
    _write_json_atomic(output_root / "manifest.json", manifest)
    return output_root / "manifest.json"


def _parse_entry(raw: Mapping[str, object]) -> Stage1CacheEntry:
    try:
        shape_raw = raw["shape"]
        if not isinstance(shape_raw, list) or len(shape_raw) != 2:
            raise TypeError("shape must be a two-item list")
        return Stage1CacheEntry(
            sample_id=str(raw["sample_id"]),
            sequence_key=str(raw["sequence_key"]),
            dataset_id=str(raw["dataset_id"]),
            keyframe_id=str(raw["keyframe_id"]),
            frame_id=int(raw["frame_id"]),
            relative_path=str(raw["file"]),
            shape=(int(shape_raw[0]), int(shape_raw[1])),
            disparity_dtype=str(raw["disparity_dtype"]),
            mask_dtype=str(raw["mask_dtype"]),
            rectification_identity=str(raw["rectification_identity"]),
            disp_const=float(raw["disp_const"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Stage1CacheError(f"invalid Stage1 cache manifest entry: {raw!r}") from error


class Stage1GTCache:
    """Read-only access to a validated derived Stage1 disparity cache."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        expected_sample_ids: Sequence[str] | None = None,
        expected_split_manifest_sha256: str | None = None,
        expected_split_manifest_path: str | Path | None = None,
    ) -> None:
        self.root = Path(cache_root)
        try:
            payload = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Stage1CacheError(f"unable to read Stage1 GT cache manifest: {error}") from error
        if not isinstance(payload, dict):
            raise Stage1CacheError("Stage1 GT cache manifest must be an object")
        if payload.get("schema_version") not in {STAGE1_LEGACY_CACHE_SCHEMA, STAGE1_CACHE_SCHEMA}:
            raise Stage1CacheError("unsupported Stage1 GT cache schema")
        if payload.get("dataset_revision") != SCARED_C_UPSTREAM_REVISION:
            raise Stage1CacheError("Stage1 GT cache has the wrong SCARED-C revision")
        if payload.get("mode") != CORRECTED_VIDEO_MODE:
            raise Stage1CacheError("Stage1 GT cache is not corrected-video data")
        raw_allowlist = payload.get("dataset_allowlist")
        if raw_allowlist is not None:
            if (
                not isinstance(raw_allowlist, list)
                or not all(isinstance(value, str) for value in raw_allowlist)
                or set(raw_allowlist) != set(STAGE1_TRAIN_DATASET_IDS)
            ):
                raise Stage1CacheError("Stage1 GT cache dataset allowlist is not dataset_1..3")
        raw_split_hash = payload.get("split_manifest_sha256")
        expected_manifest_path = (
            None if expected_split_manifest_path is None else Path(expected_split_manifest_path)
        )
        if expected_split_manifest_sha256 is not None and not _matches_split_manifest(
            raw_split_hash,
            expected_split_manifest_sha256,
            manifest_path=expected_manifest_path,
        ):
            raise Stage1CacheError(
                "Stage1 GT cache was built from a different frame split manifest"
            )
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise Stage1CacheError("Stage1 GT cache manifest entries must be a list")
        entries = tuple(_parse_entry(item) for item in raw_entries if isinstance(item, dict))
        if len(entries) != len(raw_entries):
            raise Stage1CacheError("Stage1 GT cache contains a non-object entry")
        try:
            normalized_entry_ids = normalize_sample_ids(
                tuple(entry.sample_id for entry in entries), name="cache sample_ids"
            )
        except Stage1FrameSplitError as error:
            raise Stage1CacheError(str(error)) from error
        if any(
            sample_id.split("/", maxsplit=3)[1] not in STAGE1_TRAIN_DATASET_IDS
            for sample_id in normalized_entry_ids
        ):
            raise Stage1CacheError("Stage1 GT cache contains a sample outside dataset_1..3")
        self.source_code_sha = str(payload.get("source_code_sha", ""))
        self.dataset_revision = str(payload["dataset_revision"])
        self._entries = {entry.sample_id: entry for entry in entries}
        if len(self._entries) != len(entries):
            raise Stage1CacheError("Stage1 GT cache contains duplicate sample IDs")
        if any(
            entry.dataset_id in {"dataset_6", "dataset_7"}
            or "dataset_6" in entry.sample_id
            or "dataset_7" in entry.sample_id
            for entry in entries
        ):
            raise Stage1CacheError("Stage1 GT cache contains forbidden dataset6/dataset7 content")
        if expected_sample_ids is not None:
            expected = set(expected_sample_ids)
            if set(self._entries) != expected:
                missing = sorted(expected - set(self._entries))
                extra = sorted(set(self._entries) - expected)
                raise Stage1CacheError(
                    f"Stage1 GT cache sample IDs differ from the split; "
                    f"missing={missing[:3]} extra={extra[:3]}"
                )
        self.split_manifest_sha256 = str(
            expected_split_manifest_sha256
            if expected_split_manifest_sha256 is not None
            else raw_split_hash or ""
        )
        self.selection = payload.get("selection", {})

    def __len__(self) -> int:
        return len(self._entries)

    def entry(self, sample_id: str) -> Stage1CacheEntry:
        try:
            return self._entries[sample_id]
        except KeyError as error:
            raise KeyError(f"sample is not present in Stage1 GT cache: {sample_id}") from error

    def contains(self, sample_id: str) -> bool:
        """Return whether one sample has a manifest entry."""

        return sample_id in self._entries

    def load(self, sample_id: str) -> tuple[np.ndarray, np.ndarray, float, Stage1CacheEntry]:
        """Load ``(disparity, valid_mask, disp_const, manifest_entry)``."""

        entry = self.entry(sample_id)
        path = self.root / entry.relative_path
        try:
            with np.load(path, allow_pickle=False) as payload:
                disparity = np.asarray(payload["disparity"])
                valid_mask = np.asarray(payload["valid_mask"])
                disp_const = float(np.asarray(payload["disp_const"]).item())
        except (OSError, ValueError, KeyError) as error:
            raise Stage1CacheError(f"unable to read cache item {path}: {error}") from error
        if disparity.dtype != np.float32 or valid_mask.dtype != np.bool_:
            raise Stage1CacheError(f"cache item {path} has unexpected dtypes")
        if disparity.shape != entry.shape or valid_mask.shape != entry.shape:
            raise Stage1CacheError(f"cache item {path} has unexpected shape")
        if not np.isfinite(disp_const) or disp_const <= 0:
            raise Stage1CacheError(f"cache item {path} has invalid disp_const {disp_const}")
        return disparity, valid_mask, disp_const, entry


def stage1_cache_records(
    dataset_root: str | Path,
    *,
    split_path: Path,
    sample_ids: Sequence[str] | None = None,
    dataset_ids: Sequence[str] | None = None,
    keyframe_entries: Sequence[str] | None = None,
) -> tuple[ScaredCSampleRecord, ...]:
    """Return records for one explicit Stage1 cache selector."""

    legacy_selection = sample_ids is None and dataset_ids is None and keyframe_entries is None
    if (
        not legacy_selection
        and sum(
            value is not None and len(value) > 0
            for value in (sample_ids, dataset_ids, keyframe_entries)
        )
        != 1
    ):
        raise Stage1CacheError(
            "exactly one non-empty cache selector is required: sample_ids, dataset_ids, or keyframe_entries"
        )
    if legacy_selection:
        keyframe_entries = tuple(
            f"dataset_{sequence.split('_', maxsplit=1)[0]}/"
            f"keyframe_{sequence.split('_', maxsplit=1)[1]}"
            for sequence in STAGE1_CACHE_SEQUENCES
        )
    indexed_dataset_ids: tuple[str, ...] | None = None
    if sample_ids is not None:
        try:
            sample_ids = normalize_sample_ids(tuple(sample_ids))
        except Stage1FrameSplitError as error:
            raise Stage1CacheError(str(error)) from error
        sample_dataset_ids = {sample_id.split("/", maxsplit=3)[1] for sample_id in sample_ids}
        if not sample_dataset_ids <= set(STAGE1_TRAIN_DATASET_IDS):
            raise Stage1CacheError("sample_ids contains dataset7/final or non-Stage1 content")
        indexed_dataset_ids = tuple(sorted(sample_dataset_ids))
    elif dataset_ids is not None:
        try:
            dataset_ids = normalize_dataset_ids(tuple(dataset_ids))
        except Stage1FrameSplitError as error:
            raise Stage1CacheError(str(error)) from error
        if not set(dataset_ids) <= set(STAGE1_TRAIN_DATASET_IDS):
            raise Stage1CacheError("dataset_ids contains dataset7/final or non-Stage1 content")
        indexed_dataset_ids = dataset_ids
    elif keyframe_entries is not None:
        try:
            keyframe_entries = normalize_keyframe_entries(tuple(keyframe_entries))
        except Stage1FrameSplitError as error:
            raise Stage1CacheError(str(error)) from error
        if not all(
            entry.split("/", maxsplit=1)[0] in STAGE1_TRAIN_DATASET_IDS
            for entry in keyframe_entries
        ):
            raise Stage1CacheError("keyframe_entries contains dataset7/final or non-Stage1 content")
    index = build_scared_c_index(
        Path(dataset_root),
        mode=CORRECTED_VIDEO_MODE,
        manifest_path=split_path,
        strict=True,
        archive_validation="lazy",
        dataset_ids=indexed_dataset_ids,
        keyframe_entries=keyframe_entries,
    )
    if legacy_selection:
        records = tuple(
            record for record in index.records if record.sequence_key in STAGE1_CACHE_SEQUENCES
        )
        _validate_target_records(records, legacy_fixed_counts=True)
    else:
        records = select_stage1_records(
            index.records,
            dataset_ids=dataset_ids,
            keyframe_entries=keyframe_entries,
            sample_ids=sample_ids,
        )
        _validate_target_records(records, expected_sample_ids=sample_ids)
    return records


__all__ = [
    "EXPECTED_STAGE1_FRAME_COUNTS",
    "STAGE1_CACHE_SEQUENCES",
    "STAGE1_CACHE_SPLIT_NAME",
    "STAGE1_MEAN_TRAIN",
    "STAGE1_MEAN_VALIDATION",
    "Stage1CacheEntry",
    "Stage1CacheError",
    "Stage1GTCache",
    "build_stage1_gt_cache",
    "stage1_cache_records",
]
