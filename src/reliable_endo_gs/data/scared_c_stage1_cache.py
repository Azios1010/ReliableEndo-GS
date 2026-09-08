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
    ScaredCError,
    ScaredCSampleRecord,
    build_scared_c_index,
    load_scared_c_role_manifest,
)
from reliable_endo_gs.data.scared_c_stereo import (
    RectificationCache,
    StackedStereoVideo,
    build_rectified_geometry,
    load_stereo_calibration,
    read_xyz_tar_member,
)

STAGE1_CACHE_SCHEMA = 1
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


def _validate_target_records(records: Sequence[ScaredCSampleRecord]) -> None:
    observed: dict[str, int] = defaultdict(int)
    for record in records:
        if record.mode != CORRECTED_VIDEO_MODE:
            raise Stage1CacheError("Stage1 cache accepts corrected-video records only")
        if record.sequence_key not in STAGE1_CACHE_SEQUENCES:
            raise Stage1CacheError(
                f"cache target {record.sequence_key!r} is outside the Stage1 train/validation roles"
            )
        if record.final_role or record.sequence_key.startswith("6_"):
            raise Stage1CacheError("dataset_6/final-role content is forbidden in the Stage1 cache")
        if record.scene_points_archive_path is None or record.scene_points_member is None:
            raise Stage1CacheError(f"record {record.sample_id} has no scene-points pointer")
        observed[record.sequence_key] += 1
    for sequence_key, expected in EXPECTED_STAGE1_FRAME_COUNTS.items():
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
) -> Path:
    """Build the complete inactive Stage1 mean train/validation GT cache.

    Only the five explicitly named data-1/2/3 sequences are materialized.  The
    index may inspect metadata for the rest of SCARED-C, but this function never
    opens an archive or decodes a payload outside those five sequences.
    """

    root = Path(dataset_root)
    output_root = Path(cache_root)
    role_manifest = load_scared_c_role_manifest(split_path)
    if role_manifest.active:
        raise Stage1CacheError("Stage1 remediation cache requires an inactive split manifest")
    index = build_scared_c_index(
        root,
        mode=CORRECTED_VIDEO_MODE,
        manifest_path=split_path,
        strict=True,
        archive_validation="lazy",
    )
    target_records = tuple(
        record for record in index.records if record.sequence_key in STAGE1_CACHE_SEQUENCES
    )
    _validate_target_records(target_records)

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
                relative_path = _cache_path(output_root, record.sequence_key, int(record.frame_id)).relative_to(
                    output_root
                ).as_posix()
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
        raise Stage1CacheError(f"cache entry count {len(entries)} != target count {len(target_records)}")
    entries.sort(key=lambda entry: entry.sample_id)
    manifest: dict[str, object] = {
        "schema_version": STAGE1_CACHE_SCHEMA,
        "dataset": "scared_c",
        "dataset_revision": SCARED_C_UPSTREAM_REVISION,
        "source_code_sha": source_code_sha,
        "split_name": STAGE1_CACHE_SPLIT_NAME,
        "split_manifest": str(split_path),
        "mode": CORRECTED_VIDEO_MODE,
        "sequences": {
            "MEAN_TRAIN": list(STAGE1_MEAN_TRAIN),
            "MEAN_VALIDATION": list(STAGE1_MEAN_VALIDATION),
        },
        "frame_counts": dict(EXPECTED_STAGE1_FRAME_COUNTS),
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

    def __init__(self, cache_root: str | Path) -> None:
        self.root = Path(cache_root)
        try:
            payload = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Stage1CacheError(f"unable to read Stage1 GT cache manifest: {error}") from error
        if not isinstance(payload, dict):
            raise Stage1CacheError("Stage1 GT cache manifest must be an object")
        if payload.get("schema_version") != STAGE1_CACHE_SCHEMA:
            raise Stage1CacheError("unsupported Stage1 GT cache schema")
        if payload.get("dataset_revision") != SCARED_C_UPSTREAM_REVISION:
            raise Stage1CacheError("Stage1 GT cache has the wrong SCARED-C revision")
        if payload.get("mode") != CORRECTED_VIDEO_MODE:
            raise Stage1CacheError("Stage1 GT cache is not corrected-video data")
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise Stage1CacheError("Stage1 GT cache manifest entries must be a list")
        entries = tuple(_parse_entry(item) for item in raw_entries if isinstance(item, dict))
        if len(entries) != len(raw_entries):
            raise Stage1CacheError("Stage1 GT cache contains a non-object entry")
        self.source_code_sha = str(payload.get("source_code_sha", ""))
        self.dataset_revision = str(payload["dataset_revision"])
        self._entries = {entry.sample_id: entry for entry in entries}
        if len(self._entries) != len(entries):
            raise Stage1CacheError("Stage1 GT cache contains duplicate sample IDs")

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
) -> tuple[ScaredCSampleRecord, ...]:
    """Return only the five explicitly allowed Stage1 cache records."""

    index = build_scared_c_index(
        Path(dataset_root),
        mode=CORRECTED_VIDEO_MODE,
        manifest_path=split_path,
        strict=True,
        archive_validation="lazy",
    )
    records = tuple(record for record in index.records if record.sequence_key in STAGE1_CACHE_SEQUENCES)
    _validate_target_records(records)
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
