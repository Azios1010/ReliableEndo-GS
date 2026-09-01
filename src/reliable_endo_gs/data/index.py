"""Deterministic, path-independent indexing for the inspected SCARED-C layout."""

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

SCARED_C_DATASET_ID = "scared_c"
SCARED_C_PROTOCOL = "scared_c_endoscope_stereo_calibration_v1"
INDEX_SCHEMA_VERSION = 1


class IndexError(ValueError):
    """Raised when a dataset cannot produce a deterministic sample index."""


@dataclass(frozen=True, slots=True)
class ScaredCRecord:
    """One top-level static stereo keyframe record.

    The paths are operational pointers only.  They are never included in the
    logical identity or serialized scientific manifest.
    """

    dataset_id: str
    protocol: str
    sequence_id: str
    keyframe_id: str
    frame_id: str
    sample_id: str
    keyframe_root: Path
    left_image_path: Path
    right_image_path: Path
    left_depth_path: Path
    right_depth_path: Path
    endoscope_calibration_path: Path
    colmap_intrinsics_path: Path
    frame_log_path: Path
    frame_data_archive_path: Path | None
    relative_members: tuple[str, ...]

    @property
    def logical_id(self) -> str:
        """Return the stable sample identity."""

        return self.sample_id


@dataclass(frozen=True, slots=True)
class SampleIndex:
    """An immutable deterministic index and its validation issues."""

    dataset_id: str
    protocol: str
    records: tuple[ScaredCRecord, ...]
    issues: tuple[str, ...] = ()

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(record.sample_id for record in self.records)

    @property
    def sequence_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(record.sequence_id for record in self.records))

    @property
    def index_hash(self) -> str:
        """Hash logical records and relative member metadata, never root paths."""

        identity = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "protocol": self.protocol,
            "issues": list(self.issues),
            "records": [
                {
                    "sample_id": record.sample_id,
                    "sequence_id": record.sequence_id,
                    "keyframe_id": record.keyframe_id,
                    "frame_id": record.frame_id,
                    "relative_members": list(record.relative_members),
                    "sizes": [
                        record.keyframe_root.joinpath(member).stat().st_size
                        for member in record.relative_members
                    ],
                }
                for record in self.records
            ],
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ScaredSample:
    """Backward-compatible legacy-layout record used by existing unit tests."""

    sequence_id: str
    keyframe_id: str
    frame_id: str
    left_image_path: Path
    right_image_path: Path | None
    depth_path: Path | None

    @property
    def logical_id(self) -> str:
        return f"{self.sequence_id}_{self.keyframe_id}_{self.frame_id}"


def make_sample_id(
    dataset_id: str, sequence_id: str, keyframe_id: str, frame_id: str = "reference"
) -> str:
    """Form a stable, human-readable logical ID from validated components."""

    parts = (dataset_id, sequence_id, keyframe_id, frame_id)
    if any(not part or "/" in part or "\\" in part for part in parts):
        raise ValueError("logical ID components must be non-empty path-safe names")
    return "/".join(parts)


def _sorted_directories(root: Path, prefix: str) -> list[Path]:
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.casefold().startswith(prefix)
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )


def _png_size(path: Path) -> tuple[int, int]:
    try:
        with path.open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError("not a PNG file")
            length = struct.unpack(">I", stream.read(4))[0]
            chunk_type = stream.read(4)
            if chunk_type != b"IHDR" or length < 13:
                raise ValueError("PNG is missing a valid IHDR")
            width, height = struct.unpack(">II", stream.read(8))
            bit_depth = stream.read(1)[0]
            color_type = stream.read(1)[0]
    except (OSError, struct.error, IndexError) as error:
        raise ValueError(f"unable to inspect PNG header: {error}") from error
    if width <= 0 or height <= 0:
        raise ValueError("PNG dimensions must be positive")
    if bit_depth != 8 or color_type not in (2, 6):
        raise ValueError("SCARED-C RGB inputs must be 8-bit RGB or RGBA PNG")
    return width, height


def _missing_members(keyframe_root: Path) -> list[str]:
    required = (
        "Left_Image.png",
        "Right_Image.png",
        "left_depth_map.tiff",
        "right_depth_map.tiff",
        "endoscope_calibration.yaml",
        "intrinsics_colmap.yaml",
        "frame_log.json",
    )
    return [member for member in required if not keyframe_root.joinpath(member).is_file()]


def _candidate_records(
    dataset_root: Path, *, protocol: str
) -> tuple[list[ScaredCRecord], list[str]]:
    records: list[ScaredCRecord] = []
    issues: list[str] = []
    seen: set[str] = set()
    dataset_dirs = _sorted_directories(dataset_root, "dataset_")
    if not dataset_dirs:
        issues.append("no dataset_* sequence directories found")

    for sequence_dir in dataset_dirs:
        keyframe_dirs = _sorted_directories(sequence_dir, "keyframe_")
        if not keyframe_dirs:
            issues.append(f"{sequence_dir.name}: no keyframe_* directories found")
        for keyframe_root in keyframe_dirs:
            sequence_id = sequence_dir.name
            keyframe_id = keyframe_root.name
            sample_id = make_sample_id(SCARED_C_DATASET_ID, sequence_id, keyframe_id, "reference")
            normalized_id = sample_id.casefold()
            if normalized_id in seen:
                issues.append(f"duplicate sample ID: {sample_id}")
                continue
            seen.add(normalized_id)
            missing = _missing_members(keyframe_root)
            if missing:
                issues.append(f"{sample_id}: missing member(s): {', '.join(missing)}")
                continue
            try:
                frame_log = json.loads(
                    (keyframe_root / "frame_log.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                issues.append(f"{sample_id}: invalid frame_log.json: {error}")
                continue
            if not isinstance(frame_log, dict):
                issues.append(f"{sample_id}: frame_log.json must contain an object")
                continue
            included_frames = frame_log.get("included_frames")
            if not isinstance(included_frames, list):
                issues.append(f"{sample_id}: frame_log.json lacks included_frames list")
                continue
            left_image = keyframe_root / "Left_Image.png"
            right_image = keyframe_root / "Right_Image.png"
            try:
                if _png_size(left_image) != _png_size(right_image):
                    issues.append(f"{sample_id}: left/right image dimensions do not match")
                    continue
            except ValueError as error:
                issues.append(f"{sample_id}: invalid image header: {error}")
                continue
            relative_members = tuple(
                sorted(
                    (
                        "Left_Image.png",
                        "Right_Image.png",
                        "left_depth_map.tiff",
                        "right_depth_map.tiff",
                        "endoscope_calibration.yaml",
                        "intrinsics_colmap.yaml",
                        "frame_log.json",
                    )
                )
            )
            frame_archive = keyframe_root / "data" / "frame_data.tar.gz"
            records.append(
                ScaredCRecord(
                    dataset_id=SCARED_C_DATASET_ID,
                    protocol=protocol,
                    sequence_id=sequence_id,
                    keyframe_id=keyframe_id,
                    frame_id="reference",
                    sample_id=sample_id,
                    keyframe_root=keyframe_root,
                    left_image_path=left_image,
                    right_image_path=right_image,
                    left_depth_path=keyframe_root / "left_depth_map.tiff",
                    right_depth_path=keyframe_root / "right_depth_map.tiff",
                    endoscope_calibration_path=keyframe_root / "endoscope_calibration.yaml",
                    colmap_intrinsics_path=keyframe_root / "intrinsics_colmap.yaml",
                    frame_log_path=keyframe_root / "frame_log.json",
                    frame_data_archive_path=frame_archive if frame_archive.is_file() else None,
                    relative_members=relative_members,
                )
            )
    records.sort(key=lambda record: (record.sequence_id.casefold(), record.keyframe_id.casefold()))
    return records, issues


def build_scared_c_index(
    dataset_root: Path, *, strict: bool = True, protocol: str = SCARED_C_PROTOCOL
) -> SampleIndex:
    """Enumerate top-level static SCARED-C keyframe records deterministically."""

    if not dataset_root.exists():
        raise IndexError("dataset root does not exist")
    if not dataset_root.is_dir():
        raise IndexError("dataset root is not a directory")
    records, issues = _candidate_records(dataset_root, protocol=protocol)
    index = SampleIndex(SCARED_C_DATASET_ID, protocol, tuple(records), tuple(sorted(issues)))
    if strict and index.issues:
        raise IndexError("; ".join(index.issues))
    if strict and not index.records:
        raise IndexError("no complete SCARED-C keyframe records found")
    return index


def build_scared_index(dataset_root: Path) -> list[ScaredSample]:
    """Enumerate the pre-Plan-02 synthetic legacy layout without using it for SCARED-C."""

    samples: list[ScaredSample] = []
    seen_ids: set[str] = set()
    if not dataset_root.is_dir():
        return samples
    for sequence_dir in _sorted_directories(dataset_root, "dataset_"):
        for keyframe_dir in _sorted_directories(sequence_dir, "keyframe_"):
            left_dir = keyframe_dir / "data" / "left"
            if not left_dir.is_dir():
                continue
            for left_image in sorted(left_dir.glob("*.png"), key=lambda path: path.name.casefold()):
                frame_id = left_image.stem
                logical_id = f"{sequence_dir.name}_{keyframe_dir.name}_{frame_id}"
                if logical_id in seen_ids:
                    raise IndexError(f"duplicate sample detected: {logical_id}")
                seen_ids.add(logical_id)
                right_image = keyframe_dir / "data" / "right" / left_image.name
                depth = keyframe_dir / "data" / "depth" / f"{frame_id}.tiff"
                samples.append(
                    ScaredSample(
                        sequence_id=sequence_dir.name,
                        keyframe_id=keyframe_dir.name,
                        frame_id=frame_id,
                        left_image_path=left_image,
                        right_image_path=right_image if right_image.exists() else None,
                        depth_path=depth if depth.exists() else None,
                    )
                )
    return samples
