"""Lazy, contract-driven SCARED-C indexing and sample access.

This module is deliberately separate from the historical SCARED-C reference
loader in :mod:`reliable_endo_gs.data.loaders`.  It implements the audited
SCARED-C corrected-video/static-keyframe contract and never extracts the
dataset archives into a duplicate image directory.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal, cast

import numpy as np

from reliable_endo_gs.data.paths import DATA_ROOT_ENV_VAR, resolve_data_root
from reliable_endo_gs.data.scared_c_stereo import (
    ArchiveMember,
    RectificationCache,
    RectificationMaps,
    StackedStereoVideo,
    StereoCalibrationContract,
    StereoContractError,
    archive_member_map,
    build_rectified_geometry,
    index_archive_members,
    load_stereo_calibration,
    read_archive_xyz,
    read_frame_pose,
    read_static_rgb,
    read_static_xyz,
    valid_xyz_mask,
    validate_archive_frame_identity,
)

SCARED_C_DATASET_ID = "scared_c"
SCARED_C_PROTOCOL = "scared_c_endoscope_stereo_calibration_v1"
SCARED_C_UPSTREAM_REPOSITORY = "juseonghan/SCARED-C"
SCARED_C_UPSTREAM_REVISION = "44baac1187c8729c96db0d1def569bd94c9d9417"
SCARED_C_GEOMETRY_UNIT = "MILLIMETRES"
STATIC_KEYFRAME = "STATIC_KEYFRAME"
CORRECTED_VIDEO_FRAME = "CORRECTED_VIDEO_FRAME"
STATIC_MODE = "static_keyframe"
CORRECTED_VIDEO_MODE = "corrected_video"
FINAL_DATASET_ID = "dataset_6"

Mode = Literal["static_keyframe", "corrected_video"]


class ScaredCError(ValueError):
    """Base error for SCARED-C indexing and contract failures."""


class ScaredCIndexError(ScaredCError):
    """Raised when a strict index cannot be constructed."""


class FinalDatasetGuardError(ScaredCError):
    """Raised before any final-role sample content is materialized."""


class InactiveSplitError(ScaredCError):
    """Raised when an inactive scientific split is requested as an experiment."""


@dataclass(frozen=True, slots=True)
class FrameLog:
    """Validated temporal metadata; frame IDs are one-based."""

    key: str | None
    total_frames_on_disk: int
    included_frames: tuple[int, ...]
    excluded_frames: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ScaredCRoleManifest:
    """Validated strict-shift role partition."""

    path: Path
    dataset: str
    dataset_version: str
    split_name: str
    active: bool
    roles: Mapping[str, tuple[str, ...]]
    upstream_repository: str | None
    upstream_revision: str | None

    def require_active_role(self, role: str) -> None:
        """Reject scientific use of a manifest that is still explicitly inactive."""

        if role not in self.roles:
            raise InactiveSplitError(f"unknown SCARED-C role: {role}")
        if not self.active:
            raise InactiveSplitError(
                f"SCARED-C split {self.split_name} is inactive; it cannot launch training"
            )

    def role_for(self, sequence_key: str, *, source_type: str) -> str | None:
        """Return the role for one case/keyframe, preserving final nesting."""

        if source_type == STATIC_KEYFRAME and sequence_key in self.roles.get(
            "FINAL_STATIC_UNTOUCHED", ()
        ):
            return "FINAL_STATIC_UNTOUCHED"
        for role, members in self.roles.items():
            if role == "FINAL_STATIC_UNTOUCHED":
                continue
            if sequence_key in members:
                return role
        return None

    def is_final(self, sequence_key: str, *, source_type: str) -> bool:
        """Return whether content is protected by the dataset-6 final guard."""

        if sequence_key.startswith("6_"):
            return True
        role = self.role_for(sequence_key, source_type=source_type)
        return role in {"FINAL_UNTOUCHED", "FINAL_STATIC_UNTOUCHED"}


@dataclass(frozen=True, slots=True)
class ScaredCSampleRecord:
    """Metadata and lazy pointers for one static or corrected-video sample."""

    dataset_id: str
    sequence_id: str
    keyframe_id: str
    frame_id: int | str
    sample_id: str
    source_type: str
    mode: Mode
    keyframe_root: Path
    role: str | None
    final_role: bool
    endoscope_calibration_path: Path
    colmap_intrinsics_path: Path | None
    left_static_path: Path | None = None
    right_static_path: Path | None = None
    left_static_xyz_path: Path | None = None
    right_static_xyz_path: Path | None = None
    video_path: Path | None = None
    frame_data_archive_path: Path | None = None
    rgb_frames_archive_path: Path | None = None
    scene_points_archive_path: Path | None = None
    frame_data_member: str | None = None
    rgb_frames_member: str | None = None
    scene_points_member: str | None = None
    video_frame_count: int | None = None
    archive_members_verified: bool = False

    @property
    def sequence_key(self) -> str:
        """Return the manifest key, e.g. ``1_1``."""

        dataset = self.sequence_id.removeprefix("dataset_")
        keyframe = self.keyframe_id.removeprefix("keyframe_")
        return f"{dataset}_{keyframe}"

    @property
    def video_index(self) -> int | None:
        """Return the zero-based OpenCV index for a corrected frame."""

        return self.frame_id - 1 if isinstance(self.frame_id, int) else None


@dataclass(frozen=True, slots=True)
class ScaredCIndex:
    """Immutable deterministic index for one explicit loader mode."""

    root: Path
    mode: Mode
    records: tuple[ScaredCSampleRecord, ...]
    issues: tuple[str, ...] = ()
    manifest: ScaredCRoleManifest | None = None
    archive_validation: Literal["eager", "lazy"] = "eager"

    def __len__(self) -> int:
        return len(self.records)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(record.sample_id for record in self.records)

    @property
    def sequence_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(record.sequence_id for record in self.records))

    @property
    def final_records(self) -> tuple[ScaredCSampleRecord, ...]:
        return tuple(record for record in self.records if record.final_role)

    def records_for_role(self, role: str) -> tuple[ScaredCSampleRecord, ...]:
        """Return role members only after an explicitly active split is supplied."""

        if self.manifest is None:
            raise InactiveSplitError("no SCARED-C role manifest is available")
        self.manifest.require_active_role(role)
        return tuple(record for record in self.records if record.role == role)


@dataclass(frozen=True, slots=True)
class _CorrectedKeyframeAssets:
    sequence_id: str
    keyframe_id: str
    keyframe_root: Path
    endoscope_calibration_path: Path
    colmap_intrinsics_path: Path
    video_path: Path
    frame_data_archive_path: Path
    rgb_frames_archive_path: Path
    scene_points_archive_path: Path
    frame_log: FrameLog
    frame_data_members: tuple[ArchiveMember, ...]
    rgb_frame_members: tuple[ArchiveMember, ...]
    scene_point_members: tuple[ArchiveMember, ...]
    video_frame_count: int
    archive_members_verified: bool


def resolve_scared_c_root(
    root: str | Path | None = None,
    *,
    global_root: str | Path | None = None,
    config_root: str | Path = "scared_c",
    env_var: str = DATA_ROOT_ENV_VAR,
) -> Path:
    """Resolve ``scared_c`` through ``RELIABLE_ENDO_DATA_ROOT`` when relative."""

    dataset_root = Path(config_root if root is None else root)
    base_root = None if global_root is None else Path(global_root)
    return resolve_data_root(dataset_root, global_root=base_root, env_var=env_var)


def _default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[3] / "splits" / "scared_c" / "strict_shift_v1.json"


def _validate_manifest_role_partition(roles: Mapping[str, tuple[str, ...]]) -> None:
    ordinary_roles = {
        role: set(members) for role, members in roles.items() if role != "FINAL_STATIC_UNTOUCHED"
    }
    names = tuple(ordinary_roles)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = ordinary_roles[first] & ordinary_roles[second]
            if overlap:
                raise ScaredCError(
                    f"sequence role overlap between {first} and {second}: {sorted(overlap)}"
                )
    final = ordinary_roles.get("FINAL_UNTOUCHED", set())
    final_static = set(roles.get("FINAL_STATIC_UNTOUCHED", ()))
    if not final <= final_static:
        raise ScaredCError("FINAL_UNTOUCHED must be contained in FINAL_STATIC_UNTOUCHED")


def load_scared_c_role_manifest(path: Path | None = None) -> ScaredCRoleManifest:
    """Load and validate the inactive strict-shift role manifest."""

    manifest_path = _default_manifest_path() if path is None else Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScaredCError(
            f"unable to read strict-shift manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ScaredCError("strict-shift manifest must contain a JSON object")
    if payload.get("dataset") != SCARED_C_DATASET_ID:
        raise ScaredCError("strict-shift manifest has the wrong dataset identity")
    if payload.get("active") is not False:
        raise InactiveSplitError("strict_shift_v1 must remain ACTIVE = FALSE during loader work")
    if payload.get("dataset_version") != f"upstream_{SCARED_C_UPSTREAM_REVISION}":
        raise ScaredCError(
            "strict-shift manifest upstream revision does not match the frozen revision"
        )
    raw_roles = payload.get("roles")
    if not isinstance(raw_roles, dict):
        raise ScaredCError("strict-shift manifest roles must be an object")
    roles: dict[str, tuple[str, ...]] = {}
    for role, members in raw_roles.items():
        if not isinstance(role, str) or not isinstance(members, list):
            raise ScaredCError("strict-shift role names and members must be strings/lists")
        normalized: list[str] = []
        for member in members:
            if not isinstance(member, str) or not re.fullmatch(r"\d+_\d+", member):
                raise ScaredCError(f"invalid SCARED-C sequence key in {role}: {member!r}")
            normalized.append(member)
        if len(set(normalized)) != len(normalized):
            raise ScaredCError(f"duplicate sequence key in role {role}")
        roles[role] = tuple(normalized)
    _validate_manifest_role_partition(roles)
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ScaredCError("strict-shift manifest provenance must be an object")
    return ScaredCRoleManifest(
        path=manifest_path,
        dataset=SCARED_C_DATASET_ID,
        dataset_version=cast(str, payload["dataset_version"]),
        split_name=cast(str, payload.get("split_name", "strict_shift_v1")),
        active=False,
        roles=roles,
        upstream_repository=(
            cast(str, provenance["upstream_repository"])
            if isinstance(provenance.get("upstream_repository"), str)
            else None
        ),
        upstream_revision=(
            cast(str, provenance["upstream_revision"])
            if isinstance(provenance.get("upstream_revision"), str)
            else None
        ),
    )


def parse_frame_log(
    path: Path,
    *,
    sequence_id: str | None = None,
    keyframe_id: str | None = None,
) -> FrameLog:
    """Parse one frame log without opening any image/video/archive content."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScaredCIndexError(f"invalid frame log {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ScaredCIndexError(f"frame log {path} must be a JSON object")
    raw_included = payload.get("included_frames")
    if not isinstance(raw_included, list):
        raise ScaredCIndexError(f"frame log {path} lacks included_frames")
    try:
        included = tuple(int(frame_id) for frame_id in raw_included)
    except (TypeError, ValueError) as error:
        raise ScaredCIndexError(f"frame log {path} has a non-integer frame ID") from error
    if any(frame_id < 1 for frame_id in included):
        raise ScaredCIndexError(f"frame log {path} contains a non-positive frame ID")
    if len(set(included)) != len(included):
        raise ScaredCIndexError(f"frame log {path} contains duplicate frame IDs")
    total = payload.get("total_frames_on_disk")
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ScaredCIndexError(f"frame log {path} has invalid total_frames_on_disk")
    if any(frame_id > total for frame_id in included):
        raise ScaredCIndexError(f"frame log {path} contains an ID beyond total_frames_on_disk")
    key = payload.get("key")
    if key is not None and not isinstance(key, str):
        raise ScaredCIndexError(f"frame log {path} key must be a string or null")
    if sequence_id is not None and keyframe_id is not None and key is not None:
        expected_key = (
            f"{sequence_id.removeprefix('dataset_')}_{keyframe_id.removeprefix('keyframe_')}"
        )
        if key != expected_key:
            raise ScaredCIndexError(f"frame log key {key!r} does not match {expected_key!r}")
    raw_excluded = payload.get("excluded_frames", [])
    if not isinstance(raw_excluded, list):
        raise ScaredCIndexError(f"frame log {path} excluded_frames must be a list")
    try:
        excluded = tuple(sorted(int(frame_id) for frame_id in raw_excluded))
    except (TypeError, ValueError) as error:
        raise ScaredCIndexError(f"frame log {path} has a non-integer excluded frame ID") from error
    return FrameLog(
        key=key,
        total_frames_on_disk=total,
        included_frames=tuple(sorted(included)),
        excluded_frames=excluded,
    )


def _sorted_dataset_dirs(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (path for path in root.iterdir() if path.is_dir() and path.name.startswith("dataset_")),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )


def _sorted_keyframe_dirs(sequence_root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in sequence_root.iterdir()
                if path.is_dir() and path.name.startswith("keyframe_")
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Read only the PNG signature/IHDR needed for lazy static indexing."""

    try:
        with path.open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ScaredCIndexError(f"not a PNG file: {path}")
            length_bytes = stream.read(4)
            chunk_type = stream.read(4)
            if len(length_bytes) != 4 or chunk_type != b"IHDR":
                raise ScaredCIndexError(f"PNG has no valid IHDR: {path}")
            length = struct.unpack(">I", length_bytes)[0]
            header = stream.read(13)
            if length < 13 or len(header) != 13:
                raise ScaredCIndexError(f"PNG IHDR is truncated: {path}")
            width, height, bit_depth, color_type = struct.unpack(">IIBB", header[:10])
    except OSError as error:
        raise ScaredCIndexError(f"unable to inspect PNG {path}: {error}") from error
    if width < 1 or height < 1 or bit_depth != 8 or color_type not in (2, 6):
        raise ScaredCIndexError(f"unsupported static RGB PNG header: {path}")
    return width, height


def _manifest_key(sequence_id: str, keyframe_id: str) -> str:
    return f"{sequence_id.removeprefix('dataset_')}_{keyframe_id.removeprefix('keyframe_')}"


def _record_role(
    manifest: ScaredCRoleManifest | None,
    *,
    sequence_id: str,
    keyframe_id: str,
    source_type: str,
) -> tuple[str | None, bool]:
    sequence_key = _manifest_key(sequence_id, keyframe_id)
    if manifest is None:
        return (
            "FINAL_UNTOUCHED" if sequence_id == FINAL_DATASET_ID else None,
            sequence_id == FINAL_DATASET_ID,
        )
    role = manifest.role_for(sequence_key, source_type=source_type)
    return role, manifest.is_final(sequence_key, source_type=source_type)


def _static_records(
    root: Path,
    manifest: ScaredCRoleManifest | None,
) -> tuple[tuple[ScaredCSampleRecord, ...], tuple[str, ...]]:
    records: list[ScaredCSampleRecord] = []
    issues: list[str] = []
    for sequence_root in _sorted_dataset_dirs(root):
        for keyframe_root in _sorted_keyframe_dirs(sequence_root):
            files = {
                "left": keyframe_root / "Left_Image.png",
                "right": keyframe_root / "Right_Image.png",
                "left_xyz": keyframe_root / "left_depth_map.tiff",
                "right_xyz": keyframe_root / "right_depth_map.tiff",
                "calibration": keyframe_root / "endoscope_calibration.yaml",
            }
            display_names = {
                "left": "Left_Image.png",
                "right": "Right_Image.png",
                "left_xyz": "left_depth_map.tiff",
                "right_xyz": "right_depth_map.tiff",
                "calibration": "endoscope_calibration.yaml",
            }
            missing = tuple(
                display_names[name] for name, path in files.items() if not path.is_file()
            )
            if missing:
                issues.append(
                    f"{sequence_root.name}/{keyframe_root.name}: missing static asset(s): "
                    + ", ".join(missing)
                )
                continue
            try:
                left_size = _png_dimensions(files["left"])
                right_size = _png_dimensions(files["right"])
            except ScaredCIndexError as error:
                issues.append(str(error))
                continue
            if left_size != right_size:
                issues.append(
                    f"{sequence_root.name}/{keyframe_root.name}: static left/right dimensions differ"
                )
                continue
            role, final_role = _record_role(
                manifest,
                sequence_id=sequence_root.name,
                keyframe_id=keyframe_root.name,
                source_type=STATIC_KEYFRAME,
            )
            sample_id = f"{SCARED_C_DATASET_ID}/{sequence_root.name}/{keyframe_root.name}/reference"
            records.append(
                ScaredCSampleRecord(
                    dataset_id=SCARED_C_DATASET_ID,
                    sequence_id=sequence_root.name,
                    keyframe_id=keyframe_root.name,
                    frame_id="reference",
                    sample_id=sample_id,
                    source_type=STATIC_KEYFRAME,
                    mode=STATIC_MODE,
                    keyframe_root=keyframe_root,
                    role=role,
                    final_role=final_role,
                    endoscope_calibration_path=files["calibration"],
                    colmap_intrinsics_path=(
                        keyframe_root / "intrinsics_colmap.yaml"
                        if (keyframe_root / "intrinsics_colmap.yaml").is_file()
                        else None
                    ),
                    left_static_path=files["left"],
                    right_static_path=files["right"],
                    left_static_xyz_path=files["left_xyz"],
                    right_static_xyz_path=files["right_xyz"],
                )
            )
    return tuple(records), tuple(sorted(issues))


def _corrected_assets(
    sequence_id: str,
    keyframe_root: Path,
    *,
    archive_validation: Literal["eager", "lazy"],
) -> tuple[_CorrectedKeyframeAssets | None, tuple[str, ...]]:
    data_root = keyframe_root / "data"
    paths = {
        "calibration": keyframe_root / "endoscope_calibration.yaml",
        "colmap": keyframe_root / "intrinsics_colmap.yaml",
        "video": data_root / "rgb.mp4",
        "frame_data": data_root / "frame_data.tar.gz",
        "rgb_frames": data_root / "rgb_frames.tar.gz",
        "scene_points": data_root / "scene_points.tar.gz",
        "frame_log": keyframe_root / "frame_log.json",
    }
    dynamic_paths = (
        paths["video"],
        paths["frame_data"],
        paths["rgb_frames"],
        paths["scene_points"],
        paths["frame_log"],
    )
    if not paths["frame_log"].is_file() and not any(path.is_file() for path in dynamic_paths[:-1]):
        # Legitimate structured-light-only keyframes have no temporal assets.
        return None, ()
    missing = tuple(name for name, path in paths.items() if not path.is_file())
    if missing:
        return None, (
            f"{sequence_id}/{keyframe_root.name}: missing corrected-video asset(s): "
            + ", ".join(missing),
        )
    try:
        frame_log = parse_frame_log(
            paths["frame_log"], sequence_id=sequence_id, keyframe_id=keyframe_root.name
        )
        if archive_validation == "eager":
            frame_data_members = index_archive_members(paths["frame_data"], kind="frame_data")
            rgb_frame_members = index_archive_members(paths["rgb_frames"], kind="rgb_frames")
            scene_point_members = index_archive_members(paths["scene_points"], kind="scene_points")
        else:
            # Canonical member names are retained as a lazy index.  The strict
            # eager path above remains available for complete archive identity
            # validation; lazy construction defers compressed archive scans to
            # explicit sample access/audit code.
            frame_ids = frame_log.included_frames
            frame_data_members = tuple(
                ArchiveMember(frame_id, f"frame_data{frame_id:06d}.json") for frame_id in frame_ids
            )
            rgb_frame_members = tuple(
                ArchiveMember(frame_id, f"frame{frame_id:06d}.png") for frame_id in frame_ids
            )
            scene_point_members = tuple(
                ArchiveMember(frame_id, f"scene_points{frame_id:06d}.tiff")
                for frame_id in frame_ids
            )
        with StackedStereoVideo(paths["video"]) as video:
            video_frame_count = video.metadata().frame_count
    except (OSError, ScaredCError, StereoContractError) as error:
        return None, (f"{sequence_id}/{keyframe_root.name}: {error}",)
    if video_frame_count != frame_log.total_frames_on_disk:
        return None, (
            f"{sequence_id}/{keyframe_root.name}: frame_log total {frame_log.total_frames_on_disk} "
            f"does not match video count {video_frame_count}",
        )
    return (
        _CorrectedKeyframeAssets(
            sequence_id=sequence_id,
            keyframe_id=keyframe_root.name,
            keyframe_root=keyframe_root,
            endoscope_calibration_path=paths["calibration"],
            colmap_intrinsics_path=paths["colmap"],
            video_path=paths["video"],
            frame_data_archive_path=paths["frame_data"],
            rgb_frames_archive_path=paths["rgb_frames"],
            scene_points_archive_path=paths["scene_points"],
            frame_log=frame_log,
            frame_data_members=frame_data_members,
            rgb_frame_members=rgb_frame_members,
            scene_point_members=scene_point_members,
            video_frame_count=video_frame_count,
            archive_members_verified=archive_validation == "eager",
        ),
        (),
    )


def _corrected_records(
    root: Path,
    manifest: ScaredCRoleManifest | None,
    *,
    archive_validation: Literal["eager", "lazy"],
) -> tuple[tuple[ScaredCSampleRecord, ...], tuple[str, ...]]:
    records: list[ScaredCSampleRecord] = []
    issues: list[str] = []
    for sequence_root in _sorted_dataset_dirs(root):
        for keyframe_root in _sorted_keyframe_dirs(sequence_root):
            assets, asset_issues = _corrected_assets(
                sequence_root.name,
                keyframe_root,
                archive_validation=archive_validation,
            )
            if assets is None:
                issues.extend(asset_issues)
                continue
            frame_log_ids = set(assets.frame_log.included_frames)
            archive_members = {
                "frame_data": assets.frame_data_members,
                "rgb_frames": assets.rgb_frame_members,
                "scene_points": assets.scene_point_members,
            }
            if assets.archive_members_verified:
                try:
                    common_ids, identity_issues = validate_archive_frame_identity(
                        frame_log_ids, archive_members
                    )
                except StereoContractError as error:
                    issues.append(f"{sequence_root.name}/{keyframe_root.name}: {error}")
                    continue
            else:
                common_ids, identity_issues = tuple(sorted(frame_log_ids)), ()
            issues.extend(
                f"{sequence_root.name}/{keyframe_root.name}: {issue}" for issue in identity_issues
            )
            frame_data_map = archive_member_map(assets.frame_data_members)
            rgb_map = archive_member_map(assets.rgb_frame_members)
            scene_map = archive_member_map(assets.scene_point_members)
            for frame_id in common_ids:
                if frame_id > assets.video_frame_count:
                    issues.append(
                        f"{sequence_root.name}/{keyframe_root.name}: frame {frame_id} "
                        "exceeds video frame count"
                    )
                    continue
                role, final_role = _record_role(
                    manifest,
                    sequence_id=sequence_root.name,
                    keyframe_id=keyframe_root.name,
                    source_type=CORRECTED_VIDEO_FRAME,
                )
                sample_id = (
                    f"{SCARED_C_DATASET_ID}/{sequence_root.name}/{keyframe_root.name}/{frame_id}"
                )
                records.append(
                    ScaredCSampleRecord(
                        dataset_id=SCARED_C_DATASET_ID,
                        sequence_id=sequence_root.name,
                        keyframe_id=keyframe_root.name,
                        frame_id=frame_id,
                        sample_id=sample_id,
                        source_type=CORRECTED_VIDEO_FRAME,
                        mode=CORRECTED_VIDEO_MODE,
                        keyframe_root=keyframe_root,
                        role=role,
                        final_role=final_role,
                        endoscope_calibration_path=assets.endoscope_calibration_path,
                        colmap_intrinsics_path=assets.colmap_intrinsics_path,
                        video_path=assets.video_path,
                        frame_data_archive_path=assets.frame_data_archive_path,
                        rgb_frames_archive_path=assets.rgb_frames_archive_path,
                        scene_points_archive_path=assets.scene_points_archive_path,
                        frame_data_member=frame_data_map[frame_id],
                        rgb_frames_member=rgb_map[frame_id],
                        scene_points_member=scene_map[frame_id],
                        video_frame_count=assets.video_frame_count,
                        archive_members_verified=assets.archive_members_verified,
                    )
                )
    records.sort(
        key=lambda record: (
            record.sequence_id.casefold(),
            record.keyframe_id.casefold(),
            int(record.frame_id),
        )
    )
    return tuple(records), tuple(sorted(issues))


def build_scared_c_index(
    root: str | Path,
    *,
    mode: Mode = STATIC_MODE,
    manifest_path: Path | None = None,
    strict: bool = True,
    archive_validation: Literal["eager", "lazy"] = "eager",
) -> ScaredCIndex:
    """Build a deterministic static or corrected-video index.

    Static indexing inspects only static filenames and PNG headers.  Corrected
    video indexing reads frame logs, archive headers, and video metadata but
    does not decode an image, XYZ TIFF, or archive payload.
    """

    dataset_root = Path(root)
    if not dataset_root.is_dir():
        raise ScaredCIndexError(f"SCARED-C root is not a directory: {dataset_root}")
    if mode not in (STATIC_MODE, CORRECTED_VIDEO_MODE):
        raise ValueError(f"unsupported SCARED-C mode: {mode!r}")
    if archive_validation not in ("eager", "lazy"):
        raise ValueError(f"unsupported archive_validation: {archive_validation!r}")
    manifest = load_scared_c_role_manifest(manifest_path)
    if mode == STATIC_MODE:
        records, issues = _static_records(dataset_root, manifest)
    else:
        records, issues = _corrected_records(
            dataset_root,
            manifest,
            archive_validation=archive_validation,
        )
    index = ScaredCIndex(dataset_root, mode, records, issues, manifest, archive_validation)
    if strict and issues:
        raise ScaredCIndexError("; ".join(issues))
    if strict and not records:
        raise ScaredCIndexError(f"no complete SCARED-C samples found for mode {mode}")
    return index


def validate_scared_c_split_manifest(path: Path | None = None) -> ScaredCRoleManifest:
    """Public alias for manifest validation; the manifest remains inactive."""

    return load_scared_c_role_manifest(path)


def require_active_scared_c_role(manifest: ScaredCRoleManifest, role: str) -> None:
    """Public guard for future training/evaluation entry points."""

    manifest.require_active_role(role)


class ScaredCSample(dict[str, object]):
    """Dictionary-like corrected/static sample with attribute access."""

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _rectify_rgb_pair(
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    rectification: RectificationMaps,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - depends on runtime image
        raise ImportError("SCARED-C RGB rectification requires OpenCV") from error
    left_rect = cv2.remap(
        left_raw,
        rectification.left_map_x,
        rectification.left_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    right_rect = cv2.remap(
        right_raw,
        rectification.right_map_x,
        rectification.right_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return np.ascontiguousarray(left_rect), np.ascontiguousarray(right_rect)


def _calibration_and_rectification(
    record: ScaredCSampleRecord,
    *,
    image_size: tuple[int, int],
    rectification_cache: RectificationCache,
) -> tuple[StereoCalibrationContract, RectificationMaps]:
    calibration = load_stereo_calibration(
        record.endoscope_calibration_path,
        image_size=image_size,
        colmap_intrinsics_path=record.colmap_intrinsics_path,
    )
    return calibration, rectification_cache.get_or_create(record.keyframe_root, calibration)


def materialize_scared_c_rectified_rgb(
    record: ScaredCSampleRecord,
    *,
    allow_final_content: bool = False,
    rectification_cache: RectificationCache | None = None,
    video_readers: dict[Path, StackedStereoVideo] | None = None,
) -> ScaredCSample:
    """Read and rectify one corrected-video stereo pair without reading XYZ.

    Stage1 uses this path together with the derived disparity cache.  RGB
    rectification remains the same canonical OpenCV operation used by
    :func:`materialize_scared_c_sample`; only the repeated XYZ archive read is
    removed from the training-time path.
    """

    if record.final_role and not allow_final_content:
        raise FinalDatasetGuardError(
            "final-role SCARED-C content is guarded; explicit override required"
        )
    if record.mode != CORRECTED_VIDEO_MODE:
        raise ScaredCError("rectified RGB-only materialization requires corrected-video records")
    if (
        record.video_path is None
        or not isinstance(record.frame_id, int)
        or record.frame_id < 1
    ):
        raise ScaredCError("corrected-video record has incomplete RGB pointers")

    cache = RectificationCache() if rectification_cache is None else rectification_cache
    readers = {} if video_readers is None else video_readers
    reader = readers.get(record.video_path)
    if reader is None:
        reader = StackedStereoVideo(record.video_path)
        readers[record.video_path] = reader
    left_raw, right_raw = reader.read(record.frame_id)
    if left_raw.shape != right_raw.shape:
        raise ScaredCError("left/right RGB shapes differ")
    height, width = left_raw.shape[:2]
    calibration, rectification = _calibration_and_rectification(
        record,
        image_size=(width, height),
        rectification_cache=cache,
    )
    left_rect, right_rect = _rectify_rgb_pair(left_raw, right_raw, rectification)
    q32 = float(rectification.Q[3, 2])
    if abs(q32) < 1e-8:
        raise ScaredCError("rectification Q[3, 2] is zero; cannot construct disp_const")
    disp_const = float(rectification.Q[2, 3] / q32)
    if not np.isfinite(disp_const) or disp_const <= 0:
        raise ScaredCError(f"rectification disp_const must be positive and finite; got {disp_const}")
    return ScaredCSample(
        {
            "dataset_id": record.dataset_id,
            "keyframe_id": record.keyframe_id,
            "frame_id": record.frame_id,
            "sample_id": record.sample_id,
            "source_type": record.source_type,
            "rgb_left_raw": left_raw,
            "rgb_right_raw": right_raw,
            "rgb_left_rect": left_rect,
            "rgb_right_rect": right_rect,
            "M1": calibration.M1,
            "D1": calibration.D1,
            "M2": calibration.M2,
            "D2": calibration.D2,
            "R": calibration.R,
            "T": calibration.T,
            "R1": rectification.R1,
            "R2": rectification.R2,
            "P1": rectification.P1,
            "P2": rectification.P2,
            "Q": rectification.Q,
            "disp_const": disp_const,
            "geometry_unit": SCARED_C_GEOMETRY_UNIT,
            "upstream_repo": SCARED_C_UPSTREAM_REPOSITORY,
            "upstream_revision": SCARED_C_UPSTREAM_REVISION,
            "protocol": SCARED_C_PROTOCOL,
            "role": record.role,
            "video_index": record.video_index,
        }
    )


def _materialize_record(
    record: ScaredCSampleRecord,
    *,
    rectification_cache: RectificationCache,
    video_readers: dict[Path, StackedStereoVideo],
) -> ScaredCSample:
    if record.mode == STATIC_MODE:
        if (
            record.left_static_path is None
            or record.right_static_path is None
            or record.left_static_xyz_path is None
        ):
            raise ScaredCError("static record has incomplete static asset pointers")
        left_raw = read_static_rgb(record.left_static_path)
        right_raw = read_static_rgb(record.right_static_path)
        xyz_left = read_static_xyz(record.left_static_xyz_path)
        pose = None
    else:
        if (
            record.video_path is None
            or record.frame_data_archive_path is None
            or record.scene_points_archive_path is None
            or record.frame_data_member is None
            or record.scene_points_member is None
            or not isinstance(record.frame_id, int)
        ):
            raise ScaredCError("corrected-video record has incomplete archive pointers")
        reader = video_readers.get(record.video_path)
        if reader is None:
            reader = StackedStereoVideo(record.video_path)
            video_readers[record.video_path] = reader
        left_raw, right_raw = reader.read(record.frame_id)
        xyz_left = read_archive_xyz(record.scene_points_archive_path, record.scene_points_member)
        pose = read_frame_pose(record.frame_data_archive_path, record.frame_data_member)

    if left_raw.shape != right_raw.shape:
        raise ScaredCError("left/right RGB shapes differ")
    height, width = left_raw.shape[:2]
    if xyz_left.shape[:2] != (height, width):
        raise ScaredCError(
            f"RGB and corrected XYZ shapes differ: {(width, height)} vs "
            f"{(xyz_left.shape[1], xyz_left.shape[0])}"
        )
    calibration, rectification = _calibration_and_rectification(
        record,
        image_size=(width, height),
        rectification_cache=rectification_cache,
    )
    left_rect, right_rect = _rectify_rgb_pair(left_raw, right_raw, rectification)
    geometry = build_rectified_geometry(xyz_left, calibration, rectification)
    source_valid = valid_xyz_mask(xyz_left)
    return ScaredCSample(
        {
            "dataset_id": record.dataset_id,
            "keyframe_id": record.keyframe_id,
            "frame_id": record.frame_id,
            "sample_id": record.sample_id,
            "source_type": record.source_type,
            "rgb_left_raw": left_raw,
            "rgb_right_raw": right_raw,
            "rgb_left_rect": left_rect,
            "rgb_right_rect": right_rect,
            "xyz_left_corrected": np.ascontiguousarray(xyz_left, dtype=np.float32),
            "valid_xyz_mask": source_valid,
            "depth_left_rect": geometry.depth_left_rect,
            "valid_depth_mask": geometry.valid_depth_mask,
            "disparity_left_rect": geometry.disparity_left_rect,
            "valid_disparity_mask": geometry.valid_disparity_mask,
            "K_colmap": calibration.K_colmap,
            "M1": calibration.M1,
            "D1": calibration.D1,
            "M2": calibration.M2,
            "D2": calibration.D2,
            "R": calibration.R,
            "T": calibration.T,
            "R1": rectification.R1,
            "R2": rectification.R2,
            "P1": rectification.P1,
            "P2": rectification.P2,
            "Q": rectification.Q,
            "camera_pose_raw": pose,
            "camera_pose_direction": "WORLD_TO_CAMERA_RELATIVE_KEYFRAME_LEFT",
            "pose_reference": "KEYFRAME_LEFT",
            "geometry_unit": SCARED_C_GEOMETRY_UNIT,
            "upstream_repo": SCARED_C_UPSTREAM_REPOSITORY,
            "upstream_revision": SCARED_C_UPSTREAM_REVISION,
            "protocol": SCARED_C_PROTOCOL,
            "role": record.role,
            "video_index": record.video_index,
        }
    )


class ScaredCLazyDataset(Sequence[ScaredCSample]):
    """A lazy dataset for exactly one explicit SCARED-C mode."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        global_root: str | Path | None = None,
        config_root: str | Path = "scared_c",
        mode: Mode = STATIC_MODE,
        manifest_path: Path | None = None,
        allow_final_content: bool = False,
        strict: bool = True,
        archive_validation: Literal["eager", "lazy"] = "eager",
    ) -> None:
        self.root = resolve_scared_c_root(root, global_root=global_root, config_root=config_root)
        self.mode = mode
        self.allow_final_content = allow_final_content
        self.index = build_scared_c_index(
            self.root,
            mode=mode,
            manifest_path=manifest_path,
            strict=strict,
            archive_validation=archive_validation,
        )
        self._rectification_cache = RectificationCache()
        self._video_readers: dict[Path, StackedStereoVideo] = {}
        self._lock = RLock()

    def __len__(self) -> int:
        return len(self.index)

    def metadata(self, index: int | str) -> ScaredCSampleRecord:
        """Return sample pointers without materializing any content."""

        return self._record(index)

    def _record(self, index: int | str) -> ScaredCSampleRecord:
        if isinstance(index, str):
            for record in self.index.records:
                if record.sample_id == index:
                    return record
            raise KeyError(index)
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("sample index must be an integer or sample ID")
        return self.index.records[index]

    def _guard(self, record: ScaredCSampleRecord) -> None:
        if record.final_role and not self.allow_final_content:
            raise FinalDatasetGuardError(
                "dataset-6/final-role content is guarded; pass allow_final_content=True only "
                "for an explicit future final-evaluation override"
            )

    def __getitem__(self, index: int | str) -> ScaredCSample:
        record = self._record(index)
        self._guard(record)
        with self._lock:
            return _materialize_record(
                record,
                rectification_cache=self._rectification_cache,
                video_readers=self._video_readers,
            )

    def __iter__(self) -> Iterator[ScaredCSample]:
        for index in range(len(self)):
            yield self[index]

    def close(self) -> None:
        """Release lazy video handles and rectification-map references."""

        with self._lock:
            for reader in self._video_readers.values():
                reader.close()
            self._video_readers.clear()
            self._rectification_cache.clear()

    def __enter__(self) -> ScaredCLazyDataset:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


ScaredCDataset = ScaredCLazyDataset
LazyScaredCDataset = ScaredCLazyDataset
ScaredCDataIndex = ScaredCIndex


def materialize_scared_c_sample(
    record: ScaredCSampleRecord,
    *,
    allow_final_content: bool = False,
    rectification_cache: RectificationCache | None = None,
) -> ScaredCSample:
    """Materialize one record while applying the final-role guard first."""

    if record.final_role and not allow_final_content:
        raise FinalDatasetGuardError(
            "final-role SCARED-C content is guarded; explicit override required"
        )
    cache = RectificationCache() if rectification_cache is None else rectification_cache
    return _materialize_record(record, rectification_cache=cache, video_readers={})


__all__ = [
    "CORRECTED_VIDEO_FRAME",
    "CORRECTED_VIDEO_MODE",
    "FINAL_DATASET_ID",
    "FinalDatasetGuardError",
    "FrameLog",
    "InactiveSplitError",
    "LazyScaredCDataset",
    "ScaredCDataIndex",
    "ScaredCError",
    "ScaredCIndex",
    "ScaredCIndexError",
    "ScaredCLazyDataset",
    "ScaredCRoleManifest",
    "ScaredCSample",
    "ScaredCSampleRecord",
    "STATIC_KEYFRAME",
    "STATIC_MODE",
    "build_scared_c_index",
    "load_scared_c_role_manifest",
    "materialize_scared_c_sample",
    "materialize_scared_c_rectified_rgb",
    "parse_frame_log",
    "require_active_scared_c_role",
    "resolve_scared_c_root",
    "validate_scared_c_split_manifest",
]
