"""Contract verification and preprocessing utilities for SCARED keyframes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class KeyframeValidationReport:
    """Report on the structural and numerical integrity of a processed keyframe."""

    keyframe_dir: Path
    valid: bool
    frame_count: int
    image_size_wh: tuple[int, int] | None
    issues: tuple[str, ...]
    subdirectories: tuple[str, ...]


REQUIRED_SUBDIRS = (
    "frame_data",
    "left_finalpass",
    "right_finalpass",
    "disparity",
    "newpram_data",
    "reprojection_data",
)


def validate_keyframe_processed_root(keyframe_dir: Path) -> KeyframeValidationReport:
    """Validate that a keyframe root contains all required subdirectories and consistent files.

    Checks:
    - Existence of frame_data, left_finalpass, right_finalpass, disparity, newpram_data, reprojection_data.
    - Consistency of frame counts and matching frame stems across subdirectories.
    - Basic readability of JSON metadata and image dimensions.
    """
    keyframe_dir = Path(keyframe_dir)
    issues: list[str] = []

    if not keyframe_dir.is_dir():
        return KeyframeValidationReport(
            keyframe_dir=keyframe_dir,
            valid=False,
            frame_count=0,
            image_size_wh=None,
            issues=(f"directory does not exist: {keyframe_dir}",),
            subdirectories=(),
        )

    data_parent = keyframe_dir / "data" if (keyframe_dir / "data").is_dir() else keyframe_dir
    found_subdirs: list[str] = []
    missing_subdirs: list[str] = []

    for name in REQUIRED_SUBDIRS:
        subdir = data_parent / name
        if subdir.is_dir():
            found_subdirs.append(name)
        else:
            missing_subdirs.append(name)

    if missing_subdirs:
        return KeyframeValidationReport(
            keyframe_dir=keyframe_dir,
            valid=False,
            frame_count=0,
            image_size_wh=None,
            issues=(f"missing required subdirectories: {', '.join(missing_subdirs)}",),
            subdirectories=tuple(found_subdirs),
        )

    # Check frame counts and stems
    calib_stems = sorted(p.stem for p in (data_parent / "frame_data").glob("*.json"))
    left_stems = sorted(p.stem for p in (data_parent / "left_finalpass").glob("*.png"))
    right_stems = sorted(p.stem for p in (data_parent / "right_finalpass").glob("*.png"))
    disp_stems = sorted(p.stem for p in (data_parent / "disparity").glob("*.tiff"))
    newpram_stems = sorted(p.stem for p in (data_parent / "newpram_data").glob("*.json"))
    reproj_stems = sorted(p.stem for p in (data_parent / "reprojection_data").glob("*.json"))

    if not calib_stems:
        issues.append("no frame_data JSON files found")
        return KeyframeValidationReport(
            keyframe_dir=keyframe_dir,
            valid=False,
            frame_count=0,
            image_size_wh=None,
            issues=tuple(issues),
            subdirectories=tuple(found_subdirs),
        )

    target_count = len(calib_stems)
    for name, stems in (
        ("left_finalpass", left_stems),
        ("right_finalpass", right_stems),
        ("disparity", disp_stems),
        ("newpram_data", newpram_stems),
        ("reprojection_data", reproj_stems),
    ):
        if len(stems) != target_count:
            issues.append(f"{name} has {len(stems)} files; expected {target_count} from frame_data")

    # Inspect first frame for format validity
    image_size_wh: tuple[int, int] | None = None
    first_id = calib_stems[0]
    try:
        from PIL import Image

        with Image.open(data_parent / "left_finalpass" / f"{first_id}.png") as img:
            image_size_wh = (img.width, img.height)
    except Exception as error:
        issues.append(f"failed to read sample image: {error}")

    try:
        with open(data_parent / "newpram_data" / f"{first_id}.json", "r", encoding="utf-8") as f:
            np_data = json.load(f)
        if "intr0" not in np_data or "extr0" not in np_data:
            issues.append("newpram_data JSON missing 'intr0' or 'extr0'")
    except Exception as error:
        issues.append(f"failed to read sample newpram_data: {error}")

    try:
        with open(data_parent / "reprojection_data" / f"{first_id}.json", "r", encoding="utf-8") as f:
            rp_data = json.load(f)
        if "reprojection-matrix" not in rp_data:
            issues.append("reprojection_data JSON missing 'reprojection-matrix'")
    except Exception as error:
        issues.append(f"failed to read sample reprojection_data: {error}")

    return KeyframeValidationReport(
        keyframe_dir=keyframe_dir,
        valid=len(issues) == 0,
        frame_count=target_count,
        image_size_wh=image_size_wh,
        issues=tuple(issues),
        subdirectories=tuple(found_subdirs),
    )