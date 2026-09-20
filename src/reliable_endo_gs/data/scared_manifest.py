"""Deterministic manifest and split helpers for SCARED multi-sequence datasets."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from reliable_endo_gs.data.splits import (
    SplitManifest,
    SplitManifestError,
    load_split_manifest,
    validate_split_manifest,
)

SCARED_DATASET_NAME = "scared"

# Four-keyframe v2 split constants (current canonical split)
SCARED_FOUR_KEYFRAME_SPLIT = "controlled_four_keyframe_v2"
SCARED_FOUR_KEYFRAME_VERSION = "four_keyframe_expansion_v2"
SCARED_FOUR_KEYFRAME_TRAIN: tuple[str, ...] = (
    "dataset_1/keyframe_1",
    "dataset_2/keyframe_1",
    "dataset_3/keyframe_1",
)
SCARED_FOUR_KEYFRAME_VALIDATION: tuple[str, ...] = (
    "dataset_7/keyframe_2",
)

# Canonical aliases pointing to v2
SCARED_CONTROLLED_SPLIT = SCARED_FOUR_KEYFRAME_SPLIT
SCARED_CONTROLLED_VERSION = SCARED_FOUR_KEYFRAME_VERSION
SCARED_CONTROLLED_TRAIN = SCARED_FOUR_KEYFRAME_TRAIN
SCARED_CONTROLLED_VALIDATION = SCARED_FOUR_KEYFRAME_VALIDATION

# Historical five-keyframe v1 split constants (retained for backward compatibility)
SCARED_FIVE_KEYFRAME_SPLIT = "controlled_five_keyframe_v1"
SCARED_FIVE_KEYFRAME_VERSION = "five_keyframe_expansion_v1"
SCARED_FIVE_KEYFRAME_TRAIN: tuple[str, ...] = (
    "dataset_1/keyframe_1",
    "dataset_2/keyframe_1",
    "dataset_3/keyframe_1",
)
SCARED_FIVE_KEYFRAME_VALIDATION: tuple[str, ...] = (
    "dataset_7/keyframe_2",
    "dataset_9/keyframe_3",
)


def parse_scared_keyframe_id(entry: str) -> tuple[str, str]:
    """Parse and validate a portable SCARED keyframe path.

    Portable keyframe paths must be relative to the external SCARED root, e.g.
    ``dataset_1/keyframe_1``. Absolute paths, empty components, and directory
    traversals are rejected.
    """
    if not isinstance(entry, str) or not entry.strip():
        raise SplitManifestError("keyframe identifier must be a non-empty string")
    cleaned = entry.strip().replace("\\", "/")
    if cleaned.startswith("/") or ":" in cleaned:
        raise SplitManifestError(
            f"keyframe path {entry!r} must be portable and relative, not absolute"
        )
    parts = [p for p in cleaned.split("/") if p]
    if any(p in (".", "..") for p in parts):
        raise SplitManifestError(f"keyframe path {entry!r} contains invalid traversal segments")
    if len(parts) != 2:
        raise SplitManifestError(
            f"keyframe path {entry!r} must have format 'dataset_X/keyframe_Y'"
        )
    dataset_id, keyframe_id = parts[0], parts[1]
    if not dataset_id.startswith("dataset_"):
        raise SplitManifestError(
            f"dataset identifier {dataset_id!r} in {entry!r} must start with 'dataset_'"
        )
    if not keyframe_id.startswith("keyframe_"):
        raise SplitManifestError(
            f"keyframe identifier {keyframe_id!r} in {entry!r} must start with 'keyframe_'"
        )
    return dataset_id, keyframe_id


def make_scared_sample_id(dataset_id: str, keyframe_id: str, frame_id: str) -> str:
    """Format the canonical sample provenance identifier."""
    if not dataset_id.strip() or not keyframe_id.strip() or not frame_id.strip():
        raise ValueError("sample provenance components must be non-empty")
    return f"{dataset_id.strip()}/{keyframe_id.strip()}/{frame_id.strip()}"


def validate_scared_split_manifest(manifest: SplitManifest) -> None:
    """Validate a SCARED split manifest for portable paths and case/keyframe isolation.

    Ensures:
    1. Base SplitManifest integrity (uniqueness within splits, disjointness).
    2. Dataset identity matches 'scared'.
    3. Every entry in train, validation, and test is a valid portable relative keyframe path.
    4. Complete case-level and keyframe-level isolation: no dataset case (e.g. dataset_1)
       may cross split boundaries.
    """
    validate_split_manifest(manifest)
    if manifest.dataset != SCARED_DATASET_NAME:
        raise SplitManifestError(
            f"expected dataset {SCARED_DATASET_NAME!r}, got {manifest.dataset!r}"
        )

    split_cases: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for split_name, entries in (
        ("train", manifest.train),
        ("validation", manifest.validation),
        ("test", manifest.test),
    ):
        for entry in entries:
            dataset_id, _ = parse_scared_keyframe_id(entry)
            split_cases[split_name].add(dataset_id)

    # Check case-level cross-split leakage
    train_val_overlap = split_cases["train"] & split_cases["validation"]
    if train_val_overlap:
        raise SplitManifestError(
            f"case-level leakage detected: case(s) {sorted(train_val_overlap)} "
            "appear in both train and validation"
        )
    train_test_overlap = split_cases["train"] & split_cases["test"]
    if train_test_overlap:
        raise SplitManifestError(
            f"case-level leakage detected: case(s) {sorted(train_test_overlap)} "
            "appear in both train and test"
        )
    val_test_overlap = split_cases["validation"] & split_cases["test"]
    if val_test_overlap:
        raise SplitManifestError(
            f"case-level leakage detected: case(s) {sorted(val_test_overlap)} "
            "appear in both validation and test"
        )


def validate_sample_provenance_leakage(
    train_samples: Sequence[str],
    validation_samples: Sequence[str],
    test_samples: Sequence[str] = (),
) -> None:
    """Verify that sample provenance identifiers do not leak across splits."""
    train_set = set(train_samples)
    val_set = set(validation_samples)
    test_set = set(test_samples)

    overlap_tv = train_set & val_set
    if overlap_tv:
        raise SplitManifestError(
            f"sample leakage detected: {len(overlap_tv)} samples in both train and validation"
        )
    overlap_tt = train_set & test_set
    if overlap_tt:
        raise SplitManifestError(
            f"sample leakage detected: {len(overlap_tt)} samples in both train and test"
        )
    overlap_vt = val_set & test_set
    if overlap_vt:
        raise SplitManifestError(
            f"sample leakage detected: {len(overlap_vt)} samples in both validation and test"
        )


def load_scared_split_manifest(path: Path) -> SplitManifest:
    """Load and validate a SCARED split manifest from a JSON file."""
    manifest = load_split_manifest(path)
    validate_scared_split_manifest(manifest)
    return manifest


def write_scared_split_manifest(manifest: SplitManifest, path: Path) -> None:
    """Write a validated SCARED split manifest to JSON without UTF-8 BOM."""
    validate_scared_split_manifest(manifest)
    payload = {
        "schema_version": manifest.schema_version,
        "dataset": manifest.dataset,
        "dataset_version": manifest.dataset_version,
        "split_name": manifest.split_name,
        "train": list(manifest.train),
        "validation": list(manifest.validation),
        "test": list(manifest.test),
        "notes": manifest.notes,
    }
    canonical = json.dumps(payload, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical, encoding="utf-8")


parse_keyframe_id = parse_scared_keyframe_id