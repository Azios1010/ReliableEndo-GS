"""Tests for split validation and deterministic scientific hashing."""

import pytest

from reliable_endo_gs.data.splits import (
    SplitManifestError,
    hash_split_manifest,
    parse_split_manifest,
)


def _manifest(**overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": "scared",
        "dataset_version": None,
        "split_name": "development",
        "train": [],
        "validation": [],
        "test": [],
        "notes": "",
    }
    manifest.update(overrides)
    return manifest


def test_valid_empty_development_manifest() -> None:
    manifest = parse_split_manifest(_manifest())

    assert manifest.train == ()
    assert manifest.validation == ()
    assert manifest.test == ()


def test_overlap_is_rejected() -> None:
    with pytest.raises(SplitManifestError, match="sequence_01 appears in train, test"):
        parse_split_manifest(_manifest(train=["sequence_01"], test=["sequence_01"]))


def test_duplicate_inside_split_is_rejected() -> None:
    with pytest.raises(SplitManifestError, match="duplicate identifier.*train"):
        parse_split_manifest(_manifest(train=["sequence_01", "sequence_01"]))


def test_hash_is_stable_when_id_order_and_notes_change() -> None:
    first = parse_split_manifest(
        _manifest(
            split_name="first-name",
            train=["sequence_02", "sequence_01"],
            validation=["sequence_03"],
            notes="first note",
        )
    )
    second = parse_split_manifest(
        _manifest(
            split_name="renamed",
            train=["sequence_01", "sequence_02"],
            validation=["sequence_03"],
            notes="different note",
        )
    )

    assert hash_split_manifest(first) == hash_split_manifest(second)


def test_hash_changes_when_split_membership_changes() -> None:
    first = parse_split_manifest(_manifest(train=["sequence_01"]))
    second = parse_split_manifest(_manifest(test=["sequence_01"]))

    assert hash_split_manifest(first) != hash_split_manifest(second)
