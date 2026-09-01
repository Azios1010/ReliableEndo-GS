"""Grouped split validation without fabricating a local final split."""

import pytest

from reliable_endo_gs.data.splits import (
    SplitManifest,
    SplitManifestError,
    validate_grouped_membership,
    validate_sequence_groups,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        schema_version=1,
        dataset="scared_c",
        dataset_version="development",
        split_name="grouped_v1",
        train=("dataset_1",),
        validation=(),
        test=(),
        notes="synthetic schema test",
    )


def test_grouped_membership_accepts_samples_from_one_sequence() -> None:
    validate_grouped_membership(
        _manifest(),
        {
            "scared_c/dataset_1/keyframe_1/reference": "dataset_1",
            "scared_c/dataset_1/keyframe_2/reference": "dataset_1",
        },
    )


def test_grouped_membership_rejects_unknown_sequence() -> None:
    with pytest.raises(SplitManifestError, match="absent from manifest"):
        validate_grouped_membership(
            _manifest(), {"scared_c/dataset_2/keyframe_1/reference": "dataset_2"}
        )


def test_sequence_groups_reject_cross_split_assignment() -> None:
    with pytest.raises(SplitManifestError, match="crosses"):
        validate_sequence_groups(
            {"sample-a": "train", "sample-b": "validation"},
            {"sample-a": "sequence-a", "sample-b": "sequence-a"},
        )
