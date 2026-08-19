"""Immutable upstream identity and provenance serialization tests."""

import pytest

from reliable_endo_gs.baseline.provenance import BaselineProvenance
from reliable_endo_gs.baseline.upstream import PINNED_COMMIT, UPSTREAM_PROJECT, UPSTREAM_REPOSITORY


def test_unpatched_provenance_serializes_exact_sha() -> None:
    provenance = BaselineProvenance(
        upstream_project=UPSTREAM_PROJECT,
        upstream_repository=UPSTREAM_REPOSITORY,
        upstream_commit=PINNED_COMMIT,
        integration_strategy="git-submodule",
        patched=False,
        upstream_dirty=False,
        reliable_endo_gs_commit="3e181c7a00000000000000000000000000000000",
    )

    serialized = provenance.to_dict()
    assert serialized["upstream_commit"] == PINNED_COMMIT
    assert serialized["integration_strategy"] == "git-submodule"
    assert serialized["patched"] is False
    assert serialized["patch_ids"] == []
    assert serialized["checkpoint_id"] is None


def test_provenance_rejects_moving_revision_and_invented_checkpoint_identity() -> None:
    with pytest.raises(ValueError, match="40-character"):
        BaselineProvenance(
            upstream_project=UPSTREAM_PROJECT,
            upstream_repository=UPSTREAM_REPOSITORY,
            upstream_commit="main",
            integration_strategy="git-submodule",
            patched=False,
            upstream_dirty=False,
            reliable_endo_gs_commit=None,
        )
    with pytest.raises(ValueError, match="present together"):
        BaselineProvenance(
            upstream_project=UPSTREAM_PROJECT,
            upstream_repository=UPSTREAM_REPOSITORY,
            upstream_commit=PINNED_COMMIT,
            integration_strategy="git-submodule",
            patched=False,
            upstream_dirty=False,
            reliable_endo_gs_commit=None,
            checkpoint_id="unverified",
        )
