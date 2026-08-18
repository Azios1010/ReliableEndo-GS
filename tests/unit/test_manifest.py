"""Tests for run manifests and safe Git metadata lookup."""

from datetime import datetime
from pathlib import Path

from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.runtime.manifest import create_run_manifest


def test_manifest_contains_required_fields_and_valid_timestamp() -> None:
    config_hash = "a" * 64
    manifest = create_run_manifest(
        run_id="20260819T120000Z_smoke_aaaaaaaa",
        project_git_commit=None,
        git_dirty=None,
        config_hash=config_hash,
        seed=42,
        timestamp_utc="2026-08-19T12:00:00Z",
        python_version="3.10.0",
    )
    serialized = manifest.to_dict()

    assert {
        "run_id",
        "timestamp_utc",
        "project_git_commit",
        "git_dirty",
        "config_hash",
        "seed",
        "python_version",
    } <= serialized.keys()
    assert serialized["config_hash"] == config_hash
    parsed = datetime.fromisoformat(manifest.timestamp_utc.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_git_lookup_failure_is_safe(tmp_path: Path) -> None:
    assert get_git_commit(tmp_path) is None
    assert is_git_dirty(tmp_path) is None
