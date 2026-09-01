"""Synthetic manifest tests for immutable development artifacts."""

from pathlib import Path

import pytest

from reliable_endo_gs.evaluation import (
    ArtifactPayload,
    load_artifact,
    make_artifact,
    write_artifact,
)


def test_artifact_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    payload = ArtifactPayload("metrics.json", "a" * 64, 12)
    artifact = make_artifact(
        config={"seed": 1},
        config_hash="b" * 64,
        dataset={"name": "synthetic"},
        split={"hash": "split"},
        sample_selection={"ids": ["synthetic/0"]},
        seeds={"global": 1},
        upstream={"commit": "c" * 40},
        checkpoint={"id": None},
        metric_schema={"version": "1"},
        metrics={"epe": {"value": 0.0, "valid_count": 1}},
        profiling={"device": "cpu"},
        payloads=(payload,),
        created_utc="2026-01-01T00:00:00Z",
    )
    path = tmp_path / "artifact.json"
    write_artifact(path, artifact)
    assert load_artifact(path).artifact_id == artifact.artifact_id
    with pytest.raises(ValueError, match="overwrite"):
        write_artifact(path, artifact)
