"""Smoke preflight using the synthetic SCARED-C fixture only."""

import json
from pathlib import Path

from scripts.reproduce_baseline import reproduce_baseline
from tests.data.conftest import create_scared_c_keyframe


def test_reproduction_loads_sample_then_refuses_without_checkpoint(
    tmp_path: Path,
) -> None:
    create_scared_c_keyframe(tmp_path / "scared_c")
    config = Path("configs/experiment/p0_baseline_reproduction.yaml")
    result = reproduce_baseline(config, data_root=tmp_path)
    assert result["status"] == "refused"
    assert result["sample_id"] == "scared_c/dataset_1/keyframe_1/reference"
    assert "checkpoint" in str(result["reason"])
    assert result["baseline_inference_capability"] == "BASELINE_INFERENCE_BLOCKED_ON_CHECKPOINT"
    assert result["scientific_status"] == "development_only"
    assert result["scientific_reproduction"] is False
    assert len(str(result["artifact_id"])) == 64
    assert str(tmp_path) not in json.dumps(result)
