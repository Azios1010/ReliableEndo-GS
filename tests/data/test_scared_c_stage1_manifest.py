"""Focused inactive Stage1 split-role and cache-guard checks."""

import json
import os
from pathlib import Path

MANIFEST = Path("splits/scared_c/stage1_mean_v1.json")


def test_stage1_mean_roles_are_sequence_disjoint_and_inactive() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["active"] is False
    roles = payload["roles"]
    assert roles["MEAN_TRAIN"] == ["1_1", "2_2", "3_2"]
    assert roles["MEAN_VALIDATION"] == ["1_2", "3_1"]
    assert roles["STRUCTURAL_DIAGNOSTIC"] == ["2_1"]
    assert roles["UNCERTAINTY_TRAIN"] == ["2_3", "3_3"]
    assert roles["CALIBRATION"] == ["1_3", "2_4", "3_4"]
    assert roles["DEVELOPMENT_TRANSFER"] == ["7_1", "7_2", "7_3", "7_4"]
    assert roles["FINAL_UNTOUCHED"] == ["6_1", "6_2", "6_3", "6_4"]
    counts = payload["corrected_video_frame_counts"]
    assert sum(counts[key] for key in roles["MEAN_TRAIN"]) == 2827
    assert sum(counts[key] for key in roles["MEAN_VALIDATION"]) == 609


def test_stage1_cache_manifest_is_final_and_dataset6_free_when_present() -> None:
    data_root = os.environ.get("RELIABLE_ENDO_DATA_ROOT")
    if not data_root:
        return
    cache_manifest = Path(data_root) / "scared_c/.cache/stage1_scared_c_mean_v1/manifest.json"
    if not cache_manifest.is_file():
        return
    payload = json.loads(cache_manifest.read_text(encoding="utf-8"))
    assert payload["dataset_revision"] == "44baac1187c8729c96db0d1def569bd94c9d9417"
    assert all("dataset_6" not in entry["sample_id"] for entry in payload["entries"])
    assert all("dataset_7" not in entry["sample_id"] for entry in payload["entries"])
