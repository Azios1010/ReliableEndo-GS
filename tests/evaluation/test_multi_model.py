"""Resumable row loop and the disparity-shift helpers (CPU, no dataset)."""

import json

import pytest
import torch

from reliable_endo_gs.evaluation.disparity_shift import SHIFTS, shift_slope, shift_stats
from reliable_endo_gs.evaluation.multi_model import finished_ids, run_rows


def _maker(calls, fail_on=None):
    def make(sample_id):
        calls.append(sample_id)
        if sample_id == fail_on:
            raise OSError("simulated crash")
        return {
            "sample_id": sample_id,
            "metrics": {"x": {"value": 1.0, "valid_count": 1, "definition": "d"}},
        }

    return make


def test_run_rows_resumes_without_repeating_or_reordering(tmp_path):
    ids = [f"dataset_8/keyframe_0/frame_data{k:06d}" for k in range(6)]
    path = tmp_path / "rows.jsonl"
    calls: list[str] = []
    with pytest.raises(OSError):
        run_rows(ids, path, _maker(calls, fail_on=ids[3]))
    assert finished_ids(path) == set(ids[:3])
    more: list[str] = []
    assert run_rows(ids, path, _maker(more)) == 3 and more == ids[3:]
    assert [json.loads(line)["sample_id"] for line in path.read_text().splitlines()] == ids
    assert run_rows(ids, path, _maker([])) == 0  # nothing left


def test_run_rows_rejects_a_row_with_the_wrong_id(tmp_path):
    with pytest.raises(ValueError, match="differs"):
        run_rows(["a"], tmp_path / "r.jsonl", lambda sid: {"sample_id": "b", "metrics": {}})


def test_shift_stats_and_slope_distinguish_following_from_collapse():
    follow, collapsed = [], []
    for shift in SHIFTS:
        truth = 40.0 - shift
        follow.append({"shift": shift, "n": 10, "mean_pred": truth + 0.5, "mean_gt": truth})
        collapsed.append({"shift": shift, "n": 10, "mean_pred": 45.0, "mean_gt": truth})
    assert (
        shift_slope(follow)["reading"] == "follows the shift"
        and abs(shift_slope(follow)["slope"] - 1.0) < 1e-6
    )
    assert shift_slope(collapsed)["reading"] == "insensitive"
    pred = torch.full((1, 1, 2, 3), 5.0)
    stats = shift_stats(
        pred,
        torch.full_like(pred, 4.0),
        torch.ones_like(pred, dtype=torch.bool),
        torch.ones_like(pred, dtype=torch.bool),
    )
    assert stats["n"] == 6 and stats["bias"] == 1.0 and stats["epe"] == 1.0
    assert (
        shift_stats(
            pred,
            pred,
            torch.zeros_like(pred, dtype=torch.bool),
            torch.ones_like(pred, dtype=torch.bool),
        )
        is None
    )
