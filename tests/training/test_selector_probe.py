"""Immutable snapshot parsing and gain-fit diagnostics without real datasets."""

import copy
import json
from pathlib import Path

import pytest
import torch

from reliable_endo_gs.config.intervention import load_intervention_config
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES
from reliable_endo_gs.training.metric_quality import metric_targets
from reliable_endo_gs.training.selector_probe import (
    SelectorSnapshots,
    read_selector_snapshots,
    replay_logged_selection,
    run_selector_probe,
    selector_fit_metrics,
)


@pytest.fixture
def config():
    return load_intervention_config(Path("configs/routing/intervention_metric_stereo_gain.yaml"))


def record(config, sid="dataset_1/keyframe_1/frame_data000000", gain=0.2):
    features = [[0.1] * len(BALANCED_FEATURE_NAMES)]
    values = [0.0] * 12
    values[1] = gain
    triples = [(0.0, 0.0, False)] + [metric_targets(values, config, a) for a in (1, 2)]
    actions = [
        {
            "action": name,
            "global_gain": t[0],
            "local_gain": t[1],
            "harm": t[2],
            "metric_changes": values if a else None,
            "selector_label_available": True,
        }
        for a, (name, t) in enumerate(zip(("STOP", "STEREO", "GAUSSIAN"), triples, strict=True))
    ]
    return {
        "event": "sample_progress",
        "sample_id": sid,
        "features_raw": features,
        "interventions": [{"actions": actions}],
        "actual_gains": [[t[0] for t in triples]],
        "actual_harms": [[float(t[2]) for t in triples]],
        "actual_label_available": [[True] * 3],
        "valid_regions": [True],
        "counterfactual_metric_outcomes": [
            {"region_id": 0, "action": name, "metric_changes": values}
            for name in ("STEREO", "GAUSSIAN")
        ],
    }


def write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_reader_uses_last_training_snapshot_and_matches_development(config, tmp_path):
    p = tmp_path / "snapshots.jsonl"
    first, last = record(config, gain=0.1), record(config, gain=0.3)
    write(p, [first, last])
    train = read_selector_snapshots(p, config, training=True)
    assert len(train.sample_ids) == 1
    assert train.global_gain[0, 1] == pytest.approx(last["actual_gains"][0][1])
    write(p, [last])
    dev = read_selector_snapshots(p, config, training=False)
    assert torch.equal(train.global_gain, dev.global_gain)
    assert torch.equal(train.local_gain, dev.local_gain)


def test_reader_drops_latest_unknown_and_never_falls_back_to_old_labels(config, tmp_path):
    p = tmp_path / "snapshots.jsonl"
    old, unknown, valid = (
        record(config),
        record(config),
        record(config, sid="dataset_2/keyframe_1/frame_data000001"),
    )
    unknown["interventions"][0]["actions"][1].update(
        selector_label_available=False, global_gain=None, local_gain=None, harm=None
    )
    write(p, [old, unknown, valid])
    table = read_selector_snapshots(p, config, training=True)
    assert table.sample_ids == (valid["sample_id"],)
    assert table.excluded_unknown_regions == 1


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("bad", ["identity", "label", "features", "stop"])
def test_reader_rejects_forbidden_ids_wrong_labels_and_invalid_inputs(
    config, tmp_path, training, bad
):
    row = record(config)
    if bad == "identity":
        row["sample_id"] = "dataset_9/keyframe_1/frame_data000000"
    elif bad == "features":
        row["features_raw"][0][0] = float("nan")
    elif bad == "stop":
        row["interventions"][0]["actions"][0]["global_gain"] = 0.1
        row["actual_gains"][0][0] = 0.1
    else:
        row["interventions"][0]["actions"][1]["global_gain"] = 999
        row["actual_gains"][0][1] = 999
    p = tmp_path / "bad.jsonl"
    write(p, [row])
    with pytest.raises(ValueError):
        read_selector_snapshots(p, config, training=training)


def test_development_duplicate_is_rejected(config, tmp_path):
    p = tmp_path / "duplicates.jsonl"
    row = record(config)
    write(p, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        read_selector_snapshots(p, config, training=False)


def test_fit_metrics_are_read_only_and_isolate_true_safe_gain_ranking(config):
    head = BalancedSelector(8, config).eval()
    table = SelectorSnapshots(
        torch.zeros(2, len(BALANCED_FEATURE_NAMES)),
        torch.tensor([[0.0, 0.2, 0.1], [0.0, 0.4, 0.1]]),
        torch.zeros(2, 3),
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ("train1", "train2"),
        0,
    )
    before = copy.deepcopy(head.state_dict())
    result = selector_fit_metrics(head, table)
    assert result["stereo_best_regions"] == 1
    assert result["stereo_best_correct"] == 0
    assert all(torch.equal(v, head.state_dict()[k]) for k, v in before.items())
    assert all(p.grad is None for p in head.parameters())


def test_replay_keeps_unknown_gt_regions_eligible(config, tmp_path):
    head = BalancedSelector(8, config).eval()
    with torch.no_grad():
        head.gain_network[-1].bias[:2].copy_(torch.tensor([2.0, 1.0]))
    row = record(config)
    row["remaining_ms"] = 200
    row["actual_label_available"][0][1] = False
    p = tmp_path / "unknown.jsonl"
    write(p, [row])
    result = replay_logged_selection(head, p, torch.tensor([0.0, 100.0, 50.0]))
    assert result["actions"] == {"STEREO": 1}
    assert result["unknown_interventions"] == 1
    assert result["known_mean_utility"] is None


def test_probe_refuses_existing_outputs_before_loading_any_artifact(tmp_path):
    with pytest.raises(ValueError, match="new external"):
        run_selector_probe(pilot_dir=tmp_path / "missing", output_dir=tmp_path)


def test_probe_cli_help():
    from reliable_endo_gs.cli.selector_probe import main

    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
