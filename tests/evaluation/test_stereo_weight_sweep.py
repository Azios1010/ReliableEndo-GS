"""Counterfactual scores reuse component/metric evidence without executing actions."""

import json

import pytest

from reliable_endo_gs.cli.selection import main
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.evaluation.stereo_weight_sweep import sweep_stereo_weights


@pytest.fixture
def pilot(tmp_path):
    root = tmp_path / "pilot"
    cfg = InterventionConfig(schema="intervention_selection.v2")
    before = dict(
        global_rgb=0.07,
        global_warp=0.1,
        local_rgb=0.07,
        local_warp=0.1,
        outside_rgb=0.07,
        outside_warp=0.1,
    )
    after = {
        "STOP": before,
        "GAUSSIAN": {**before, "global_rgb": 0.063, "local_rgb": 0.063},
        "STEREO": {**before, "global_warp": 0.096, "local_warp": 0.096},
    }
    identity = dict(
        source_sha256="old-source",
        role_hash="role",
        manifest_hash="manifest",
        intervention_config_hash=cfg.config_hash,
    )
    actions = [
        {"action": name, "before": before, "after": after[name]}
        for name in ("STOP", "STEREO", "GAUSSIAN")
    ]
    policies = {}
    for name, action in (
        ("base", "STOP"),
        ("fixed_stereo", "STEREO"),
        ("fixed_gaussian", "GAUSSIAN"),
    ):
        policies[name] = dict(
            action=action,
            region=None if action == "STOP" else 0,
            components_before=before,
            components_after=after[action],
            estimated_action_ms=0 if action == "STOP" else 1,
            metrics={
                "stereo_depth_mae_mm": {"value": 9 if action == "STEREO" else 10},
                "left_psnr": {"value": 20},
                "right_psnr": {"value": 20},
            },
        )
    contents = {
        "quality/result.json": {"config": cfg.to_dict(), "role_hash": "role"},
        "train/run.json": {"identity": identity},
        "train/train.jsonl": {"event": "sample_progress", "interventions": [{"actions": actions}]},
        "evaluation/result.json": {
            "samples": 1,
            "source_sha256": "old-source",
            "config_hash": cfg.config_hash,
            "training_manifest_hash": "manifest",
        },
        "evaluation/samples.jsonl": {
            "sample_id": "dataset_1/keyframe_1/a",
            "remaining_ms": 100,
            "policies": policies,
        },
    }
    for rel, body in contents.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body) + "\n")
    return root


def test_sweep_changes_ranking_without_training_or_log_mutation(pilot, tmp_path):
    before = {p: p.read_bytes() for p in pilot.rglob("*") if p.is_file()}
    result = sweep_stereo_weights(pilot_dir=pilot, output_dir=tmp_path / "sweep")
    a, b, c = result["weights"]
    assert a["development_restricted_safe_actions"] == {"GAUSSIAN": 1}
    assert (
        b["development_restricted_safe_actions"]
        == c["development_restricted_safe_actions"]
        == {"STEREO": 1}
    )
    assert a["development_restricted_metrics"]["stereo_depth_mae_mm"]["macro_mean"] == 10
    assert b["development_restricted_metrics"]["stereo_depth_mae_mm"]["macro_mean"] == 9
    assert not result["training_executed"] and not result["renderer_executed"]
    assert all(p.read_bytes() == body for p, body in before.items())
    with pytest.raises(FileExistsError):
        sweep_stereo_weights(pilot_dir=pilot, output_dir=tmp_path / "sweep")


def test_sweep_harm_is_not_removed_by_increased_utility(pilot, tmp_path):
    path = pilot / "evaluation/samples.jsonl"
    row = json.loads(path.read_text())
    row["policies"]["fixed_stereo"]["components_after"]["local_rgb"] = 0.08
    path.write_text(json.dumps(row) + "\n")
    result = sweep_stereo_weights(pilot_dir=pilot, output_dir=tmp_path / "safe")
    assert all(
        r["development_restricted_safe_actions"] == {"GAUSSIAN": 1} for r in result["weights"]
    )


def test_sweep_cli_and_guards(pilot, tmp_path):
    output = tmp_path / "cli"
    assert main(["weight-sweep", "--pilot-dir", str(pilot), "--output-dir", str(output)]) == 0
    for weights in ((0.0,), (float("inf"),), (1.0, 1.0)):
        with pytest.raises(ValueError):
            sweep_stereo_weights(pilot_dir=pilot, output_dir=tmp_path / "bad", weights=weights)
    with pytest.raises(ValueError):
        sweep_stereo_weights(pilot_dir=pilot, output_dir=pilot / "bad")
    path = pilot / "evaluation/result.json"
    data = json.loads(path.read_text())
    data["source_sha256"] = "different-source"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity"):
        sweep_stereo_weights(pilot_dir=pilot, output_dir=tmp_path / "mismatch")
