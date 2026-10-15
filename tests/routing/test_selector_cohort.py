"""Larger cohort preparation, temporal ROIs, immutable sharding and resume."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from reliable_endo_gs.config.intervention import load_intervention_config
from reliable_endo_gs.config.selector_refit import (
    SelectorRefitConfig,
    load_selector_refit_config,
    partition_selector_training,
)
from reliable_endo_gs.data.selector_cohort import (
    finalize_selector_rois,
    reviewed_roi_box,
    validate_selector_worklist,
    verify_selector_source_snapshot,
)
from reliable_endo_gs.data.target_masks import TargetMaskIndex
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES
from reliable_endo_gs.training.selector_outcomes import SelectorOutcomes, load_selector_outcomes
from reliable_endo_gs.training.selector_refit import _validate_deployment_cohort
from reliable_endo_gs.training.selector_shards import collect_selector_shards


def test_legacy_default_config_identity_preserved():
    assert (
        SelectorRefitConfig().config_hash
        == "059d44399fa1e9672c1e3d3fd6bc2f5d650d2d7328dc781224ad92868e9b719b"
    )
    assert (
        load_selector_refit_config(
            Path("configs/routing/selector_refit_quarter.yaml")
        ).max_training_samples
        == 5738
    )


def test_stratified_partition_covers_all_folds_in_all_three_roles():
    ids = tuple(
        f"dataset_{d}/keyframe_{k}/frame_data{i:06d}"
        for d in range(1, 8)
        for k in range(1, 4)
        for i in range(4)
    )
    p = partition_selector_training(
        ids, replace(SelectorRefitConfig(), partition_policy="stratified_keyframe_hash")
    )
    sets = []
    for name in ("fit", "validation", "calibration"):
        groups = set(p[name + "_keyframes"])
        sets.append(groups)
        assert {k.split("/")[0] for k in groups} == {f"dataset_{d}" for d in range(1, 8)}
    assert not any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2)))
    with pytest.raises(ValueError, match="three keyframes"):
        partition_selector_training(
            tuple(s for s in ids if not s.startswith("dataset_1/keyframe_3")),
            replace(SelectorRefitConfig(), partition_policy="stratified_keyframe_hash"),
        )


def test_larger_worklist_requires_declared_bounds_and_no_overlap():
    train = [f"dataset_{d}/keyframe_1/frame_data{i:06d}" for d in range(1, 8) for i in range(24)]
    dev = [f"dataset_{d}/keyframe_1/frame_data999999" for d in range(1, 8)]
    raw = {"schema": "selector_cohort_worklist.v1", "training": train, "development": dev}
    with pytest.raises(ValueError, match="training"):
        validate_selector_worklist(raw, set(train), set(dev), SelectorRefitConfig())
    policy = replace(SelectorRefitConfig(), max_training_samples=256)
    assert len(validate_selector_worklist(raw, set(train), set(dev), policy)[0]) == 168
    with pytest.raises(ValueError, match="overlaps"):
        validate_selector_worklist(raw, set(train + dev), set(dev), policy)


@pytest.mark.parametrize(
    "values",
    [
        {"partition_policy": "random_labels"},
        {"collection_shard_size": 257},
        {"max_training_samples": 0},
        {"max_development_samples": True},
    ],
)
def test_invalid_collection_controls(values):
    with pytest.raises(ValueError):
        replace(SelectorRefitConfig(), **values)


def test_temporal_roi_assignment_uses_original_frame_ids_and_recorded_anchors():
    sid = "dataset_1/keyframe_1/"
    p = {
        "reviews": {
            "dataset_1/keyframe_1": {
                "sample_ids": [sid + "frame_data000000", sid + "frame_data000100"]
            }
        },
        "bounds_yxyx_by_keyframe": {"dataset_1/keyframe_1": [0, 0, 100, 100]},
        "bounds_by_review_sample": {sid + "frame_data000100": [20, 20, 80, 80]},
    }
    assert reviewed_roi_box(p, sid + "frame_data000050") == [0, 0, 100, 100]
    assert reviewed_roi_box(p, sid + "frame_data000051") == [20, 20, 80, 80]


def test_bounded_preflight_checks_every_deployment_member_and_propagates_failure():
    checked = []
    mask_ids = []

    def validate(sid, *, data_role):
        assert data_role == "final_train"
        checked.append(sid)

    def load(sid, hw, device):
        assert hw == (512, 640) and device.type == "cpu"
        mask_ids.append(sid)
        if sid == "broken":
            raise ValueError("mask invalid")

    adapter = SimpleNamespace(validate_deployment_files=validate)
    masks = SimpleNamespace(load=load)
    ids = tuple(str(i) for i in range(20))
    _validate_deployment_cohort(adapter, masks, ids, (512, 640), "train")
    assert sorted(checked) == sorted(mask_ids) == sorted(ids)
    checked.clear()
    mask_ids.clear()
    _validate_deployment_cohort(adapter, masks, ids, (512, 640), "train", reuse_rgb_validation=True)
    assert not checked and sorted(mask_ids) == sorted(ids)
    with pytest.raises(ValueError, match="mask invalid"):
        _validate_deployment_cohort(adapter, masks, ("broken",), (512, 640), "train")


def test_outcome_views_keep_original_row_order_and_unknown_records():
    ids = tuple(f"dataset_1/keyframe_1/frame_data{i:06d}" for i in range(3))
    x = torch.arange(4).reshape(2, 2).float()
    y = torch.zeros(2, 3)
    table = SelectorOutcomes({}, [{"sample_id": sid} for sid in ids], x, y, y, y, (ids[0], ids[2]))
    view = table.view((ids[2], ids[1], ids[0]))
    assert view.sample_ids == (ids[0], ids[2]) and view.records == table.records
    torch.testing.assert_close(view.features, x)
    with pytest.raises(ValueError, match="no known"):
        table.view((ids[1],))
    with pytest.raises(ValueError, match="unknown sample"):
        table.view((ids[0], ids[0]))


def test_cost_preparation_cli_forwards_stop_flag_and_summarizes_sealed_ids(monkeypatch, capsys):
    from reliable_endo_gs.cli.selector_refit import main

    calls = []
    partition = {"partition_sha256": "sealed"}
    for name in ("fit", "validation", "calibration"):
        partition[name + "_sample_ids"] = ["unprinted-sample-id"] * 100
        partition[name + "_keyframes"] = ["dataset_1/keyframe_1"]

    def run(**kwargs):
        calls.append(kwargs)
        return {"partition": partition, "training_executed": False}

    monkeypatch.setattr("reliable_endo_gs.training.selector_refit.run_selector_refit", run)
    args = []
    for name in (
        "checkpoint",
        "data-root",
        "training-manifest",
        "development-manifest",
        "roles-path",
        "worklist-path",
        "target-mask-index",
        "refit-config-path",
        "output-dir",
    ):
        args.extend(["--" + name, "unused-fixture-path"])
    assert main(args + ["--checkpoint-sha256", "fixture", "--prepare-costs"]) == 0
    output = capsys.readouterr().out
    assert calls[0]["prepare_costs"] is True and calls[0]["resume"] is False
    assert "unprinted-sample-id" not in output
    assert json.loads(output)["partition"]["roles"]["fit"]["samples"] == 100


def test_roi_finalization_requires_review_and_shares_masks_with_source_fingerprints(tmp_path):
    root = tmp_path / "data"
    base = root / "dataset_1/keyframe_1/data"
    for folder in ("left_finalpass", "right_finalpass", "frame_data", "newpram_data", "disparity"):
        (base / folder).mkdir(parents=True)
    (base / "a00_protocol.json").write_text("{}")
    ids = ["dataset_1/keyframe_1/frame_data000000", "dataset_1/keyframe_1/frame_data000001"]
    for sid in ids:
        frame = sid.split("/")[-1]
        for folder in ("left_finalpass", "right_finalpass"):
            Image.new("RGB", (1280, 1024), (180, 100, 100)).save(base / folder / (frame + ".png"))
        for folder, suffix in (
            ("frame_data", ".json"),
            ("newpram_data", ".json"),
            ("disparity", ".tiff"),
        ):
            (base / folder / (frame + suffix)).write_bytes(b"unit fixture only")
    out = tmp_path / "assets"
    out.mkdir()
    (out / "sheet.png").write_bytes(b"fixture review sheet")
    policy = {
        "schema": "selector_rgb_roi_review.v1",
        "bounds_yxyx_by_keyframe": {"dataset_1/keyframe_1": [0, 0, 64, 64]},
        "reviews": {
            "dataset_1/keyframe_1": {
                "sample_ids": ids,
                "rgb_sha256": {
                    s: hashlib.sha256(
                        (base / "left_finalpass" / (s.split("/")[-1] + ".png")).read_bytes()
                    ).hexdigest()
                    for s in ids
                },
                "sheet": "sheet.png",
                "sheet_sha256": hashlib.sha256(b"fixture review sheet").hexdigest(),
            }
        },
        "reviewed_keyframes": [],
        "ground_truth_used": False,
        "action_outcomes_used": False,
        "output_hw": [512, 640],
        "review_scope": "synthetic fixture",
    }
    (out / "roi-review.json").write_text(json.dumps(policy))
    (out / "worklist-draft.json").write_text(
        json.dumps({"training": [ids[0]], "development": [ids[1]]})
    )
    with pytest.raises(ValueError, match="visual review"):
        finalize_selector_rois(data_root=root, output_dir=out)
    assert not (out / "masks").exists()
    policy["reviewed_keyframes"] = ["dataset_1/keyframe_1"]
    (out / "roi-review.json").write_text(json.dumps(policy))
    report = finalize_selector_rois(data_root=root, output_dir=out)
    assert report["mask_files"] == 1 and report["training_executed"] is False
    index = TargetMaskIndex(out / "target-mask-index.json")
    for sid in ids:
        assert int(index.load(sid, (512, 640), torch.device("cpu")).pixels.sum()) == 4096
    digest = hashlib.sha256((out / "data-source-index.json").read_bytes()).hexdigest()
    verify_selector_source_snapshot(root, out / "data-source-index.json", digest)
    ledger = out / "roi-members.jsonl"
    original = ledger.read_bytes()
    mask_path = next((out / "masks").glob("*.npy"))
    mask_mtime = mask_path.stat().st_mtime_ns
    # A killed writer may leave an incomplete tail; completed rows remain reusable.
    ledger.write_bytes(original + b'{"sample_id":')
    resumed = finalize_selector_rois(data_root=root, output_dir=out, resume=True)
    assert resumed == report
    assert ledger.read_bytes() == original
    assert mask_path.stat().st_mtime_ns == mask_mtime
    assert len(list(out.glob("roi-members-interrupted-*.jsonl"))) == 1
    (base / "disparity/frame_data000001.tiff").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source member changed"):
        verify_selector_source_snapshot(root, out / "data-source-index.json", digest)


def test_sharded_resume_reuses_completed_work_and_preserves_unknown_partial_evidence(
    monkeypatch, tmp_path
):
    cfg = load_intervention_config(Path("configs/routing/intervention_metric_gain_fit.yaml"))
    learner = SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        intervention_config=cfg,
        frozen_action_identity=lambda: {"model": "fixed"},
    )
    ids = tuple(f"dataset_1/keyframe_1/frame_data{i:06d}" for i in range(6))
    calls = []
    fail = [True]

    def collect(learner, samples, gt_provider, *, collection_role, provenance, output_dir):
        output_dir.mkdir()
        chosen = [s.sample_id for s in samples]
        calls.extend(chosen)
        if chosen[0] == ids[2] and fail[0]:
            fail[0] = False
            (output_dir / "samples.jsonl").write_text("partial evidence\n")
            raise RuntimeError("simulated interruption")
        rows = []
        for sid in chosen:
            actions = [
                {
                    "action": name,
                    "global_gain": 0.0,
                    "local_gain": 0.0,
                    "harm": False,
                    "metric_changes": None if i == 0 else [0.0] * 12,
                    "selector_label_available": True,
                    "cost_ms": 0.0,
                }
                for i, name in enumerate(("STOP", "STEREO", "GAUSSIAN"))
            ]
            if sid in ids[:2]:
                for row in actions[1:]:
                    row.update(
                        global_gain=None,
                        local_gain=None,
                        harm=None,
                        metric_changes=None,
                        selector_label_available=False,
                    )
            rows.append(
                {
                    "sample_id": sid,
                    "features_raw": [[0.0] * len(BALANCED_FEATURE_NAMES)],
                    "valid_regions": [True],
                    "interventions": [{"actions": actions}],
                }
            )
        content = "".join(json.dumps(r) + "\n" for r in rows)
        (output_dir / "samples.jsonl").write_bytes(content.encode("utf-8"))
        meta = {
            **provenance,
            "schema": "frozen_selector_outcomes.v1",
            "collection_role": "train",
            "sample_ids": chosen,
            "action_identity": learner.frozen_action_identity(),
            "intervention_config": cfg.to_dict(),
            "feature_names": list(BALANCED_FEATURE_NAMES),
            "samples_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "weights_unchanged": True,
            "teacher_inference_executed": False,
            "selector_executed": False,
        }
        (output_dir / "result.json").write_text(json.dumps(meta))
        return meta

    monkeypatch.setattr(
        "reliable_endo_gs.training.selector_shards.collect_selector_outcomes", collect
    )

    def provider(s):
        return SimpleNamespace(sample_id=s)

    kwargs = dict(provenance={"source_sha256": "source"}, output_dir=tmp_path / "out", shard_size=2)
    with pytest.raises(RuntimeError, match="interruption"):
        collect_selector_shards(learner, ids, provider, lambda s: None, **kwargs)
    first = (kwargs["output_dir"] / "shard-000000/samples.jsonl").stat().st_mtime_ns
    calls.clear()
    r = collect_selector_shards(learner, ids, provider, lambda s: None, resume=True, **kwargs)
    assert r["reused_samples"] == 2 and calls == list(ids[2:])
    assert (kwargs["output_dir"] / "shard-000000/samples.jsonl").stat().st_mtime_ns == first
    assert list(kwargs["output_dir"].glob("shard-000002-interrupted-*"))
    table = load_selector_outcomes(kwargs["output_dir"], "train")
    assert len(table.records) == 6 and len(table.sample_ids) == 4
    calls.clear()
    collect_selector_shards(learner, ids, provider, lambda s: None, resume=True, **kwargs)
    assert not calls
    with pytest.raises(ValueError, match="identity"):
        collect_selector_shards(
            learner,
            ids,
            provider,
            lambda s: None,
            resume=True,
            **{**kwargs, "provenance": {"source_sha256": "different"}},
        )
    path = kwargs["output_dir"] / "shard-000004/samples.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="identity"):
        collect_selector_shards(learner, ids, provider, lambda s: None, resume=True, **kwargs)
