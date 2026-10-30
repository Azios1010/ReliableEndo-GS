"""Frozen protocol corrections, shared-store fitting and fail-closed evidence gates.

All outcomes here are synthetic contracts, not research performance evidence.
"""

import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from reliable_endo_gs.config.intervention import load_intervention_config
from reliable_endo_gs.config.selector_refit import SelectorRefitConfig, partition_selector_training
from reliable_endo_gs.config.selector_study import SelectorStudyConfig
from reliable_endo_gs.data.selector_cohort import (
    temporal_keyframe_cohort,
    validate_selector_worklist,
)
from reliable_endo_gs.evaluation.selector_evidence import (
    keyframe_interval,
    oracle_gate,
    outcome_distribution,
    stage_a_gate,
)
from reliable_endo_gs.evaluation.selector_policy import replay_policy
from reliable_endo_gs.evaluation.selector_promotion import evaluate_selector_promotion
from reliable_endo_gs.routing.calibrated_selector import load_calibrated_selector
from reliable_endo_gs.routing.masks import MaskProposalConfig
from reliable_endo_gs.routing.repairability import select_repairability
from reliable_endo_gs.routing.support_grid import AutomaticSupportGrid, SupportGridConfig
from reliable_endo_gs.routing.tokens import selector_feature_names
from reliable_endo_gs.training.metric_quality import metric_targets
from reliable_endo_gs.training.selector_operating_points import calibrate_operating_points
from reliable_endo_gs.training.selector_outcomes import (
    collect_selector_outcomes,
    load_selector_outcomes,
    project_selector_outcomes,
)
from reliable_endo_gs.training.selector_shards import (
    collect_selector_shards,
    load_verified_selector_shards,
)
from reliable_endo_gs.training.selector_study import (
    run_selector_study,
    selector_variant,
    verify_gate_a,
)
from reliable_endo_gs.training.student_checkpoint import (
    import_frozen_repair_artifact,
    import_frozen_student_actions,
    load_student_checkpoint,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_learning import StudentLearner
from tests.routing.test_balanced import setup as balanced_fixture
from tests.routing.test_metric_joint import metric_learner

setup = balanced_fixture


def config():
    return load_intervention_config(
        Path("configs/routing/intervention_metric_geometry_frozen_student.yaml")
    )


def ids():
    return tuple(
        f"dataset_{d}/keyframe_{k}/frame_data{i:06d}"
        for d in (1, 2, 3, 6)
        for k in range(1, 4 if d == 1 else 5)
        for i in range(8)
    )


def test_actual_subset_and_independent_partition_seed():
    train = ids()
    dev = tuple(f"dataset_7/keyframe_{k}/frame_data000001" for k in range(1, 5))
    policy = SelectorRefitConfig(
        partition_seed=0,
        fit_seed=0,
        partition_policy="stratified_keyframe_hash",
        max_training_samples=200,
    )
    work = {
        "schema": "selector_cohort_worklist.v1",
        "training": list(train),
        "development": list(dev),
    }
    assert validate_selector_worklist(work, set(train), set(dev), policy) == (train, dev)
    a = partition_selector_training(train, policy)
    assert [len(a[n + "_keyframes"]) for n in ("fit", "validation", "calibration")] == [7, 4, 4]
    assert partition_selector_training(train, replace(policy, fit_seed=2, seed=999)) == a
    assert partition_selector_training(train, replace(policy, partition_seed=1)) != a
    bad = {**work, "training": [s for s in train if not s.startswith("dataset_6/")]}
    with pytest.raises(ValueError, match="training"):
        validate_selector_worklist(bad, set(train), set(dev), policy)
    selected = temporal_keyframe_cohort(train, 3)
    assert selected == temporal_keyframe_cohort(tuple(reversed(train)), 3)
    assert len(selected) == 15 * 3 and len(set(selected)) == len(selected)


def test_automatic_grid_covers_both_spatial_axes_preserves_holes_and_bounds():
    h, w = 16, 24
    support = torch.ones(1, 1, h, w, dtype=torch.bool)
    support[..., 3, 5] = False
    state = SimpleNamespace(
        batch=SimpleNamespace(left=torch.zeros(1, 3, h, w)),
        support=support,
        base=SimpleNamespace(
            prior=SimpleNamespace(source_index=torch.arange(h * w)[None]),
            gaussians=SimpleNamespace(valid_mask=torch.ones(1, h * w, dtype=torch.bool)),
        ),
    )
    limits = MaskProposalConfig(max_regions=4, max_pixels=16, max_primitives=16, max_read_pixels=36)
    producer = AutomaticSupportGrid(SupportGridConfig(tile_hw=(2, 2)), limits)
    regions = producer.propose(state, halo=1)
    assert regions == producer.propose(state, halo=1)
    assert len(regions) == 4 and sum(r.pixel_count for r in regions) <= 16
    centres = [((r.core[0] + r.core[2]) / 2, (r.core[1] + r.core[3]) / 2) for r in regions]
    assert {(y < h / 2, x < w / 2) for y, x in centres} == {
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    }
    for r in regions:
        assert not bool((r.mask(state.batch.left) & ~support).any())
    support.zero_()
    assert producer.propose(state, halo=1) == ()


def test_render_reward_uses_both_views_guards_and_cost_exactly_once():
    cfg = replace(config(), metric_utility_mode="render_psnr", cost_weight_per_ms=0.01)
    changes = [0.0] * 12
    changes[5:7] = [0.2, 0.4]
    assert metric_targets(changes, cfg, 1)[0] == pytest.approx(0.3)
    assert metric_targets(changes, cfg, 2)[0] == pytest.approx(0.3)
    changes[3] = -1
    assert metric_targets(changes, cfg, 1)[2] is True
    assert replace(cfg, cost_weight_per_ms=100).config_hash != cfg.config_hash
    assert metric_targets(changes, replace(cfg, cost_weight_per_ms=100), 1)[0] == pytest.approx(0.3)


def test_read_only_import_allows_selector_change_but_never_action_change_or_resume(setup, tmp_path):
    old, _, _, _ = metric_learner(setup)
    parent = StudentLearner(
        old.model,
        old.config,
        old.routing,
        intervention_config=replace(config(), search_radius_px=4),
    )
    path = tmp_path / "stage-a.pt"
    save_student_checkpoint(
        path,
        parent,
        {"run_kind": "full_training", "source_sha256": "synthetic_fixture_source"},
        {"epoch": 3, "next_sample_index": 0, "samples_seen": 10, "optimizer_steps": 10},
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg = selector_variant(parent.intervention_config, "C1", 8, "render_psnr")
    limits = MaskProposalConfig(max_pixels=32, max_primitives=128)
    target = AutomaticSupportGrid(SupportGridConfig(tile_hw=(3, 3)), limits)
    child = StudentLearner(
        copy.deepcopy(parent.model),
        parent.config,
        parent.routing,
        intervention_config=cfg,
        mask_config=limits,
        target_masks=target,
    )
    with pytest.raises(ValueError, match="protocol"):
        import_frozen_student_actions(path, child, expected_sha256=digest)
    result = import_frozen_repair_artifact(path, child, expected_sha256=digest)
    assert result["selector_imported"] is False and result["checkpoint_modified"] is False
    assert parent.operative_action_identity() == child.operative_action_identity()
    assert parent.frozen_action_identity() != child.frozen_action_identity()
    child.intervention_config = replace(cfg, search_radius_px=8)
    with pytest.raises(ValueError, match="execution"):
        import_frozen_repair_artifact(path, child, expected_sha256=digest)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_real_feature_bank_precedes_gt_and_preserves_frozen_states(setup, tmp_path):
    old, sample, _, truth = metric_learner(setup)
    cfg = replace(config(), search_radius_px=4)
    limits = MaskProposalConfig(max_regions=2, max_pixels=32, max_primitives=128)
    producer = AutomaticSupportGrid(SupportGridConfig(tile_hw=(3, 3)), limits)
    learner = StudentLearner(
        old.model,
        old.config,
        old.routing,
        intervention_config=cfg,
        mask_config=limits,
        target_masks=producer,
    )
    before = learner.operative_action_identity()
    called = []
    original = learner.selection_features

    def features(*args):
        called.append("features")
        return original(*args)

    learner.selection_features = features

    def gt(sid):
        assert called == ["features"] and sample.supervision is None
        return truth

    provenance = {
        k: "identity"
        for k in (
            "checkpoint_sha256",
            "checkpoint_source_sha256",
            "source_sha256",
            "training_manifest_hash",
            "development_manifest_hash",
            "role_hash",
            "target_mask_index_sha256",
        )
    }
    collect_selector_outcomes(
        learner,
        [sample],
        gt,
        collection_role="train",
        provenance=provenance,
        output_dir=tmp_path / "store",
        store_feature_bank=True,
    )
    table = load_selector_outcomes(tmp_path / "store", "train")
    assert len(table.metadata["feature_bank_names"]) == 61 + 3 * old.config.width
    assert learner.operative_action_identity() == before
    collect_selector_shards(
        learner,
        (sample.sample_id,),
        lambda sid: sample,
        lambda sid: replace(truth, valid_mask=torch.zeros_like(truth.valid_mask)),
        provenance=provenance,
        output_dir=tmp_path / "unknown-store",
        store_feature_bank=True,
    )
    unknown = load_verified_selector_shards(tmp_path / "unknown-store")
    assert not unknown.sample_ids and unknown.observation_summary()["unknown_regions"] > 0
    checkpoint = tmp_path / "automatic-initial.pt"
    cursor = {"epoch": 0, "next_sample_index": 0, "samples_seen": 0, "optimizer_steps": 0}
    save_student_checkpoint(checkpoint, learner, {}, cursor)
    assert load_student_checkpoint(checkpoint, learner, {}, 10) == cursor
    c1 = project_selector_outcomes(table, selector_variant(cfg, "C1", old.config.width, "geometry"))
    assert c1.features.shape[1] == 85
    assert torch.equal(table.global_gain, c1.global_gain)
    assert torch.equal(table.features, c1.features[:, :61])
    with pytest.raises(ValueError, match="lacks"):
        project_selector_outcomes(table, selector_variant(cfg, "C1", 64, "geometry"))


def synthetic_store(tmp_path, *, heterogeneous=True):
    root = tmp_path / "collection"
    store = root / "train_outcomes"
    shard = store / "shard-000000"
    shard.mkdir(parents=True)
    study = SelectorStudyConfig(
        utility_modes=("render_psnr",), bootstrap_repetitions=100, cutoff_candidates=(0.2, 0.95)
    )
    policy = SelectorRefitConfig(
        partition_seed=0,
        fit_seed=0,
        partition_policy="stratified_keyframe_hash",
        max_training_samples=200,
        max_steps=2,
        min_steps=1,
        validate_every=1,
        patience=1,
    )
    cfg, sample_ids = config(), ids()
    partition = partition_selector_training(sample_ids, policy)
    rows = []
    for sid in sample_ids:
        i = int(sid.rsplit("frame_data", 1)[1])
        actions = [
            {
                "action": "STOP",
                "global_gain": 0.0,
                "local_gain": 0.0,
                "harm": False,
                "metric_changes": None,
                "selector_label_available": True,
                "cost_ms": 0.0,
            }
        ]
        for ai, action in ((1, "STEREO"), (2, "GAUSSIAN")):
            changes = [0.0] * 12
            win = (i % 2 == ai - 1) if heterogeneous else ai == 1
            if win:
                changes[5:7] = [0.2, 0.2]
            else:
                changes[10] = -0.1
            gain, local, harm = metric_targets(changes, cfg, ai)
            actions.append(
                {
                    "action": action,
                    "global_gain": gain,
                    "local_gain": local,
                    "harm": harm,
                    "metric_changes": changes,
                    "selector_label_available": True,
                    "cost_ms": 1.0,
                }
            )
        stats = [0.0] * 61
        stats[0] = float(i % 2)
        rows.append(
            {
                "sample_id": sid,
                "features_raw": [stats],
                "feature_bank_raw": [stats + [float(i % 2)] * 12],
                "valid_regions": [True],
                "interventions": [{"actions": actions}],
            }
        )
    content = "".join(json.dumps(r) + "\n" for r in rows)
    meta = {
        "schema": "frozen_selector_outcomes.v1",
        "collection_role": "train",
        "sample_ids": list(sample_ids),
        "action_identity": {"legacy": "same"},
        "operative_action_identity": {"schema": "operative_repair_identity.v1", "weights": "same"},
        "candidate_protocol_identity": {
            "schema": "repair_candidate_protocol.v1",
            "regions": "same",
        },
        "intervention_config": cfg.to_dict(),
        "feature_names": list(selector_feature_names(cfg)),
        "feature_bank_names": list(
            selector_feature_names(selector_variant(cfg, "C1", 4, "geometry"))
        ),
        "student_feature_channels": 4,
        "selector_width": 8,
        "deployment_costs_ms": [0.0, 1.0, 1.0],
        "source_sha256": "collection_source",
        "checkpoint_sha256": "a" * 64,
        "target_mask_index_sha256": "candidate",
        "study_config_hash": study.config_hash,
        "partition_sha256": partition["partition_sha256"],
        "weights_unchanged": True,
        "selector_executed": False,
        "teacher_inference_executed": False,
        "samples_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    for directory in (store, shard):
        (directory / "samples.jsonl").write_text(content)
        m = {
            **meta,
            **(
                {"shards": [{"path": "shard-000000", "samples_sha256": meta["samples_sha256"]}]}
                if directory == store
                else {}
            ),
        }
        (directory / "result.json").write_text(json.dumps(m))
    (root / "partition.json").write_text(json.dumps(partition))
    (root / "preflight.json").write_text(
        json.dumps(
            {"study_config_hash": study.config_hash, "refit_config_hash": policy.config_hash}
        )
    )
    sp, pp = tmp_path / "study.yaml", tmp_path / "policy.yaml"
    sp.write_text(yaml.safe_dump(json.loads(json.dumps(asdict(study)))))
    pp.write_text(yaml.safe_dump(asdict(policy)))
    gate = tmp_path / "gate_a.json"
    gate.write_text(
        json.dumps(
            {
                "schema": "frozen_repair_gate_a.v1",
                "status": "pass",
                "checkpoint_sha256": "a" * 64,
                "operative_action_identity": meta["operative_action_identity"],
                "target_mask_index_sha256": "candidate",
                "final_test_executed": False,
                "bootstrap_unit": "keyframe",
                "candidate_protocol_identity": meta["candidate_protocol_identity"],
                "evidence_samples_sha256": "b" * 64,
                "actions": {
                    "fixed_stereo": {
                        "pass": True,
                        "violation_fraction": 0.0,
                        "unknown_interventions": 0,
                        "left_psnr_gain_db": {
                            "keyframes": 4,
                            "ci95": [0.1, 0.2],
                            "by_keyframe": {f"dataset_7/keyframe_{k}": 0.1 for k in range(1, 5)},
                        },
                    }
                },
            }
        )
    )
    for directory in (store, shard):
        m = json.loads((directory / "result.json").read_text())
        m["gate_a_report_sha256"] = hashlib.sha256(gate.read_bytes()).hexdigest()
        (directory / "result.json").write_text(json.dumps(m))
    return root, study, policy, sp, pp, gate


def test_shared_store_projection_unknowns_audit_and_keyframe_gate(tmp_path):
    root, study, policy, _, _, _ = synthetic_store(tmp_path)
    table = load_verified_selector_shards(root / "train_outcomes")
    cfg = selector_variant(config(), "C1", 4, "render_psnr")
    projected = project_selector_outcomes(table, cfg)
    assert projected.features.shape[1] == 73 and projected.risk_components.shape[-1] == 2
    partition = partition_selector_training(ids(), policy)
    gate = oracle_gate(projected.view(tuple(partition["fit_sample_ids"])), cfg, study)
    assert gate["status"] == "pass"
    assert all(len(v) == 7 for v in gate["unique_win_keyframes"].values())
    assert outcome_distribution(projected, partition)["unknown_is_safe"] is False
    assert (
        keyframe_interval(
            [(f"dataset_1/keyframe_{k}/f{i}", float(k)) for k in (1, 2) for i in range(30)],
            repetitions=100,
            seed=0,
        )["keyframes"]
        == 2
    )
    # Unknown action labels stay in records; fixed decisions do not inspect them.
    rec = projected.records[0]["interventions"][0]["actions"][1]
    rec.update(
        global_gain=None,
        local_gain=None,
        harm=None,
        metric_changes=None,
        selector_label_available=False,
    )
    selected = replay_policy(
        projected, cfg, "fixed_stereo", repair_budget_ms=study.repair_budget_ms
    )
    assert selected[0]["known"] is False and selected[0]["harm"] is None


def test_no_go_never_constructs_or_fits_a_selector(tmp_path, monkeypatch):
    root, _, _, sp, pp, gate = synthetic_store(tmp_path, heterogeneous=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("fitting must not run after a failed gate")

    monkeypatch.setattr("reliable_endo_gs.training.selector_study.fit_selector_heads", forbidden)
    result = run_selector_study(
        collection_dir=root,
        study_protocol_path=sp,
        refit_config_path=pp,
        output_dir=tmp_path / "no-go",
        gate_a_report=gate,
    )
    assert result["status"] == "oracle_gate_no_go" and not list((tmp_path / "no-go").glob("*.pt"))
    with pytest.raises(ValueError, match="Gate A|gate-a-report"):
        verify_gate_a(None, "a" * 64, {}, "candidate")


def test_collect_once_fit_many_artifacts_bind_same_partition_and_actors(tmp_path):
    root, study, _, sp, pp, gate = synthetic_store(tmp_path)
    outcome_path = root / "train_outcomes/samples.jsonl"
    before = hashlib.sha256(outcome_path.read_bytes()).hexdigest()
    rng = torch.get_rng_state().clone()
    out = tmp_path / "fits"
    result = run_selector_study(
        collection_dir=root,
        study_protocol_path=sp,
        refit_config_path=pp,
        output_dir=out,
        gate_a_report=gate,
    )
    assert result["selector_fitted"] and len(list(out.glob("*.pt"))) == 3 * 3 * 4
    assert torch.equal(rng, torch.get_rng_state())
    partitions, actors = [], []
    for path in out.glob("*.pt"):
        raw = torch.load(path, weights_only=True)
        partitions.append(raw["partition"])
        actors.append(raw["action_identity"])
        model = load_calibrated_selector(
            path,
            expected_action_identity=raw["action_identity"],
            expected_config=type(config())(**raw["decision_config"]),
            expected_mask_identity="candidate",
            expected_source_sha256=result["source_sha256"],
        )
        assert model.operating_point_available == raw["operating_point_available"]
    assert all(p == partitions[0] for p in partitions)
    assert all(a == actors[0] for a in actors)
    assert hashlib.sha256(outcome_path.read_bytes()).hexdigest() == before
    # Promotion must use all predeclared seeds of the chosen variant. A single
    # favorable development seed cannot promote the selector.
    mapping = {}
    for choice in result["choices_before_development"]:
        for entry in result["comparisons"]:
            if entry["variant"] != choice["variant"] or entry["budget"] != choice["budget"]:
                continue
            directory = tmp_path / ("native-" + entry["artifact"])
            directory.mkdir()
            gain = 0.1 if entry["seed"] == 0 else 0.01
            records = []
            for k in range(1, 5):
                policies = {}
                for name, g, action in (
                    ("base", 0.0, "STOP"),
                    ("fixed_stereo", 0.02, "STEREO"),
                    ("fixed_gaussian", 0.03, "GAUSSIAN"),
                    ("selector", gain, "STEREO"),
                ):
                    policies[name] = {
                        "action": action,
                        "gain": g,
                        "harm": False,
                        "metric_feasible": True,
                        "estimated_action_ms": 0.0 if action == "STOP" else 1.0,
                        "metrics": {
                            m: {"value": 20.0 + g}
                            for m in ("left_psnr", "right_psnr", "left_ssim", "right_ssim")
                        },
                    }
                records.append(
                    {"sample_id": f"dataset_7/keyframe_{k}/frame_data000001", "policies": policies}
                )
            content = "".join(json.dumps(r) + "\n" for r in records)
            (directory / "samples.jsonl").write_text(content)
            meta = {
                "weights_unchanged": True,
                "samples_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "selector_refit_sha256": entry["artifact_sha256"],
                "operative_action_identity": result["operative_action_identity"],
                "target_mask_index_sha256": "candidate",
                "study_config_hash": study.config_hash,
                "source_sha256": result["source_sha256"],
            }
            (directory / "result.json").write_text(json.dumps(meta))
            mapping[entry["artifact"]] = str(directory)
    index = tmp_path / "evaluations.json"
    index.write_text(json.dumps(mapping))
    promotion = evaluate_selector_promotion(
        study_dir=out, evaluation_index=index, output_dir=tmp_path / "promotion"
    )
    assert all(r["status"] == "no_go" and r["passing_seeds"] == 1 for r in promotion["comparisons"])
    assert promotion["paper_claim_ready"] is False


def test_cal_cutoff_selection_respects_harm_budget_and_abstains_on_unknown(tmp_path):
    root, study, policy, _, _, _ = synthetic_store(tmp_path)
    cfg = selector_variant(config(), "C0", 4, "render_psnr")
    projected = project_selector_outcomes(
        load_verified_selector_shards(root / "train_outcomes"), cfg
    )
    partition = partition_selector_training(ids(), policy)
    calibration = projected.view(tuple(partition["calibration_sample_ids"]))

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = cfg

        def select(self, features, costs, valid, remaining):
            win = features[:, 0].long() + 1
            gains = features.new_tensor([[0.0, 0.4, 0.4]]).expand(len(features), -1).clone()
            risks = features.new_tensor([[0.0, 0.8, 0.8]]).expand(len(features), -1).clone()
            gains.scatter_(1, win[:, None], 0.2)
            risks.scatter_(1, win[:, None], 0.05)
            return select_repairability(gains, risks, costs, valid, remaining, self.config)

    model = Predictor()
    points = calibrate_operating_points(model, calibration, study)
    assert all(p["status"] == "selected" and p["selected"]["cutoff"] == 0.2 for p in points)
    assert all(p["selected"]["selected_harm_rate"] == 0 for p in points)
    assert model.config == cfg
    # Keep deployment decisions fixed while removing their offline observation.
    for r in calibration.records:
        for a in r["interventions"][0]["actions"][1:]:
            a.update(
                global_gain=None, harm=None, metric_changes=None, selector_label_available=False
            )
    assert all(
        p["status"] == "abstain" for p in calibrate_operating_points(model, calibration, study)
    )


def test_gate_a_requires_real_hashed_val7_and_uses_keyframes(tmp_path):
    p = tmp_path / "evaluation"
    p.mkdir()
    rows = []
    for k in range(1, 5):
        base = {"metrics": {"left_psnr": {"value": 20.0}}, "action": "STOP"}
        action = {
            "metrics": {"left_psnr": {"value": 20.1}},
            "action": "STEREO",
            "metric_feasible": True,
        }
        rows.append(
            {
                "sample_id": f"dataset_7/keyframe_{k}/frame_data000000",
                "policies": {
                    "base": base,
                    "fixed_stereo": action,
                    "fixed_gaussian": {**action, "action": "GAUSSIAN"},
                },
            }
        )
    content = "".join(json.dumps(r) + "\n" for r in rows)
    (p / "samples.jsonl").write_text(content)
    (p / "result.json").write_text(
        json.dumps(
            {
                "weights_unchanged": True,
                "samples_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "operative_action_identity": {"actor": "fixed"},
                "checkpoint_sha256": "a" * 64,
                "target_mask_index_sha256": "candidate",
            }
        )
    )
    report = stage_a_gate(p, repetitions=100)
    assert report["status"] == "pass" and report["bootstrap_unit"] == "keyframe"
    (p / "samples.jsonl").write_text(content + "\n")
    with pytest.raises(ValueError, match="verified"):
        stage_a_gate(p)
