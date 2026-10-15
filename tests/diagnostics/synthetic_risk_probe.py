"""Bounded CPU diagnostic: python -m tests.diagnostics.synthetic_risk_probe --output PATH.

Uses analytic 5x5 fixtures, not medical data or a native renderer. Snapshot
normalization stays fixed at 32. Policy thresholds and weights are unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import tempfile
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import torch

from reliable_endo_gs.actions.repair import Action
from reliable_endo_gs.evaluation.gain_events import gain_event_metrics
from reliable_endo_gs.evaluation.risk_events import risk_event_metrics
from reliable_endo_gs.routing.repairability import select_repairability
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.training.student_checkpoint import (
    initialize_student_model,
    save_student_checkpoint,
)
from reliable_endo_gs.training.student_learning import StudentLearner, evaluation_mode
from reliable_endo_gs.training.student_runtime import _state_equal
from tests.routing.test_balanced import setup as balanced_setup
from tests.routing.test_mask_tokens_risk import learner_inputs


def _state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _source_hash(repo: Path) -> str:
    digest = hashlib.sha256()
    for root in (repo / "src", repo / "tests", repo / "configs"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in (".py", ".yaml", ".yml"):
                continue
            digest.update(path.relative_to(repo).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _variant(sample, truth, index):
    # Predetermined synthetic photometric/geometry variations, shared by both
    # modes. Evaluation uses new index pairs; no threshold fitting occurs.
    brightness = 0.7 + 0.6 * ((index * 7) % 17) / 16
    depth_factor = 0.85 + 0.3 * ((index * 5) % 13) / 12
    batch = replace(
        sample.batch,
        left=sample.batch.left * brightness,
        right=sample.batch.right * brightness,
    )
    return replace(sample, batch=batch), replace(truth, disparity=truth.disparity * depth_factor)


def _evaluate(learner, sample, truth, count):
    records, clipping = [], []
    flags = [p.requires_grad for p in learner.actions.parameters()]
    learner.actions.requires_grad_(False)
    try:
        with torch.no_grad(), evaluation_mode(learner.model, learner.actions, learner.selector):
            for index in range(1000, 1000 + count):
                current, actual = _variant(sample, truth, index)
                state = learner.state(current)
                regions = learner.candidate_regions(state, current.sample_id)
                features = learner.selection_features(state, regions)
                gain, _, risk, events, parts = learner.selector.outputs_with_components(features)
                assert events is not None and parts is not None
                tables = [
                    learner.harness.collect(
                        state, region, learner.actions, learner.model, metric_truth=actual
                    )
                    for region in regions
                ]
                available = torch.tensor(
                    [[row.action_available and row.metric_observed for row in t] for t in tables]
                )
                available[:, 0] = True
                costs = features.new_tensor(
                    [[row.cost_ms if row.cost_ms is not None else 0.0 for row in t] for t in tables]
                )
                chosen = select_repairability(
                    gain,
                    risk,
                    costs,
                    available,
                    learner.intervention_config.total_budget_ms,
                    learner.intervention_config,
                )
                actual_gain = features.new_tensor([[row.utility for row in t] for t in tables])
                actual_harm = torch.tensor([[row.harmful for row in t] for t in tables])
                oracle = select_repairability(
                    actual_gain,
                    actual_harm.float(),
                    costs,
                    available,
                    learner.intervention_config.total_budget_ms,
                    learner.intervention_config,
                )
                records.append(
                    {
                        "synthetic_variant": index,
                        "prediction_space": "raw_uncalibrated",
                        "valid_regions": available[:, 1:].any(-1).tolist(),
                        "actual_label_available": available.tolist(),
                        "actual_gains": actual_gain.tolist(),
                        "actual_harms": actual_harm.tolist(),
                        "predicted_risk_events": events.tolist(),
                        "predicted_gain_components": parts.to_report(),
                        "policies": {
                            "selector": {
                                "action": chosen.action.name,
                                "region": chosen.region_index,
                            },
                            "oracle": {
                                "action": oracle.action.name,
                                "region": oracle.region_index,
                            },
                        },
                        "counterfactual_metric_outcomes": [
                            {
                                "region_id": ri,
                                "action": row.action.name,
                                "metric_changes": row.metric_changes,
                                "cost_ms": row.cost_ms,
                            }
                            for ri, table in enumerate(tables)
                            for row in table[1:]
                        ],
                    }
                )
                clipping.append(learner.selector.normalization.diagnostics(features))
    finally:
        for parameter, flag in zip(learner.actions.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)
    decisions = Counter(record["policies"]["selector"]["action"] for record in records)
    harmful = sum(
        bool(record["actual_harms"][choice["region"]][int(Action[choice["action"]])])
        for record in records
        if (choice := record["policies"]["selector"])["action"] != "STOP"
    )
    repairs = sum(decisions[name] for name in ("STEREO", "GAUSSIAN"))
    return {
        "risk": risk_event_metrics(records, learner.intervention_config),
        "signed_gain": gain_event_metrics(records),
        "selected_actions": dict(decisions),
        "actual_harm": {
            "selected_repairs": repairs,
            "harmful_selected_repairs": harmful,
            "rate_among_selected_repairs": harmful / repairs if repairs else None,
            "rate_among_all_decisions": harmful / len(records),
        },
        "feature_clipping": clipping,
        "records": records,
    }


def run(output: Path, snapshots: int = 48, evaluation_variants: int = 12) -> dict[str, object]:
    if snapshots <= 32 or evaluation_variants < 1:
        raise ValueError("need more than 32 snapshots and positive evaluation variants")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Synthetic diagnostic output: {output}", flush=True)
    repo = Path(__file__).resolve().parents[2]
    source_hash = _source_hash(repo)
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="reg-synthetic-risk-") as temporary:
        warm, sample, teacher, truth = learner_inputs(
            balanced_setup.__wrapped__(), Path(temporary) / "roi", False
        )
        warm.step(sample, joint=False, teacher_target=teacher, metric_truth=truth)
        checkpoint = output / "synthetic_trained_model.pt"
        save_student_checkpoint(
            checkpoint,
            warm,
            {"source_sha256": source_hash, "run_kind": "synthetic_fixture"},
            {"epoch": 0, "next_sample_index": 1, "samples_seen": 1, "optimizer_steps": 1},
        )
        results = {}
        for mode in ("joint", "frozen"):
            torch.manual_seed(29)
            config = replace(
                warm.intervention_config,
                normalization_frames=32,
                selector_gain_mode="signed_expectation",
                student_update_mode=mode,
            )
            learner = StudentLearner(
                copy.deepcopy(warm.model),
                replace(warm.config, joint_epochs=snapshots),
                warm.routing,
                intervention_config=config,
                mask_config=warm.mask_config,
                target_masks=warm.target_masks,
            )
            initialization = initialize_student_model(checkpoint, learner, require_trained=True)
            start_model, start_actions = _state_hash(learner.model), _state_hash(learner.actions)
            start_selector = _state_hash(learner.selector)
            adam = copy.deepcopy(learner.model_optimizer.state_dict())
            normalization_at_32 = None
            for position in range(snapshots):
                current, actual = _variant(sample, truth, position)
                learner.step(
                    current,
                    joint=True,
                    teacher_target=teacher,
                    metric_truth=actual,
                    position=position,
                )
                if position == 31:
                    normalization_at_32 = copy.deepcopy(learner.selector.normalization.state_dict())
                if (position + 1) % 16 == 0:
                    print(f"{mode}: {position + 1}/{snapshots} synthetic snapshots", flush=True)
            results[mode] = {
                "initialization": initialization,
                "intervention_config": config.to_dict(),
                "config_hash": config.config_hash,
                "student_config": asdict(learner.config),
                "routing_config": asdict(learner.routing),
                "construction_config": learner.model.config.to_dict(),
                "mask_config": asdict(learner.mask_config),
                "target_mask_index_sha256": learner.target_masks.identity,
                "initial_model_sha256": start_model,
                "initial_actions_sha256": start_actions,
                "initial_selector_sha256": start_selector,
                "model_unchanged": start_model == _state_hash(learner.model),
                "model_adam_unchanged": _state_equal(adam, learner.model_optimizer.state_dict()),
                "actions_updated": start_actions != _state_hash(learner.actions),
                "selector_updated": start_selector != _state_hash(learner.selector),
                "optimizer_update_counts": learner.optimizer_update_counts,
                "normalization_fixed_after_32": _state_equal(
                    normalization_at_32, learner.selector.normalization.state_dict()
                ),
                **_evaluate(learner, sample, truth, evaluation_variants),
            }
            assert results[mode]["normalization_fixed_after_32"]
            assert results[mode]["actions_updated"] and results[mode]["selector_updated"]
            assert results[mode]["model_unchanged"] is (mode == "frozen")
            assert results[mode]["model_adam_unchanged"] is (mode == "frozen")
    final_source_hash = _source_hash(repo)
    if final_source_hash != source_hash:
        raise RuntimeError("source/fixture files changed during the synthetic diagnostic; rerun")
    report = {
        "schema": "synthetic_joint_frozen_risk_diagnostic.v1",
        "limitations": "Analytic 5x5 fixture, repeated synthetic perturbations, short training; no native or medical quality claim; raw outputs only, no post-hoc calibration fitted.",
        "provenance": {
            "source_and_fixture_sha256": source_hash,
            "source_and_fixture_sha256_at_end": final_source_hash,
            "source_identity_verified_unchanged": True,
            "git_commit": get_git_commit(repo),
            "git_dirty": is_git_dirty(repo),
            "seed": 29,
            "initial_fixture_seed": 3,
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "hardware": platform.processor(),
            "device": "cpu",
            "torch_threads": 1,
            "training_snapshots_per_mode": snapshots,
            "evaluation_variants": evaluation_variants,
            "selection_cost_source": "same_snapshot_measured_cpu_action_latency; no efficiency claim",
        },
        "results": results,
    }
    for key in ("initial_model_sha256", "initial_actions_sha256", "initial_selector_sha256"):
        assert results["joint"][key] == results["frozen"][key]
    (output / "risk_diagnostic.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots", type=int, default=48)
    parser.add_argument("--evaluation-variants", type=int, default=12)
    arguments = parser.parse_args()
    run(arguments.output, arguments.snapshots, arguments.evaluation_variants)
