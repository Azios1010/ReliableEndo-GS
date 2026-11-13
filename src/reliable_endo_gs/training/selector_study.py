"""Fit many controlled selectors from one immutable frozen-actor outcome store."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.config.selector_refit import (
    load_selector_refit_config,
    partition_selector_training,
)
from reliable_endo_gs.config.selector_study import load_selector_study_config
from reliable_endo_gs.evaluation.selector_evidence import oracle_gate, outcome_distribution
from reliable_endo_gs.evaluation.selector_policy import (
    replay_policy,
    replay_risk_report,
    summarize_replay,
)
from reliable_endo_gs.reconstruction.checkpoint import _atomic_torch_save, _cpu_tree
from reliable_endo_gs.routing.calibrated_selector import CalibratedSelector
from reliable_endo_gs.routing.context import BALANCED_FEATURE_NAMES
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.runtime.identity import source_sha256
from reliable_endo_gs.training.harness import capture_rng, restore_rng
from reliable_endo_gs.training.selector_fitting import calibrate_selector, fit_selector_heads
from reliable_endo_gs.training.selector_operating_points import calibrate_operating_points
from reliable_endo_gs.training.selector_outcomes import project_selector_outcomes
from reliable_endo_gs.training.selector_shards import load_verified_selector_shards
from reliable_endo_gs.utils.io import write_json_atomic


def verify_gate_a(
    path: Path | None,
    checkpoint_sha256: str,
    actor_identity: dict[str, object],
    mask_identity: str,
    candidate_identity: dict[str, object] | None = None,
) -> dict[str, Any]:
    """A prior gate belongs to the exact frozen actors and candidate protocol."""
    if path is None:
        raise ValueError("Stage B requires --gate-a-report from native repair evaluation")
    report: dict[str, Any] = json.loads(path.read_text())
    if (
        report.get("schema") != "frozen_repair_gate_a.v1"
        or report.get("status") != "pass"
        or report.get("checkpoint_sha256") != checkpoint_sha256
        or report.get("operative_action_identity") != actor_identity
        or report.get("target_mask_index_sha256") != mask_identity
        or report.get("final_test_executed") is not False
        or report.get("bootstrap_unit") != "keyframe"
        or (
            candidate_identity is not None
            and report.get("candidate_protocol_identity") != candidate_identity
        )
    ):
        raise ValueError("Gate A failed or its frozen actor/candidate binding differs")
    evidence = report.get("evidence_samples_sha256")
    if not isinstance(evidence, str) or len(evidence) != 64:
        raise ValueError("Gate A lacks hashed native evidence")
    passing = []
    for action in ("fixed_stereo", "fixed_gaussian"):
        row = report.get("actions", {}).get(action, {})
        interval = row.get("left_psnr_gain_db", {})
        ci = interval.get("ci95")
        groups = interval.get("by_keyframe", {})
        passing.append(
            row.get("pass") is True
            and isinstance(ci, list)
            and len(ci) == 2
            and all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in ci)
            and interval.get("keyframes") == 4
            and len(groups) == 4
            and all(k.startswith("dataset_7/") for k in groups)
            and sum(v > 0 for v in groups.values()) >= 3
            and row.get("unknown_interventions") == 0
            and type(row.get("violation_fraction")) in (int, float)
            and 0 <= row["violation_fraction"] <= 0.25
        )
    if not any(passing):
        raise ValueError("Gate A status disagrees with its quality/guard evidence")
    return report


def selector_variant(
    parent: InterventionConfig,
    name: str,
    channels: int,
    utility: str,
    render_cost_weight_per_ms: float | None = None,
) -> InterventionConfig:
    """Study topology switches cannot modify repair forwards or physical guards."""
    if name not in ("C0", "C1", "C2"):
        raise ValueError("unknown study variant")
    return replace(
        parent,
        selector_feature_mode="statistics" if name == "C0" else "mask_tokens",
        selector_token_channels=0 if name == "C0" else channels,
        selector_risk_mode="union" if name == "C0" else "local_global",
        selector_gain_mode="signed_expectation" if name == "C2" else "regression",
        metric_utility_mode=utility,
        cost_weight_per_ms=(
            render_cost_weight_per_ms
            if utility == "render_psnr" and render_cost_weight_per_ms is not None
            else parent.cost_weight_per_ms
        ),
    )


def run_selector_study(
    *,
    collection_dir: Path,
    study_protocol_path: Path,
    refit_config_path: Path,
    output_dir: Path,
    gate_a_report: Path | None = None,
    audit_only: bool = False,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """No data-root/development inputs: fitting cannot open either GT or val_7.

    A new external directory is mandatory. Gate A and FIT-only Gate B are checked
    before the first head construction/optimizer step. Raw outcomes stay read-only.
    """
    repo = Path(__file__).resolve().parents[3]
    if output_dir.exists() or output_dir.resolve() == repo or repo in output_dir.resolve().parents:
        raise ValueError("study requires a new external output directory")
    study, policy = (
        load_selector_study_config(study_protocol_path),
        load_selector_refit_config(refit_config_path),
    )
    if policy.partition_seed is None:
        raise ValueError("study requires an explicit partition_seed independent of fit seeds")
    preflight = json.loads((collection_dir / "preflight.json").read_text())
    partition = json.loads((collection_dir / "partition.json").read_text())
    table = load_verified_selector_shards(collection_dir / "train_outcomes")
    ids = tuple(table.metadata["sample_ids"])
    if (
        table.metadata.get("study_config_hash") != study.config_hash
        or preflight.get("study_config_hash") != study.config_hash
        or preflight.get("refit_config_hash") != policy.config_hash
        or partition != partition_selector_training(ids, policy)
        or table.metadata.get("partition_sha256") != partition["partition_sha256"]
        or not isinstance(table.metadata.get("operative_action_identity"), dict)
    ):
        raise ValueError("study collection/partition/protocol identity mismatch")
    names = table.metadata.get("feature_bank_names", [])
    channels, remainder = divmod(len(names) - len(BALANCED_FEATURE_NAMES), 3)
    if channels < 1 or remainder or channels != table.metadata.get("student_feature_channels"):
        raise ValueError("shared store lacks the complete width-derived student token bank")
    parent = InterventionConfig(**table.metadata["intervention_config"])
    output_dir.mkdir(parents=True, exist_ok=False)
    source = source_sha256(repo)
    seal = {
        "schema": "frozen_selector_study_run.v1",
        "study_config": asdict(study),
        "study_config_hash": study.config_hash,
        "source_sha256": source,
        "git_commit": get_git_commit(repo),
        "git_dirty": is_git_dirty(repo),
        "collection_source_sha256": table.metadata["source_sha256"],
        "train_outcomes_sha256": table.metadata["samples_sha256"],
        "checkpoint_sha256": table.metadata["checkpoint_sha256"],
        "partition": partition,
        "refit_config": asdict(policy),
        "operative_action_identity": table.metadata["operative_action_identity"],
        "candidate_identity": table.metadata["target_mask_index_sha256"],
        "candidate_protocol_identity": table.metadata.get("candidate_protocol_identity"),
        "development_used_for_fitting": False,
        "final_test_executed": False,
        "hardware": {"device": device_name, "torch": str(torch.__version__)},
    }
    write_json_atomic(output_dir / "protocol.json", seal)
    audit = outcome_distribution(table, partition)
    write_json_atomic(output_dir / "distribution_audit.json", audit)
    if not audit_only:
        verify_gate_a(
            gate_a_report,
            table.metadata["checkpoint_sha256"],
            table.metadata["operative_action_identity"],
            table.metadata["target_mask_index_sha256"],
            table.metadata.get("candidate_protocol_identity"),
        )
        assert gate_a_report is not None
        seal["gate_a_report_sha256"] = hashlib.sha256(gate_a_report.read_bytes()).hexdigest()
        if seal["gate_a_report_sha256"] != table.metadata.get("gate_a_report_sha256"):
            raise ValueError("fit-only Gate A differs from the one sealed at collection")
        write_json_atomic(output_dir / "protocol.json", seal)
    gates = {}
    for utility in study.utility_modes:
        cfg = selector_variant(parent, "C0", channels, utility, study.render_cost_weight_per_ms)
        projected = project_selector_outcomes(table, cfg)
        try:
            fit = projected.view(tuple(partition["fit_sample_ids"]))
        except ValueError as error:
            if "no known metric labels" not in str(error):
                raise
            gates[utility] = {"status": "no_go", "reason": "FIT has no known metric labels"}
        else:
            gates[utility] = oracle_gate(fit, cfg, study)
        validation_ids = set(partition["validation_sample_ids"])
        validation_groups = {
            s.rsplit("/", 1)[0] for s in projected.sample_ids if s in validation_ids
        }
        if len(validation_groups) < 2:
            gates[utility].update(
                status="no_go", validation_support="fewer_than_two_observed_keyframes"
            )
    write_json_atomic(output_dir / "oracle_gates.json", gates)
    if audit_only or all(g["status"] != "pass" for g in gates.values()):
        result = {
            **seal,
            "status": "audit_only" if audit_only else "oracle_gate_no_go",
            "gates": gates,
            "selector_fitted": False,
        }
        write_json_atomic(output_dir / "result.json", result)
        return result
    device = torch.device(device_name)
    rng = capture_rng(device)
    comparisons = []
    try:
        for utility in study.utility_modes:
            if gates[utility]["status"] != "pass":
                continue
            for variant in ("C0", "C1", "C2"):
                cfg = selector_variant(
                    parent, variant, channels, utility, study.render_cost_weight_per_ms
                )
                projected = project_selector_outcomes(table, cfg)
                views = {
                    name: projected.view(
                        tuple(partition[name + "_sample_ids"]), allow_empty=name == "calibration"
                    )
                    for name in ("fit", "validation", "calibration")
                }
                for seed in study.fit_seeds:
                    fit_policy = replace(policy, fit_seed=seed)
                    head, fit_report = fit_selector_heads(
                        views["fit"],
                        views["validation"],
                        cfg,
                        fit_policy,
                        int(table.metadata["selector_width"]),
                        device,
                    )
                    model = CalibratedSelector(
                        head,
                        cfg,
                        table.metadata["operative_action_identity"],
                        table.metadata["target_mask_index_sha256"],
                    ).to(device)
                    calibration = calibrate_selector(
                        model,
                        views["calibration"],
                        fit_policy,
                        excluded_keyframes=set(partition["fit_keyframes"])
                        | set(partition["validation_keyframes"]),
                    )
                    points = calibrate_operating_points(model, views["calibration"], study)
                    for pi, point in enumerate(points):
                        cutoff = (
                            point["selected"]["cutoff"]
                            if point["selected"] is not None
                            else cfg.maximum_harm_probability
                        )
                        decision = replace(
                            cfg,
                            maximum_stereo_harm_probability=cutoff,
                            maximum_gaussian_harm_probability=cutoff,
                        )
                        model.config = decision
                        model.head.config = replace(
                            head.config,
                            maximum_stereo_harm_probability=cutoff,
                            maximum_gaussian_harm_probability=cutoff,
                        )
                        model.operating_point_available = point["status"] == "selected"
                        model.validate_state()
                        validation = summarize_replay(
                            replay_policy(
                                views["validation"],
                                decision,
                                "selector",
                                repair_budget_ms=study.repair_budget_ms,
                                model=model,
                            )
                        )
                        risk_report = replay_risk_report(
                            views["validation"], model, repair_budget_ms=study.repair_budget_ms
                        )
                        path = output_dir / f"{utility}-{variant}-seed{seed}-budget{pi}.pt"
                        payload = {
                            "schema": model.artifact_schema,
                            "study_schema": study.schema,
                            "study_config_hash": study.config_hash,
                            "source_sha256": source,
                            "checkpoint_sha256": table.metadata["checkpoint_sha256"],
                            "width": int(table.metadata["selector_width"]),
                            "head_config": model.head.config.to_dict(),
                            "decision_config": decision.to_dict(),
                            "action_identity": table.metadata["operative_action_identity"],
                            "target_mask_index_sha256": table.metadata["target_mask_index_sha256"],
                            "candidate_protocol_identity": table.metadata.get(
                                "candidate_protocol_identity"
                            ),
                            "state": model.state_dict(),
                            "partition": partition,
                            "refit_config": asdict(fit_policy),
                            "fit_report": fit_report,
                            "train_outcomes_sha256": table.metadata["samples_sha256"],
                            "calibration": {
                                **calibration,
                                "cutoffs_tuned": True,
                                "operating_point": point,
                            },
                            "operating_point_available": model.operating_point_available,
                            "gate_a_report_sha256": seal["gate_a_report_sha256"],
                            "oracle_gate": gates[utility],
                            "development_used_for_fitting": False,
                        }
                        _atomic_torch_save(path, _cpu_tree(payload))
                        comparisons.append(
                            {
                                "utility": utility,
                                "variant": variant,
                                "seed": seed,
                                "budget_index": pi,
                                "budget": point["budget"],
                                "operating_point": point,
                                "validation": validation,
                                "artifact": path.name,
                                "validation_risk": risk_report,
                                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            }
                        )
    finally:
        restore_rng(rng, device)
    # Pre-declared ranking on VALIDATION, averaged across all fixed seeds. No DEV input exists here.
    choices = []
    for utility in study.utility_modes:
        for pi, budget in enumerate(study.harm_budgets):
            ranked = []
            for variant in ("C0", "C1", "C2"):
                rows = [
                    r
                    for r in comparisons
                    if r["utility"] == utility
                    and r["budget_index"] == pi
                    and r["variant"] == variant
                ]
                if len(rows) != len(study.fit_seeds) or any(
                    r["validation"]["unknown_selected"] for r in rows
                ):
                    continue
                ranked.append(
                    (statistics.mean(r["validation"]["mean_net_gain"] for r in rows), variant)
                )
            winner = (
                max(ranked, key=lambda v: (v[0], {"C1": 2, "C0": 1, "C2": 0}[v[1]]))[1]
                if ranked
                else None
            )
            choices.append(
                {
                    "utility": utility,
                    "budget": budget,
                    "variant": winner,
                    "selection_data": "VALIDATION_only",
                    "tie_order": ["C1", "C0", "C2"],
                }
            )
    if source_sha256(repo) != source:
        raise ValueError("source changed during selector study; outputs are not certified")
    result = {
        **seal,
        "status": "controlled_selector_fits_complete",
        "gates": gates,
        "comparisons": comparisons,
        "choices_before_development": choices,
        "selector_fitted": True,
        "promotion": "pending_native_development_evaluation",
        "FPS": "deferred_not_measured",
    }
    write_json_atomic(output_dir / "result.json", result)
    return result
