"""Distribution evidence and gates; adjacent frames are not independent units."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.config.selector_study import SelectorStudyConfig
from reliable_endo_gs.evaluation.selector_policy import replay_policy, summarize_replay
from reliable_endo_gs.training.metric_quality import metric_risk_targets
from reliable_endo_gs.training.selector_outcomes import SelectorOutcomes


def keyframe_interval(
    pairs: list[tuple[str, float]], *, repetitions: int, seed: int
) -> dict[str, Any]:
    """Paired bootstrap of keyframe means; conditional on this single student seed."""
    groups: dict[str, list[float]] = defaultdict(list)
    for sid, value in pairs:
        if not np.isfinite(value):
            raise ValueError("bootstrap requires finite paired outcomes")
        groups[sid.rsplit("/", 1)[0]].append(value)
    means = np.array([statistics.mean(groups[k]) for k in sorted(groups)])
    if len(means) < 2 or type(repetitions) is not int or repetitions < 1:
        return {
            "mean": None if not len(means) else float(means.mean()),
            "ci95": None,
            "keyframes": len(means),
            "by_keyframe": dict(zip(sorted(groups), means.tolist(), strict=True)),
        }
    rng = np.random.default_rng(seed)
    values = means[rng.integers(0, len(means), size=(repetitions, len(means)))].mean(1)
    return {
        "mean": float(means.mean()),
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "keyframes": len(means),
        "by_keyframe": dict(zip(sorted(groups), means.tolist(), strict=True)),
        "unit": "keyframe",
        "independent_student_seeds": 1,
    }


def outcome_distribution(table: SelectorOutcomes, partition: dict[str, Any]) -> dict[str, Any]:
    """Per-role/keyframe/action/event counts, utility quantiles and unknown coverage."""
    cfg = InterventionConfig(**table.metadata["intervention_config"])
    roles = {
        s: name
        for name in ("fit", "validation", "calibration")
        for s in partition[name + "_sample_ids"]
    }
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in table.records:
        role = roles.get(record["sample_id"], "development")
        key = record["sample_id"].rsplit("/", 1)[0]
        for region in record["interventions"]:
            for ai, row in enumerate(region["actions"][1:], 1):
                event = (
                    metric_risk_targets(row["metric_changes"], cfg, ai)
                    if row["selector_label_available"]
                    else None
                )
                for group in (role, role + "/" + key):
                    pools[group + "/" + row["action"]].append({**row, "events": event})
    result = {}
    for name, rows in sorted(pools.items()):
        known = [r for r in rows if r["selector_label_available"]]
        utility = [r["global_gain"] for r in known]
        result[name] = {
            "candidates": len(rows),
            "known": len(known),
            "unknown": len(rows) - len(known),
            "unavailable": sum(r.get("action_available") is False for r in rows),
            "unknown_rate": (len(rows) - len(known)) / len(rows),
            "harm_rate": statistics.mean(r["harm"] for r in known) if known else None,
            "event_rates": {
                e: statistics.mean(r["events"][i] for r in known) if known else None
                for i, e in enumerate(("local", "global", "any"))
            },
            "utility_quantiles_0_25_50_75_100": np.quantile(
                utility, [0, 0.25, 0.5, 0.75, 1]
            ).tolist()
            if utility
            else None,
        }
    return {
        "groups": result,
        "unknown_is_safe": False,
        "units": "descriptive_candidate_counts_not_independent_frames",
    }


def best_fixed_interval(
    policies: dict[str, list[dict[str, Any]]],
    *,
    focal: str,
    comparators: tuple[str, ...],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Paired keyframe bootstrap; reselect the best fixed comparator in EACH resample."""
    names = (focal, *comparators)
    ids = [r["sample_id"] for r in policies[focal]]
    if any([r["sample_id"] for r in policies[n]] != ids for n in names):
        raise ValueError("paired policy interval requires identical ordered frames")
    groups: dict[str, list[list[float]]] = defaultdict(list)
    for i, sid in enumerate(ids):
        values = [policies[n][i]["net_gain"] for n in names]
        if all(v is not None and np.isfinite(v) for v in values):
            groups[sid.rsplit("/", 1)[0]].append(values)
    if len(groups) < 2:
        return {"mean": None, "ci95": None, "keyframes": len(groups), "best_fixed": None}
    matrix = np.array([np.mean(groups[k], axis=0) for k in sorted(groups)])
    mean = matrix.mean(0)
    rng = np.random.default_rng(seed)
    resampled = matrix[rng.integers(0, len(matrix), size=(repetitions, len(matrix)))].mean(1)
    delta = resampled[:, 0] - resampled[:, 1:].max(1)
    return {
        "mean": float(mean[0] - mean[1:].max()),
        "ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
        "best_fixed": comparators[int(mean[1:].argmax())],
        "keyframes": len(groups),
        "by_keyframe": {
            k: float(v[0] - v[1:].max()) for k, v in zip(sorted(groups), matrix, strict=True)
        },
        "unit": "keyframe",
        "comparator_reselected_per_bootstrap": True,
    }


def oracle_gate(
    table: SelectorOutcomes, config: InterventionConfig, study: SelectorStudyConfig
) -> dict[str, Any]:
    """Gate B uses FIT only and identical fully observed frame support for all policies."""
    if table.metadata.get("collection_role") != "train":
        raise ValueError("oracle gate requires FIT from TRAIN")
    complete = [
        r
        for r in table.records
        if all(
            all(a["selector_label_available"] for a in region["actions"])
            for region in r["interventions"]
        )
    ]
    complete_table = SelectorOutcomes(
        table.metadata,
        complete,
        table.features,
        table.global_gain,
        table.local_gain,
        table.harms,
        table.sample_ids,
        table.risk_components,
    )
    policies = {
        name: replay_policy(complete_table, config, name, repair_budget_ms=study.repair_budget_ms)
        for name in ("stop", "fixed_stereo", "fixed_gaussian", "oracle")
    }
    summaries = {k: summarize_replay(v) for k, v in policies.items()}
    fixed = max(
        ("stop", "fixed_stereo", "fixed_gaussian"),
        key=lambda k: summaries[k]["mean_net_gain"] or 0.0,
    )
    interval = best_fixed_interval(
        policies,
        focal="oracle",
        comparators=("stop", "fixed_stereo", "fixed_gaussian"),
        repetitions=study.bootstrap_repetitions,
        seed=study.bootstrap_seed,
    )
    unique: dict[str, set[str]] = {"STEREO": set(), "GAUSSIAN": set()}
    # Removing an action must reduce the safe oracle value, not merely break a tie.
    for record, winner in zip(complete, policies["oracle"], strict=True):
        action = winner["action"]
        if action not in unique or winner["net_gain"] <= 0:
            continue
        alternatives = [0.0]
        for region in record["interventions"]:
            for ai, row in enumerate(region["actions"][1:], 1):
                cost = table.metadata["deployment_costs_ms"][ai]
                if (
                    row["action"] != action
                    and not row["harm"]
                    and 0 < cost <= study.repair_budget_ms
                ):
                    alternatives.append(row["global_gain"] - config.cost_weight_per_ms * cost)
        if winner["net_gain"] > max(alternatives) + 1e-8:
            unique[action].add(record["sample_id"].rsplit("/", 1)[0])
    passed = bool(
        interval["ci95"] is not None
        and interval["ci95"][0] > study.minimum_oracle_headroom
        and all(len(v) >= study.minimum_unique_win_keyframes for v in unique.values())
    )
    return {
        "status": "pass" if passed else "no_go",
        "scope": "FIT_only_development_gate_not_Gate_I_certification",
        "oracle_minus_best_fixed": interval,
        "best_fixed": fixed,
        "policies": summaries,
        "unique_win_keyframes": {k: sorted(v) for k, v in unique.items()},
        "complete_frames": len(complete),
        "excluded_incomplete_frames": len(table.records) - len(complete),
        "study_config_hash": study.config_hash,
        "selector_fitted": False,
    }


def stage_a_gate(directory: Path, *, repetitions: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Derive Gate A from a verified native evaluation, without rerunning/tuning it."""
    meta = json.loads((directory / "result.json").read_text())
    content = (directory / "samples.jsonl").read_bytes()
    if (
        meta.get("weights_unchanged") is not True
        or meta.get("samples_sha256") != hashlib.sha256(content).hexdigest()
        or not isinstance(meta.get("operative_action_identity"), dict)
    ):
        raise ValueError("Gate A needs verified native evidence and the frozen actor identity")
    records = [json.loads(line) for line in content.splitlines()]
    if not records or any(r["sample_id"].split("/")[0] != "dataset_7" for r in records):
        raise ValueError("Gate A requires the declared val_7 development cohort")
    reports = {}
    for action in ("fixed_stereo", "fixed_gaussian"):
        pairs = []
        harms, unknown = [], 0
        for r in records:
            a, b = r["policies"][action], r["policies"]["base"]
            x, y = a["metrics"]["left_psnr"]["value"], b["metrics"]["left_psnr"]["value"]
            if x is not None and y is not None:
                pairs.append((r["sample_id"], x - y))
            if a["action"] != "STOP":
                if a.get("metric_feasible") is None:
                    unknown += 1
                else:
                    harms.append(not a["metric_feasible"])
        interval = keyframe_interval(pairs, repetitions=repetitions, seed=seed)
        harm = statistics.mean(harms) if harms else None
        passed = (
            interval["keyframes"] == 4
            and interval["ci95"] is not None
            and interval["ci95"][0] > 0
            and sum(v > 0 for v in interval["by_keyframe"].values()) >= 3
            and harm is not None
            and harm <= 0.25
            and not unknown
        )
        reports[action] = {
            "pass": bool(passed),
            "left_psnr_gain_db": interval,
            "violation_fraction": harm,
            "unknown_interventions": unknown,
        }
    return {
        "schema": "frozen_repair_gate_a.v1",
        "status": "pass" if any(r["pass"] for r in reports.values()) else "no_go",
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "operative_action_identity": meta["operative_action_identity"],
        "candidate_protocol_identity": meta.get("candidate_protocol_identity"),
        "target_mask_index_sha256": meta["target_mask_index_sha256"],
        "evidence_samples_sha256": meta["samples_sha256"],
        "actions": reports,
        "bootstrap_unit": "keyframe",
        "final_test_executed": False,
    }
