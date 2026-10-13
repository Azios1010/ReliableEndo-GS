"""TRAIN-only physical tradeoffs among measured repairs, independent of a router.

This quality oracle has no budget/latency or generalization claim. It cannot
select a production coefficient from historical, misregistered GT labels.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.data.scared_a00 import A00_SUPERVISION_RESAMPLING
from reliable_endo_gs.routing.tradeoff import COMPONENT_NAMES, outcome_utility_tensor
from reliable_endo_gs.training.metric_quality import metric_action_policy
from reliable_endo_gs.training.selector_outcomes import SelectorOutcomes, load_selector_outcomes
from reliable_endo_gs.utils.io import write_json_atomic


def quality_oracle_tradeoff(
    table: SelectorOutcomes, config: InterventionConfig, *, bootstrap_seed: int = 0
) -> dict[str, Any]:
    """Measured safe alternatives per image, keyframe macro means and cluster CI.

    GT availability filters only this offline, explicitly conditional analysis;
    it must never change deployment masks, features or action eligibility.
    Counts include all observed proposals; images without complete repair
    observations are reported separately rather than counted as safe STOPs.
    """
    from reliable_endo_gs.routing.tradeoff import feasible_changes

    if config.metric_scales is None or config.metric_quality is None:
        raise ValueError("coefficient audit requires declared metric scales and physical guards")

    vectors, actions, ids = [], [], []
    for record in table.records:
        for region in record["interventions"]:
            rows = region["actions"]
            if not all(r["selector_label_available"] for r in rows):
                continue
            for ai in (1, 2):
                vector = rows[ai]["metric_changes"]
                if feasible_changes(vector, metric_action_policy(config, ai)):
                    vectors.append(vector)
                    actions.append(ai)
                    ids.append(record["sample_id"])
    changes = torch.tensor(vectors, dtype=torch.float64).reshape(-1, len(COMPONENT_NAMES))
    utilities = outcome_utility_tensor(
        changes, config.metric_scales, metric_action_policy(config, 1)
    )
    utilities *= torch.tensor(
        [config.stereo_utility_multiplier if a == 1 else 1.0 for a in actions],
        dtype=utilities.dtype,
    )
    # STOP is the initial zero alternative. Stable strict comparison preserves
    # STOP on an exact tie and the original region order on intervention ties.
    known_ids = set(table.sample_ids)
    choices = {sid: (0.0, 0, [0.0] * len(COMPONENT_NAMES)) for sid in known_ids}
    for sid, ai, score, vector in zip(ids, actions, utilities.tolist(), vectors, strict=True):
        if score > choices[sid][0]:
            choices[sid] = (score, ai, vector)
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for sid in sorted(choices):
        _, ai, vector = choices[sid]
        counts[("STOP", "STEREO", "GAUSSIAN")[ai]] += 1
        grouped[sid.rsplit("/", 1)[0]].append(vector)
    if not grouped:
        raise ValueError("no observed TRAIN images for coefficient sensitivity")
    # Equal keyframe mass, not equal region mass or artificially balanced actions.
    keyframe_means = np.asarray(
        [np.mean(grouped[k], axis=0) for k in sorted(grouped)], dtype=np.float64
    )
    means = keyframe_means.mean(axis=0)
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, len(keyframe_means), size=(500, len(keyframe_means)))
    bootstrap = keyframe_means[draws].mean(axis=1)
    lo, hi = np.quantile(bootstrap, (0.025, 0.975), axis=0)
    return {
        "actions": dict(counts),
        "observed_images": len(choices),
        "excluded_images_without_complete_repair_labels": len(table.records) - len(choices),
        "keyframes": len(grouped),
        "mean_changes_keyframe_macro": dict(zip(COMPONENT_NAMES, means.tolist(), strict=True)),
        "bootstrap_keyframe_ci95": {
            key: [float(a), float(b)] for key, a, b in zip(COMPONENT_NAMES, lo, hi, strict=True)
        },
        "semantics": "conditional_observed_quality_oracle_without_cost_or_learned_risk",
    }


def audit_utility_coefficients(
    *, outcomes: Path, partition_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Immutable audit of 24 common metric policies plus historical comparator."""
    repo = Path(__file__).resolve().parents[3]
    if output_dir.resolve() == repo or repo in output_dir.resolve().parents:
        raise ValueError("utility audit outputs must stay outside the repository")
    table = load_selector_outcomes(outcomes, "train")
    partition = json.loads(partition_path.read_text())
    seal = partition.pop("partition_sha256")
    if (
        hashlib.sha256(json.dumps(partition, sort_keys=True).encode()).hexdigest() != seal
        or seal != table.metadata["partition_sha256"]
        or partition["parent_roles_changed"] is not False
    ):
        raise ValueError("outcome/partition identity mismatch")
    groups = [set(partition[name + "_sample_ids"]) for name in ("fit", "validation", "calibration")]
    if (
        set.union(*groups) != {r["sample_id"] for r in table.records}
        or any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2)))
        or any(
            {s.rsplit("/", 1)[0] for s in groups[a]} & {s.rsplit("/", 1)[0] for s in groups[b]}
            for a, b in ((0, 1), (0, 2), (1, 2))
        )
    ):
        raise ValueError("overlapping or incomplete TRAIN roles")
    config = InterventionConfig(**table.metadata["intervention_config"])
    assert config.metric_quality is not None
    candidates = [("historical_action_multiplier", config)]
    for priority in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        for local in (0.25, 0.5, 1.0, 2.0):
            physical = dict(
                config.metric_quality,
                geometry_priority=priority,
                geometry_weight=1.0,
                local_geometry_weight=local,
            )
            candidates.append(
                (
                    f"geometry_{priority}_local_{local}",
                    replace(config, stereo_utility_multiplier=1.0, metric_quality=physical),
                )
            )
    views = {
        name: table.view(tuple(partition[name + "_sample_ids"]))
        for name in ("fit", "validation", "calibration")
    }
    records = []
    for name, cfg in candidates:
        assert cfg.metric_quality is not None
        records.append(
            {
                "name": name,
                "config_hash": cfg.config_hash,
                "geometry_priority": cfg.metric_quality["geometry_priority"],
                "local_geometry_weight": cfg.metric_quality["local_geometry_weight"],
                "stereo_utility_multiplier": cfg.stereo_utility_multiplier,
                "roles": {role: quality_oracle_tradeoff(view, cfg) for role, view in views.items()},
            }
        )
    # Describe scales on FIT alone, including harmful observed interventions.
    # These quantiles diagnose numeric imbalance; they are not a fitted policy
    # and must not silently replace the declared physical safety tolerances.
    scale_diagnostics = {}
    assert config.metric_scales is not None
    for action in (1, 2):
        measured = np.asarray(
            [
                region["actions"][action]["metric_changes"]
                for record in views["fit"].records
                for region in record["interventions"]
                if region["actions"][action]["selector_label_available"]
            ],
            dtype=np.float64,
        ).reshape(-1, len(COMPONENT_NAMES))
        scale_diagnostics[("STEREO", "GAUSSIAN")[action - 1]] = {
            "observations": len(measured),
            "components": {
                name: {
                    "declared_scale": float(scale),
                    "absolute_median": float(np.median(np.abs(measured[:, i])))
                    if len(measured)
                    else None,
                    "absolute_p90": float(np.quantile(np.abs(measured[:, i]), 0.9))
                    if len(measured)
                    else None,
                    "fraction_beyond_unit_normalization": float(
                        np.mean(np.abs(measured[:, i]) > scale)
                    )
                    if len(measured)
                    else None,
                }
                for i, (name, scale) in enumerate(
                    zip(COMPONENT_NAMES, config.metric_scales, strict=True)
                )
            },
        }
    identities = {r.get("gt_resampling_identity") for r in table.records}
    result = {
        "schema": "utility_coefficient_sensitivity.v1",
        "candidates": records,
        "fit_scale_diagnostics": scale_diagnostics,
        "outcomes_sha256": table.metadata["samples_sha256"],
        "source_sha256": table.metadata["source_sha256"],
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "partition_sha256": seal,
        "gt_resampling_identities": sorted(str(v) for v in identities),
        "corrected_grid_declared": identities == {A00_SUPERVISION_RESAMPLING},
        "coefficient_selected": False,
        "selection_limitation": "requires_corrected_GT_and_new_learned_policy_validation",
        "risk_guards_changed": False,
        "data_roles_changed": False,
        "development_or_final_test_used": False,
        "model_or_training_executed": False,
        "timing_measured": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "result.json", result)
    return result
