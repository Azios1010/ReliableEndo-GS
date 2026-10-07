"""CPU counterfactual relabeling of immutable, already executed interventions."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.runtime.identity import source_sha256
from reliable_endo_gs.training.quality_components import component_targets
from reliable_endo_gs.utils.io import write_json_atomic


def _choice(candidates: list[dict[str, Any]], *, safe: bool) -> dict[str, Any]:
    """STOP always exists; strict gain ordering retains STOP on exact ties."""
    allowed = [row for row in candidates if not safe or not row["harm"]]
    return max(allowed, key=lambda row: row["gain"])


def _gain(row: dict[str, Any], cfg: InterventionConfig) -> dict[str, Any]:
    gain, local, harm = component_targets(row["before"], row["after"], cfg)
    return {**row, "gain": gain, "local_gain": local, "harm": harm}


def sweep_stereo_weights(
    *, pilot_dir: Path, output_dir: Path, weights: tuple[float, ...] = (1.0, 3.0, 5.0)
) -> dict[str, object]:
    """Re-score fixed actions; never rerender, train, fit, or rewrite source logs.

    Training logs contain all eligible region alternatives at varying model
    snapshots. Evaluation logs contain ONLY alternatives emitted as policies;
    their counterfactual is explicitly restricted, never called a full oracle.
    """
    if (
        not weights
        or len(set(weights)) != len(weights)
        or any(not math.isfinite(w) or w <= 0 for w in weights)
    ):
        raise ValueError("sweep weights must be unique finite positive values")
    repo = Path(__file__).resolve().parents[3]
    output = output_dir.resolve()
    pilot = pilot_dir.resolve()
    if output == repo or repo in output.parents or output == pilot or pilot in output.parents:
        raise ValueError("sweep artifacts must be external and separate from source pilot")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sweep: {output}")
    paths = {
        "config": pilot / "quality/result.json",
        "run": pilot / "train/run.json",
        "train": pilot / "train/train.jsonl",
        "evaluation": pilot / "evaluation/result.json",
        "samples": pilot / "evaluation/samples.jsonl",
    }
    bodies = {key: path.read_bytes() for key, path in paths.items()}
    quality = json.loads(bodies["config"])
    cfg = InterventionConfig(**quality["config"])
    identity = json.loads(bodies["run"])["identity"]
    evaluation = json.loads(bodies["evaluation"])
    if (
        cfg.schema != "intervention_selection.v2"
        or cfg.config_hash != identity["intervention_config_hash"]
    ):
        raise ValueError("recorded config identity mismatch")
    if (
        evaluation["source_sha256"] != identity["source_sha256"]
        or evaluation["config_hash"] != cfg.config_hash
        or quality["role_hash"] != identity["role_hash"]
        or evaluation["training_manifest_hash"] != identity["manifest_hash"]
    ):
        raise ValueError("recorded training/evaluation identity mismatch")
    train = [json.loads(line) for line in bodies["train"].splitlines() if line]
    train = [row for row in train if row.get("event") == "sample_progress"]
    samples = [json.loads(line) for line in bodies["samples"].splitlines() if line]
    if not train or not samples or len(samples) != evaluation["samples"]:
        raise ValueError("missing or incomplete recorded interventions")
    summaries = []
    selections = []
    for weight in weights:
        candidate_cfg = replace(cfg, stereo_weight=weight)
        raw_regions: Counter[str] = Counter()
        safe_regions: Counter[str] = Counter()
        safe_snapshots: Counter[str] = Counter()
        for step in train:
            candidates = [{"action": "STOP", "gain": 0.0, "harm": False}]
            for region in step["interventions"]:
                rows = [_gain(row, candidate_cfg) for row in region["actions"]]
                if {r["action"] for r in rows} != {"STOP", "STEREO", "GAUSSIAN"}:
                    raise ValueError("incomplete training intervention table")
                raw_regions[_choice(rows, safe=False)["action"]] += 1
                safe_regions[_choice(rows, safe=True)["action"]] += 1
                candidates.extend(rows)
            safe_snapshots[_choice(candidates, safe=True)["action"]] += 1
        chosen_actions: Counter[str] = Counter()
        metric_values: dict[str, list[float]] = {
            key: [] for key in ("stereo_depth_mae_mm", "left_psnr", "right_psnr")
        }
        gains = []
        for sample in samples:
            choices = [
                {
                    "action": "STOP",
                    "gain": 0.0,
                    "harm": False,
                    "region": None,
                    "metrics": sample["policies"]["base"]["metrics"],
                }
            ]
            seen = set()
            for row in sample["policies"].values():
                policy_key = (row["region"], row["action"])
                if row["action"] == "STOP" or policy_key in seen:
                    continue
                seen.add(policy_key)
                if not 0 < row["estimated_action_ms"] <= sample["remaining_ms"]:
                    continue
                choices.append(
                    _gain(
                        {
                            **row,
                            "before": row["components_before"],
                            "after": row["components_after"],
                        },
                        candidate_cfg,
                    )
                )
            best = _choice(choices, safe=True)
            chosen_actions[best["action"]] += 1
            gains.append(best["gain"])
            for key in metric_values:
                value = best["metrics"][key]["value"]
                if value is not None:
                    metric_values[key].append(float(value))
            selections.append(
                {
                    "weight": weight,
                    "sample_id": sample["sample_id"],
                    "action": best["action"],
                    "region": best["region"],
                    "gain": best["gain"],
                    "known_candidates": len(choices),
                }
            )
        summaries.append(
            {
                "stereo_weight": weight,
                "training_snapshots": len(train),
                "training_regions": sum(raw_regions.values()),
                "training_region_raw_winners": dict(raw_regions),
                "training_region_safe_winners": dict(safe_regions),
                "training_snapshot_safe_best_unconstrained": dict(safe_snapshots),
                "development_restricted_safe_actions": dict(chosen_actions),
                "development_restricted_mean_gain": statistics.mean(gains),
                "development_restricted_metrics": {
                    key: {
                        "macro_mean": statistics.mean(v) if v else None,
                        "samples_available": len(v),
                    }
                    for key, v in metric_values.items()
                },
            }
        )
    result: dict[str, object] = {
        "status": "recorded_stereo_weight_sweep_complete",
        "recorded_identity": identity,
        "analysis_source_sha256": source_sha256(repo),
        "source_logs": {
            key: {"path": str(paths[key]), "sha256": hashlib.sha256(body).hexdigest()}
            for key, body in bodies.items()
        },
        "weights": summaries,
        "training_executed": False,
        "teacher_inference_executed": False,
        "renderer_executed": False,
        "GT_reloaded": False,
        "limitations": [
            "Fixed executed outputs only; retraining with new priorities can change repairs.",
            "Development candidates cover logged policies, not every region/action alternative.",
            "Training snapshot choices are unconstrained by a deployment latency budget.",
            "Harm remains component degradation, independent of utility stereo weight.",
            "The old selector predictions are not reinterpreted as a newly trained selector.",
        ],
    }
    if any(paths[key].read_bytes() != body for key, body in bodies.items()):
        raise ValueError("source logs changed during analysis")
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "result.json", result)
    write_json_atomic(output / "selections.json", selections)
    return result
