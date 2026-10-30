"""Apply the predeclared development promotion rule to already chosen selectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.config.selector_study import SelectorStudyConfig
from reliable_endo_gs.evaluation.selector_evidence import best_fixed_interval, keyframe_interval
from reliable_endo_gs.evaluation.selector_policy import keyframe_mean
from reliable_endo_gs.utils.io import write_json_atomic


def evaluate_selector_promotion(
    *, study_dir: Path, evaluation_index: Path, output_dir: Path
) -> dict[str, Any]:
    """Reads evidence only. No model fitting, calibration, variant choice or FPS claim.

    Index maps chosen selector artifact filenames to native evaluation directories.
    Every predeclared seed is required; missing/unknown outcomes cannot be ignored.
    """
    repo = Path(__file__).resolve().parents[3]
    if output_dir.exists() or output_dir.resolve() == repo or repo in output_dir.resolve().parents:
        raise ValueError("promotion requires a new external output directory")
    study_result = json.loads((study_dir / "result.json").read_text())
    if study_result.get("status") != "controlled_selector_fits_complete":
        raise ValueError("promotion requires a completed, gated selector study")
    study = SelectorStudyConfig(**study_result["study_config"])
    index = json.loads(evaluation_index.read_text())
    reports = []
    cohort_ids = None
    for choice in study_result["choices_before_development"]:
        chosen = [
            r
            for r in study_result["comparisons"]
            if r["utility"] == choice["utility"]
            and r["budget"] == choice["budget"]
            and r["variant"] == choice["variant"]
        ]
        if choice["variant"] is None:
            reports.append(
                {**choice, "status": "no_go", "reason": "no variant selected on VALIDATION"}
            )
            continue
        if {r["seed"] for r in chosen} != set(study.fit_seeds):
            raise ValueError("promotion cannot discard unfavorable selector seeds")
        seeds = []
        for row in chosen:
            artifact = study_dir / row["artifact"]
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != row["artifact_sha256"]:
                raise ValueError("selected artifact changed after VALIDATION selection")
            if row["artifact"] not in index:
                raise ValueError("missing native evidence for chosen artifact: " + row["artifact"])
            directory = Path(index[row["artifact"]])
            meta = json.loads((directory / "result.json").read_text())
            content = (directory / "samples.jsonl").read_bytes()
            if (
                meta.get("weights_unchanged") is not True
                or meta.get("samples_sha256") != hashlib.sha256(content).hexdigest()
                or meta.get("selector_refit_sha256") != row["artifact_sha256"]
                or meta.get("operative_action_identity")
                != study_result["operative_action_identity"]
                or meta.get("target_mask_index_sha256") != study_result["candidate_identity"]
                or meta.get("study_config_hash") != study.config_hash
                or meta.get("source_sha256") != study_result["source_sha256"]
            ):
                raise ValueError("native selector/actor/candidate/source evidence mismatch")
            records = [json.loads(line) for line in content.splitlines()]
            ids = [r["sample_id"] for r in records]
            if (
                not ids
                or len(set(ids)) != len(ids)
                or any(s.split("/")[0] != "dataset_7" for s in ids)
            ):
                raise ValueError("promotion requires a unique val_7 development cohort")
            if cohort_ids is None:
                cohort_ids = ids
            elif cohort_ids != ids:
                raise ValueError("all seeds/arms require the identical development cohort")
            payload = torch.load(artifact, map_location="cpu", weights_only=True)
            penalty = payload["decision_config"].get("cost_weight_per_ms", 0.0)
            policies = {}
            for name in ("selector", "base", "fixed_stereo", "fixed_gaussian"):
                policies[name] = [
                    {
                        "sample_id": r["sample_id"],
                        "net_gain": None
                        if r["policies"][name]["gain"] is None
                        else r["policies"][name]["gain"]
                        - penalty * r["policies"][name]["estimated_action_ms"],
                        "harm": r["policies"][name].get("harm"),
                    }
                    for r in records
                ]
            interval = best_fixed_interval(
                policies,
                focal="selector",
                comparators=("base", "fixed_stereo", "fixed_gaussian"),
                repetitions=study.bootstrap_repetitions,
                seed=study.bootstrap_seed,
            )
            best = interval["best_fixed"]
            selector_harm = keyframe_mean(policies["selector"], "harm")
            fixed_harm = None if best is None else keyframe_mean(policies[best], "harm")
            unknown = sum(
                r["policies"]["selector"]["action"] != "STOP"
                and r["policies"]["selector"].get("metric_feasible") is None
                for r in records
            )
            unknown_fixed = sum(
                r["policies"][name]["action"] != "STOP"
                and r["policies"][name].get("metric_feasible") is None
                for r in records
                for name in ("fixed_stereo", "fixed_gaussian")
            )
            passed = bool(
                interval["ci95"] is not None
                and interval["ci95"][0] > 0
                and not unknown
                and not unknown_fixed
                and interval["keyframes"] == 4
                and selector_harm is not None
                and fixed_harm is not None
                and selector_harm <= fixed_harm
            )
            render = {}
            for metric in (
                "left_psnr",
                "right_psnr",
                "left_ssim",
                "right_ssim",
                "left_lpips_alex",
                "right_lpips_alex",
            ):
                pairs = []
                for r in records:
                    selected = r["policies"]["selector"]["metrics"].get(metric, {}).get("value")
                    base = r["policies"]["base"]["metrics"].get(metric, {}).get("value")
                    if selected is not None and base is not None:
                        pairs.append(
                            (
                                r["sample_id"],
                                (base - selected) if "lpips" in metric else (selected - base),
                            )
                        )
                render[metric + "_gain_vs_base"] = keyframe_interval(
                    pairs, repetitions=study.bootstrap_repetitions, seed=study.bootstrap_seed
                )
            seeds.append(
                {
                    "seed": row["seed"],
                    "pass": passed,
                    "selector_minus_best_fixed": interval,
                    "selector_violation_fraction": selector_harm,
                    "best_fixed_violation_fraction": fixed_harm,
                    "unknown_selected": unknown,
                    "unknown_fixed": unknown_fixed,
                    "render_quality": render,
                    "lpips_status": meta.get("perceptual_metric", {}).get("status", "unavailable"),
                    "native_samples_sha256": meta["samples_sha256"],
                }
            )
        passes = sum(r["pass"] for r in seeds)
        reports.append(
            {
                **choice,
                "status": "pass_development_quality_gate" if passes >= 2 else "no_go",
                "passing_seeds": passes,
                "seeds": seeds,
                "lpips_complete": all(r["lpips_status"] == "measured" for r in seeds),
            }
        )
    result = {
        "schema": "selector_development_promotion.v1",
        "comparisons": reports,
        "study_result_sha256": hashlib.sha256((study_dir / "result.json").read_bytes()).hexdigest(),
        "evaluation_index_sha256": hashlib.sha256(evaluation_index.read_bytes()).hexdigest(),
        "variant_chosen_before_development": True,
        "student_seeds": 1,
        "FPS": "deferred_not_measured",
        "paper_claim_ready": False,
        "final_test_executed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "promotion.json", result)
    return result
