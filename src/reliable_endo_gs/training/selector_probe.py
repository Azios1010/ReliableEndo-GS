"""Bounded fit of the existing selector on immutable logged action outcomes.

This capacity/optimization diagnostic never constructs a student, action or
renderer. TRAIN rows are the last logged snapshot per frame, not outcomes
regenerated from the final model. Development cannot fit statistics or weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.repairability import select_repairability
from reliable_endo_gs.routing.tokens import selector_feature_names
from reliable_endo_gs.runtime.git import get_git_commit, is_git_dirty
from reliable_endo_gs.runtime.identity import source_sha256
from reliable_endo_gs.training.metric_quality import metric_risk_targets, metric_targets
from reliable_endo_gs.utils.io import write_json_atomic


@dataclass(frozen=True)
class SelectorSnapshots:
    """CPU features [R,F], global/local/harm labels [R,3], one ID per region."""

    features: torch.Tensor
    global_gain: torch.Tensor
    local_gain: torch.Tensor
    harms: torch.Tensor
    sample_ids: tuple[str, ...]
    excluded_unknown_regions: int
    risk_components: torch.Tensor | None = None


def read_selector_snapshots(
    path: Path, config: InterventionConfig, *, training: bool
) -> SelectorSnapshots:
    """Validate logged labels against their canonical owner; exclude unknowns."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if training:
        latest = {}
        for row in rows:
            if row.get("event") == "sample_progress":
                latest[row["sample_id"]] = row
        rows = list(latest.values())
    elif len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate development frame")
    features, gains, locals_, harms, ids = [], [], [], [], []
    excluded = 0
    components = []
    for row in rows:
        sid = row["sample_id"]
        if not isinstance(sid, str) or sid.split("/")[0] not in {
            f"dataset_{i}" for i in range(1, 8)
        }:
            raise ValueError("selector diagnostic only accepts original SCARED training folds")
        for ri, feature in enumerate(row.get("features_raw", [])):
            if training:
                actions = row["interventions"][ri]["actions"]
                if [action["action"] for action in actions] != ["STOP", "STEREO", "GAUSSIAN"]:
                    raise ValueError("logged action order mismatch")
                available = all(action["selector_label_available"] for action in actions)
                changes = {a: actions[a]["metric_changes"] for a in (1, 2)}
                recorded = [action["global_gain"] for action in actions]
                recorded_local = [action["local_gain"] for action in actions]
                recorded_harm = [action["harm"] for action in actions]
            else:
                if not row["valid_regions"][ri]:
                    continue
                available = all(row["actual_label_available"][ri])
                outcomes = {
                    outcome["action"]: outcome
                    for outcome in row["counterfactual_metric_outcomes"]
                    if outcome["region_id"] == ri
                }
                changes = {
                    a: outcomes[name]["metric_changes"]
                    for a, name in ((1, "STEREO"), (2, "GAUSSIAN"))
                }
                recorded = row["actual_gains"][ri]
                recorded_local = None
                recorded_harm = row["actual_harms"][ri]
            if not available:
                excluded += 1
                continue
            if (
                len(feature) != len(selector_feature_names(config))
                or any(not math.isfinite(value) for value in feature)
                or recorded[0] != 0
                or recorded_harm[0] != 0
            ):
                raise ValueError("invalid pre-action features or STOP target")
            triples = [(0.0, 0.0, False)] + [metric_targets(changes[a], config, a) for a in (1, 2)]
            if any(
                not math.isclose(triples[a][0], recorded[a], rel_tol=1e-5, abs_tol=1e-6)
                or triples[a][2] != bool(recorded_harm[a])
                or (
                    recorded_local is not None
                    and not math.isclose(
                        triples[a][1], recorded_local[a], rel_tol=1e-5, abs_tol=1e-6
                    )
                )
                for a in range(3)
            ):
                raise ValueError("logged outcome labels differ from canonical metric targets")
            features.append(feature)
            gains.append([t[0] for t in triples])
            locals_.append([t[1] for t in triples])
            harms.append([float(t[2]) for t in triples])
            ids.append(sid)
            components.append(
                [[0.0, 0.0]]
                + [list(metric_risk_targets(changes[a], config, a)[:2]) for a in (1, 2)]
            )
    if not features:
        raise ValueError("no complete observed selector snapshots")
    return SelectorSnapshots(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(gains, dtype=torch.float32),
        torch.tensor(locals_, dtype=torch.float32),
        torch.tensor(harms, dtype=torch.float32),
        tuple(ids),
        excluded,
        torch.tensor(components, dtype=torch.float32),
    )


@torch.no_grad()
def selector_fit_metrics(head: BalancedSelector, table: SelectorSnapshots) -> dict[str, object]:
    """Fit/ranking diagnostics on fixed rows; no deployment latency claim."""
    device = next(head.parameters()).device
    prediction, _, risks, events, gain_components = head.outputs_with_components(
        table.features.to(device)
    )
    events = None if events is None else events.cpu()
    prediction, risks = prediction.cpu(), risks.cpu()
    result: dict[str, object] = {}
    for a, name in ((1, "stereo"), (2, "gaussian")):
        p, y = prediction[:, a], table.global_gain[:, a]
        harm = table.harms[:, a]
        action_result: dict[str, object] = {
            "mae": float((p - y).abs().mean()),
            "prediction_mean": float(p.mean()),
            "target_mean": float(y.mean()),
            "prediction_std": float(p.std(unbiased=False)),
            "target_std": float(y.std(unbiased=False)),
            "harm_brier": float((risks[:, a] - harm).square().mean()),
            "harm_prediction_mean": float(risks[:, a].mean()),
            "harm_rate": float(harm.mean()),
        }
        result[name] = action_result
        if gain_components is not None:
            probability = gain_components.positive_probability[:, a - 1].cpu()
            positive = y > 0
            positive_gain = gain_components.positive_gain[:, a - 1].cpu()
            negative_cost = gain_components.negative_cost[:, a - 1].cpu()
            action_result["gain_components"] = {
                "positive_probability_brier": float(
                    (probability - positive.float()).square().mean()
                ),
                "positive_gain_rate": float(positive.float().mean()),
                "positive_probability_mean": float(probability.mean()),
                "positive_magnitude_mae": float(
                    (positive_gain[positive] - y[positive]).abs().mean()
                )
                if bool(positive.any())
                else None,
                "negative_magnitude_mae": float(
                    (negative_cost[~positive] + y[~positive]).abs().mean()
                )
                if bool((~positive).any())
                else None,
            }
        if events is not None:
            if table.risk_components is None:
                raise ValueError("event metrics require explicit local/global labels")
            targets = torch.cat((table.risk_components[:, a], table.harms[:, a, None]), -1)
            action_result["risk_events"] = {
                event: {
                    "brier": float((events[:, a, j] - targets[:, j]).square().mean()),
                    "mean_prediction": float(events[:, a, j].mean()),
                    "harm_rate": float(targets[:, j].mean()),
                }
                for j, event in enumerate(("local", "global", "any"))
            }
    known_safe = ~table.harms.bool()
    oracle = table.global_gain.masked_fill(~known_safe, -torch.inf).argmax(1)
    # Ranking uses true feasibility to isolate gain prediction from risk fitting.
    gain_choice = prediction.masked_fill(~known_safe, -torch.inf).argmax(1)
    stereo_wins = oracle == 1
    result.update(
        regions=len(table.sample_ids),
        frames=len(set(table.sample_ids)),
        true_safe_gain_ranking_accuracy=float((gain_choice == oracle).float().mean()),
        stereo_best_regions=int(stereo_wins.sum()),
        stereo_best_correct=int((stereo_wins & (gain_choice == 1)).sum()),
        excluded_unknown_regions=table.excluded_unknown_regions,
    )
    return result


@torch.no_grad()
def replay_logged_selection(
    head: BalancedSelector, development_log: Path, action_costs_ms: torch.Tensor
) -> dict[str, object]:
    """Replay with original budgets; GT availability never filters proposals.

    GT scores selected actions afterward. Original costs/timings are reused;
    this is not a new runtime benchmark.
    """
    device = next(head.parameters()).device
    counts: Counter[str] = Counter()
    violations = unknown = agreement = samples = oracle_samples = 0
    measured = []
    for line in development_log.read_text().splitlines():
        row = json.loads(line)
        samples += 1
        features = torch.tensor(row["features_raw"], dtype=torch.float32, device=device).reshape(
            -1, len(selector_feature_names(head.config))
        )
        gains, harms = (
            head.predict(features)
            if len(features)
            else (features.new_zeros((0, 3)), features.new_zeros((0, 3)))
        )
        decision = select_repairability(
            gains,
            harms,
            action_costs_ms.to(device).expand_as(gains),
            torch.tensor(row["valid_regions"], dtype=torch.bool, device=device),
            row["remaining_ms"],
            head.config,
        )
        counts[decision.action.name] += 1
        oracle = row.get("policies", {}).get("oracle")
        if oracle is not None:
            oracle_samples += 1
            agreement += (decision.action.name, decision.region_index) == (
                oracle["action"],
                oracle["region"],
            )
        if decision.region_index is None:
            measured.append(0.0)
            continue
        ri, ai = decision.region_index, int(decision.action)
        if not row["actual_label_available"][ri][ai]:
            unknown += 1
        else:
            measured.append(row["actual_gains"][ri][ai])
            violations += bool(row["actual_harms"][ri][ai])
    return {
        "samples": samples,
        "actions": dict(counts),
        "metric_violations": violations,
        "unknown_interventions": unknown,
        "known_mean_utility": sum(measured) / len(measured) if measured else None,
        "oracle_agreement": agreement / oracle_samples if oracle_samples else None,
        "timing_semantics": "original_logged_remaining_budget_and_action_costs_not_new_benchmark",
    }


def fit_fixed_selector(
    head: BalancedSelector,
    training: SelectorSnapshots,
    *,
    steps: int,
    learning_rate: float,
    snapshots: set[int],
) -> list[dict[str, object]]:
    """Existing objective/balancer, full TRAIN batch; no development selection."""
    if not 1 <= steps <= 2000 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("bounded selector fit requires 1..2000 steps and positive learning rate")
    device = next(head.parameters()).device
    x, global_gain, local_gain, harms = (
        value.to(device)
        for value in (training.features, training.global_gain, training.local_gain, training.harms)
    )
    head.train().requires_grad_(True)
    from reliable_endo_gs.training.selector_optimization import (
        make_selector_balancer,
        selector_objective,
    )

    balancer = make_selector_balancer(head.config).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    logs = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, pending, _ = selector_objective(
            head,
            balancer,
            x,
            global_gain,
            local_gain,
            harms,
            risk_components=None
            if training.risk_components is None
            else training.risk_components.to(device),
        )
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite fixed selector objective")
        loss.backward()
        for network in head.optimization_networks():
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        balancer.commit(pending)
        if step in snapshots or step == steps:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                **selector_fit_metrics(head, training),
            }
            logs.append(record)
            print(json.dumps({"event": "fixed_selector_progress", **record}), flush=True)
    return logs


def run_selector_probe(
    *, pilot_dir: Path, output_dir: Path, device: str = "cuda:0", steps: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """New diagnostic artifact; source checkpoint and joint runs stay immutable."""
    repo = Path(__file__).resolve().parents[3]
    if output_dir.exists() or repo == output_dir.resolve() or repo in output_dir.resolve().parents:
        raise ValueError("selector diagnostic needs a new external output directory")
    if type(seed) is not int or seed < 0 or type(steps) is not int or not 1 <= steps <= 2000:
        raise ValueError("invalid bounded selector diagnostic seed/steps")
    checkpoint = pilot_dir / "training/final.pt"
    content = checkpoint.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = InterventionConfig(**raw["intervention_config"])
    evaluation = json.loads((pilot_dir / "development/result.json").read_text())
    if (
        evaluation["checkpoint_sha256"] != digest
        or evaluation["config_hash"] != cfg.config_hash
        or evaluation["training_manifest_hash"] != raw["identity"]["manifest_hash"]
        or evaluation["target_mask_index_sha256"] != raw["target_mask_index_sha256"]
    ):
        raise ValueError("selector probe input checkpoint/manifest/config mismatch")
    train_path, dev_path = (
        pilot_dir / "training/train.jsonl",
        pilot_dir / "development/samples.jsonl",
    )
    train = read_selector_snapshots(train_path, cfg, training=True)
    dev = read_selector_snapshots(dev_path, cfg, training=False)
    if set(train.sample_ids) & set(dev.sample_ids):
        raise ValueError("selector probe TRAIN/development overlap")
    if not set(train.sample_ids) <= set(raw["identity"]["training_worklist"]):
        raise ValueError("selector probe TRAIN frame not in original worklist")
    destination = torch.device(device)
    head = BalancedSelector(raw["routing_config"]["width"], cfg).to(destination)
    head.load_state_dict(raw["selector"], strict=True)
    head.validate_state()
    before = {
        "train": selector_fit_metrics(head, train),
        "development": selector_fit_metrics(head, dev),
    }
    costs = torch.tensor(evaluation["cost_calibration"]["calibrated_estimated_ms"])
    before["policy_replay"] = replay_logged_selection(head, dev_path, costs)
    original = evaluation["summary"]["selector"]
    replay = before["policy_replay"]
    measured_utility = replay["known_mean_utility"]
    if (
        replay["actions"] != original["actions"]
        or replay["metric_violations"] != original["metric_violations"]
        or replay["unknown_interventions"] != original["unknown_metric_interventions"]
        or not isinstance(measured_utility, (int, float))
        or not math.isclose(measured_utility, original["mean_gain"], abs_tol=1e-6)
    ):
        raise ValueError("original selector policy replay differs from recorded evaluation")
    # Preserve original TRAIN-fitted normalization, reset only the two MLPs.
    torch.manual_seed(seed)
    fresh = BalancedSelector(raw["routing_config"]["width"], cfg).to(destination)
    state = fresh.state_dict()
    for key, value in raw["selector"].items():
        if key.startswith("normalization."):
            state[key] = value.to(destination)
    fresh.load_state_dict(state, strict=True)
    fresh.validate_state()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    logs = fit_fixed_selector(
        fresh,
        train,
        steps=steps,
        learning_rate=raw["routing_config"]["learning_rate"],
        snapshots={1, 25, 124, 250, 500, 1000, 2000},
    )
    report = {
        "status": "fixed_selector_fit_complete",
        "checkpoint_sha256": digest,
        "checkpoint_source_sha256": raw["identity"]["source_sha256"],
        "source_sha256": source_sha256(repo),
        "git_commit": get_git_commit(repo),
        "git_dirty": is_git_dirty(repo),
        "config_hash": cfg.config_hash,
        "role_hash": raw["identity"]["role_hash"],
        "target_mask_index_sha256": raw["target_mask_index_sha256"],
        "training_manifest_hash": raw["identity"]["manifest_hash"],
        "development_manifest_hash": evaluation["manifest_hash"],
        "seed": seed,
        "device": str(destination),
        "hardware": torch.cuda.get_device_name(destination)
        if destination.type == "cuda"
        else "cpu",
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "steps": steps,
        "batch_regions": len(train.sample_ids),
        "elapsed_seconds": time.perf_counter() - started,
        "before": before,
        "after": {
            "train": selector_fit_metrics(fresh, train),
            "development": selector_fit_metrics(fresh, dev),
            "policy_replay": replay_logged_selection(fresh, dev_path, costs),
        },
        "input_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path, dev_path)
        },
        "joint_model_actions_updated": False,
        "teacher_inference_executed": False,
        "development_used_for_fit_or_stopping": False,
        "limitations": [
            "TRAIN table uses last logged snapshots per frame, not one regenerated final-model snapshot.",
            "Full-batch 1000-step fit changes optimization exposure as well as fixing labels.",
            "Same-keyframe development and small cohort cannot establish final generalization.",
            "True-safe gain ranking isolates gain fit; it is not a deployment policy or FPS result.",
        ],
    }
    write_json_atomic(output_dir / "result.json", report)
    (output_dir / "fit.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs))
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise ValueError("source joint checkpoint changed during diagnostic")
    return report
