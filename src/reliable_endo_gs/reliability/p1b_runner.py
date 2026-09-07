"""Development-only execution and provenance for P1b reliability rescue."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from reliable_endo_gs.evaluation.p1 import oracle_disparity_error
from reliable_endo_gs.reliability.p1b import (
    P1B_COVERAGE_LEVELS,
    P1B_DEVELOPMENT_SEQUENCES,
    P1B_FINAL_SEQUENCES,
    P1B_FIT_SEQUENCES,
    P1B_HOLDOUT_SEQUENCES,
    CandidateSpec,
    bootstrap_mean,
    candidate_specs,
    classify_candidate,
    compute_geometry_features,
    compute_photometric_features,
    compute_trajectory_features,
    evaluate_frame_signal,
    shadow_diagnostics,
    summarize_candidate_rows,
    validate_p1b_sequences,
)
from reliable_endo_gs.uncertainty.extraction import extract_model_iterations

PINNED_UPSTREAM = "186fa2b4a2159b28393492f6df1aa444b54391a8"
BASELINE_SHA = "32D42DE10A4C683C31FB4B3E0DE831F900E11197CDF1769B18CF3BBAFA5EE16F"


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA-256 digest without changing the file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git_output(repo: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_state(repo: Path) -> dict[str, object]:
    source_sha = _git_output(repo, "rev-parse", "HEAD")
    dirty = _git_output(repo, "status", "--short", "--untracked-files=all") or ""
    remote_ref = _git_output(repo, "symbolic-ref", "refs/remotes/origin/HEAD")
    remote_sha = _git_output(repo, "rev-parse", remote_ref) if remote_ref is not None else None
    return {
        "source_sha": source_sha,
        "source_dirty": bool(dirty),
        "source_status": dirty.splitlines(),
        "remote_ref": remote_ref,
        "remote_sha": remote_sha,
    }


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return value


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for view in ("lmain", "rmain"):
        for key, value in batch[view].items():
            if isinstance(value, torch.Tensor):
                batch[view][key] = value.to(device, non_blocking=True)
    return batch


def _sequence_from_sample(value: object) -> str:
    parts = str(value).split("/")
    if len(parts) < 2:
        raise RuntimeError(f"cannot recover sequence identity from {value!r}")
    return "/".join(parts[:2])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _vram_start(device: torch.device) -> tuple[int, int] | None:
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(device)
    return int(torch.cuda.memory_allocated(device)), int(torch.cuda.memory_reserved(device))


def _vram_finish(device: torch.device, baseline: tuple[int, int] | None) -> dict[str, int | None]:
    if device.type != "cuda" or baseline is None:
        return {"peak_allocated_delta_bytes": None, "peak_reserved_delta_bytes": None}
    allocated, reserved = baseline
    return {
        "peak_allocated_delta_bytes": max(
            0, int(torch.cuda.max_memory_allocated(device)) - allocated
        ),
        "peak_reserved_delta_bytes": max(0, int(torch.cuda.max_memory_reserved(device)) - reserved),
    }


def _native_snapshot(model: torch.nn.Module, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Run the original Stage-2 prediction path on a private input copy."""

    private_batch = _clone_tree(batch)
    output, _, _ = model(private_batch, is_train=False)
    left = output["lmain"]
    snapshot: dict[str, torch.Tensor] = {}
    for key in ("flow_pred", "rot_maps", "scale_maps", "opacity_maps"):
        value = left.get(key)
        if isinstance(value, torch.Tensor):
            snapshot[key] = value.detach().clone()
    if "flow_pred" not in snapshot:
        raise RuntimeError("original Stage-2 path did not expose lmain.flow_pred")
    return snapshot


def _snapshot_differences(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in sorted(set(before) | set(after)):
        first, second = before.get(key), after.get(key)
        if first is None or second is None or first.shape != second.shape:
            result[key] = {
                "torch_equal": False,
                "max_abs_diff": None,
                "missing_or_shape_mismatch": True,
            }
            continue
        difference = (first - second).abs()
        result[key] = {
            "torch_equal": bool(torch.equal(first, second)),
            "max_abs_diff": float(difference.max().item()) if difference.numel() else 0.0,
            "missing_or_shape_mismatch": False,
        }
    return result


def _load_model(repo: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    # The historical loader is reused as an identity-preserving baseline
    # integration boundary.  P1b does not alter its implementation.
    from reliable_endo_gs.evaluation.p1_runner import _load_upstream_model

    return _load_upstream_model(repo, checkpoint_path, device)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_config(raw: Mapping[str, object]) -> None:
    if raw.get("protocol_status") != "development_only_measurement":
        raise RuntimeError("P1b requires development_only_measurement status")
    forbidden_flags = (
        "training_authorized",
        "calibration_authorized",
        "final_evaluation_authorized",
        "signal_fusion_authorized",
        "gaussian_mutation_authorized",
        "renderer_mutation_authorized",
    )
    if any(bool(raw.get(flag)) for flag in forbidden_flags):
        raise RuntimeError("P1b config authorizes an operation outside the rescue study")
    boundaries = raw.get("study_boundaries")
    if not isinstance(boundaries, Mapping):
        raise RuntimeError("P1b config lacks study_boundaries")
    forbidden_boundary_values = {
        "fit_sigma_d": True,
        "compute_sigma_D": True,
        "compute_center_covariance": True,
        "modify_gaussian_covariance": True,
        "modify_opacity": True,
        "modify_renderer": True,
        "train_fusion_model": True,
        "weighted_signal_sum": True,
    }
    for key, forbidden in forbidden_boundary_values.items():
        if boundaries.get(key) is forbidden:
            raise RuntimeError(f"P1b boundary {key} must remain false")


def _resolve_required_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required P1b environment variable is unset: {name}")
    return Path(value).expanduser().resolve()


def _verify_data_manifest(manifest_path: Path) -> dict[str, object]:
    manifest_label = str(manifest_path).lower()
    if "final" in manifest_label or "dataset_5" in manifest_label or "dataset_8" in manifest_label:
        raise RuntimeError("P1b refuses a manifest path associated with final evaluation")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    active_sequences: set[str] = set()
    for split_name in ("train_keyframes", "validation_keyframes", "test_keyframes"):
        entries = payload.get(split_name, [])
        if isinstance(entries, list):
            active_sequences.update(str(entry) for entry in entries)
    sequence_entries = payload.get("sequences", [])
    if isinstance(sequence_entries, list):
        for entry in sequence_entries:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("split", "")) in {"train", "validation", "test"}:
                keyframe = entry.get("keyframe")
                if keyframe is not None:
                    active_sequences.add(str(keyframe))
    if P1B_FINAL_SEQUENCES.intersection(active_sequences):
        raise RuntimeError("P1b development manifest active split contains a forbidden final sequence")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
    }


def _verify_identities(
    repo: Path,
    raw: Mapping[str, object],
    checkpoint_path: Path,
    data_root: Path,
    manifest_path: Path,
    split_path: Path,
    data_config_path: Path,
) -> dict[str, object]:
    identity = raw.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("P1b config lacks identity")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if sha256_file(checkpoint_path) != BASELINE_SHA:
        raise RuntimeError("frozen Stage-2 checkpoint SHA-256 mismatch")
    for path in (split_path, data_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    upstream = repo / "third_party" / "endo_e2e_gs"
    upstream_sha = _git_output(upstream, "rev-parse", "HEAD")
    upstream_status = _git_output(upstream, "status", "--short") or ""
    if upstream_sha != PINNED_UPSTREAM or upstream_status:
        raise RuntimeError("pinned upstream identity/status check failed")
    for sequence in P1B_DEVELOPMENT_SEQUENCES:
        if not (data_root / Path(sequence)).is_dir():
            raise FileNotFoundError(data_root / Path(sequence))
    source_sha_env = str(identity.get("source_sha_env", ""))
    expected_source = os.environ.get(source_sha_env) if source_sha_env else None
    actual_source = _git_output(repo, "rev-parse", "HEAD")
    if expected_source and expected_source != actual_source:
        raise RuntimeError(
            f"source SHA mismatch: actual={actual_source}, expected={expected_source}"
        )
    manifest_identity = _verify_data_manifest(manifest_path)
    return {
        **_source_state(repo),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "upstream_sha": upstream_sha,
        "upstream_status": upstream_status.splitlines(),
        "development_manifest": manifest_identity,
        "development_split_sha256": sha256_file(split_path),
        "development_config_sha256": sha256_file(data_config_path),
        "data_root": str(data_root),
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _config_mapping(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"P1b config section {key!r} is missing")
    return value


def _max_abs_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape:
        return float("inf")
    if not first.numel():
        return 0.0
    return float((first - second).abs().max().item())


def _rank(values: torch.Tensor) -> torch.Tensor:
    sorted_values, order = torch.sort(values, stable=True)
    ranks_sorted = torch.empty_like(sorted_values, dtype=torch.float64)
    start = 0
    for index in range(1, sorted_values.numel() + 1):
        if index == sorted_values.numel() or sorted_values[index] != sorted_values[start]:
            ranks_sorted[start:index] = (start + index - 1) / 2.0
            start = index
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def _spearman(values: Sequence[float], other: Sequence[float]) -> float | None:
    if len(values) != len(other) or len(values) < 2:
        return None
    first = torch.tensor(values, dtype=torch.float64)
    second = torch.tensor(other, dtype=torch.float64)
    first = _rank(first)
    second = _rank(second)
    first -= first.mean()
    second -= second.mean()
    denominator = torch.sqrt(first.square().sum() * second.square().sum())
    if denominator == 0:
        return None
    return float((first * second).sum().div(denominator).item())


def _pairwise_region_correlations(
    candidate_rows: Mapping[str, Sequence[Mapping[str, object]]],
    names: Sequence[str],
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1 :]:
            first_values: list[float] = []
            second_values: list[float] = []
            for first_row, second_row in zip(
                candidate_rows.get(first_name, ()), candidate_rows.get(second_name, ()), strict=True
            ):
                first_regions = first_row["region_values"]["score"]
                second_regions = second_row["region_values"]["score"]
                for first_value, second_value in zip(first_regions, second_regions, strict=True):
                    if first_value is None or second_value is None:
                        continue
                    if not all(
                        torch.isfinite(torch.tensor([float(first_value), float(second_value)]))
                    ):
                        continue
                    first_values.append(float(first_value))
                    second_values.append(float(second_value))
            output[f"{first_name}__vs__{second_name}"] = _spearman(first_values, second_values)
    return output


def _best_by_family(
    summaries: Mapping[str, Mapping[str, Mapping[str, object]]],
    specs: Mapping[str, CandidateSpec],
) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for family in ("dynamics", "geometry", "observation"):
        candidates = [
            name
            for name, spec in specs.items()
            if spec.family == family
            and name not in {"u1_historical", "u2_historical", "u3_corrected_raw", "u4_historical"}
        ]
        scored: list[tuple[float, str]] = []
        for name in candidates:
            values: list[float] = []
            for sequence in P1B_HOLDOUT_SEQUENCES:
                region = summaries.get(sequence, {}).get(name, {}).get("region", {})
                value = region.get("spearman")
                if value is not None:
                    values.append(float(value))
            if values:
                scored.append((sum(values) / len(values), name))
        output[family] = max(scored, key=lambda item: (item[0], item[1]))[1] if scored else None
    return output


def _family_decisions(
    decisions: Mapping[str, str],
    *,
    best: Mapping[str, str | None],
) -> dict[str, str]:
    def family_decision(family: str, prefix: str) -> str:
        values = [
            result
            for name, result in decisions.items()
            if name.startswith(prefix) and name != f"{prefix}_historical"
        ]
        if "PASS" in values:
            return "A"
        if "MARGINAL" in values:
            return "B"
        return "C"

    dynamics = family_decision("dynamics", "u1b")
    if dynamics == "C" and any(
        decisions.get(name) == "PASS" for name in ("u1_historical", "u2_historical")
    ):
        dynamics = "B"
    geometry = family_decision("geometry", "u3b")
    observation = family_decision("observation", "u4b")
    if observation == "C" and decisions.get("u4_historical") == "PASS":
        observation = "B"
    return {
        "dynamics": f"DYN-{dynamics}",
        "geometry": f"GEO-{geometry}",
        "observation": f"OBS-{observation}",
    }


def _overall_decision(family: Mapping[str, str], decisions: Mapping[str, str]) -> str:
    passing_families = sum(value.endswith("-A") for value in family.values())
    if passing_families >= 2:
        return "P1B-B"
    if passing_families == 1:
        return "P1B-A"
    if family.get("observation") == "OBS-B":
        return "P1B-C"
    if any(value == "MARGINAL" for value in decisions.values()):
        return "P1B-C"
    return "P1B-D"


def _state_analysis(
    candidate_rows_by_sequence: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    best: Mapping[str, str | None],
    *,
    high_error_threshold: float,
) -> dict[str, object]:
    dynamics_name = best.get("dynamics")
    geometry_name = best.get("geometry")
    observation_name = best.get("observation")
    if not dynamics_name or not geometry_name or not observation_name:
        return {
            "available": False,
            "confidently_wrong_evidence": "INCONCLUSIVE",
            "states": {},
        }

    thresholds: dict[str, float] = {}
    for name in (dynamics_name, geometry_name, observation_name):
        values: list[float] = []
        for sequence in P1B_FIT_SEQUENCES:
            for row in candidate_rows_by_sequence.get(sequence, {}).get(name, ()):
                values.extend(
                    float(value)
                    for value in row["region_values"]["score"]
                    if value is not None and torch.isfinite(torch.tensor(float(value)))
                )
        if not values:
            return {
                "available": False,
                "confidently_wrong_evidence": "INCONCLUSIVE",
                "states": {},
            }
        thresholds[name] = float(torch.tensor(values).median().item())

    states: dict[str, dict[str, float]] = defaultdict(
        lambda: {"regions": 0.0, "high_error_regions": 0.0, "high_error_fraction": 0.0}
    )
    holdout_bad_state = {sequence: False for sequence in P1B_HOLDOUT_SEQUENCES}
    for sequence, rows_by_name in candidate_rows_by_sequence.items():
        dyn_rows = rows_by_name.get(dynamics_name, ())
        geo_rows = rows_by_name.get(geometry_name, ())
        obs_rows = rows_by_name.get(observation_name, ())
        for dyn_row, geo_row, obs_row in zip(dyn_rows, geo_rows, obs_rows, strict=True):
            dyn_values = dyn_row["region_values"]["score"]
            geo_values = geo_row["region_values"]["score"]
            obs_values = obs_row["region_values"]["score"]
            errors = dyn_row["region_values"]["error"]
            for dyn, geo, obs, error in zip(
                dyn_values, geo_values, obs_values, errors, strict=True
            ):
                if any(value is None for value in (dyn, geo, obs, error)):
                    continue
                dyn_float, geo_float, obs_float, error_float = map(float, (dyn, geo, obs, error))
                if not all(
                    torch.isfinite(torch.tensor([dyn_float, geo_float, obs_float, error_float]))
                ):
                    continue
                stable = dyn_float <= thresholds[dynamics_name]
                geometry_consistent = geo_float <= thresholds[geometry_name]
                observation_consistent = obs_float <= thresholds[observation_name]
                if stable and geometry_consistent and observation_consistent:
                    state = "A"
                elif stable and not geometry_consistent and not observation_consistent:
                    state = "B"
                elif stable and geometry_consistent and not observation_consistent:
                    state = "C"
                elif not stable and not geometry_consistent and not observation_consistent:
                    state = "D"
                else:
                    state = "OTHER"
                states[state]["regions"] += 1.0
                if error_float > high_error_threshold:
                    states[state]["high_error_regions"] += 1.0
                if (
                    sequence in holdout_bad_state
                    and state in {"B", "C"}
                    and error_float > high_error_threshold
                ):
                    holdout_bad_state[sequence] = True
    for summary in states.values():
        if summary["regions"]:
            summary["high_error_fraction"] = summary["high_error_regions"] / summary["regions"]
    confidently_wrong = (
        "YES"
        if all(holdout_bad_state.values())
        and any(states.get(key, {}).get("regions", 0) for key in ("B", "C"))
        else "NO"
    )
    return {
        "available": True,
        "thresholds_fit_medians": thresholds,
        "states": dict(states),
        "holdout_bad_state": holdout_bad_state,
        "confidently_wrong_evidence": confidently_wrong,
    }


def _render_report(result: Mapping[str, object]) -> str:
    """Render a compact machine-generated P1b report from frozen JSON facts."""

    source = result.get("source", {})
    protocol = result.get("protocol", {})
    decisions = result.get("decisions", {})
    family = result.get("family_decisions", {})
    best = result.get("best_candidates", {})
    invariance = result.get("baseline_invariance", {})
    shadow = result.get("shadow_iteration_study", {})
    consensus = result.get("stereo_consensus_feasibility", {})
    per_sequence = result.get("per_sequence", {})
    macro = result.get("macro", {})
    lines = [
        "# RELIABLEENDO-GS P1b RELIABILITY RESCUE REPORT",
        "",
        "## SOURCE",
        f"source before: {source.get('before', {}).get('source_sha')}",
        f"source after: {source.get('after', {}).get('source_sha')}",
        f"remote SHA: {source.get('before', {}).get('remote_sha')}",
        f"baseline SHA: {result.get('identity', {}).get('checkpoint_sha256')}",
        f"third_party SHA: {result.get('identity', {}).get('upstream_sha')}",
        "",
        "## PROTOCOL",
        f"config: {protocol.get('config_path')}",
        f"SHA: {protocol.get('config_sha256')}",
        f"development samples: {result.get('samples')}",
        "final samples accessed: NO",
        "",
        "## HISTORICAL CONTROLS",
        "U1: u1_historical",
        "U2: u2_historical",
        "corrected U3: u3_corrected_raw",
        "U4: u4_historical",
        "",
    ]
    family_names = {
        "U1b — TRAJECTORY": [
            "u1_historical",
            "u1b_path",
            "u1b_decay",
            "u1b_decay_deviation",
            "u1b_acceleration",
            "u1b_directional_disagreement",
            "u1b_directional_agreement",
        ],
        "U2b — CONVERGENCE STRUCTURE": [
            "u2_historical",
            "u2b_normalized_dispersion",
            "u2b_path_efficiency",
            "u2b_oscillation",
        ],
        "U3b — GEOMETRY": [
            "u3_corrected_raw",
            "u3b_valid_absolute",
            "u3b_relative",
            "u3b_visibility_aware",
        ],
        "U4b — OBSERVATION CONSISTENCY": [
            "u4_historical",
            "u4b_bidirectional_mean",
            "u4b_bidirectional_max",
            "u4b_bidirectional_disagreement",
            "u4b_left_right_asymmetry",
            "u4b_normalized_photo",
            "u4b_gradient",
            "u4b_ssim_like",
            "u4b_visibility_aware",
            "u4b_specularity_aware",
        ],
    }
    sequence_labels = (
        "dataset_1/keyframe_1",
        "dataset_2/keyframe_1",
        "dataset_3/keyframe_1",
        "dataset_7/keyframe_2",
        "dataset_4/keyframe_4",
    )
    for heading, names in family_names.items():
        lines.extend(
            [
                f"## {heading}",
                "| candidate | d1 | d2 | d3 | d7 | d4 | macro | decision |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for name in names:
            values = []
            for sequence in sequence_labels:
                summary = per_sequence.get(sequence, {}).get(name, {})
                value = summary.get("region", {}).get("spearman")
                values.append("n/a" if value is None else f"{float(value):.4f}")
            macro_value = macro.get(name, {}).get("region_spearman_macro")
            macro_text = "n/a" if macro_value is None else f"{float(macro_value):.4f}"
            lines.append(
                f"| {name} | {values[0]} | {values[1]} | {values[2]} | "
                f"{values[3]} | {values[4]} | {macro_text} | {decisions.get(name, 'n/a')} |"
            )
        lines.extend(
            [
                "",
                "Risk-coverage curves, frame distributions, bootstrap intervals, and coverage are in `metrics/per_sequence.json`.",
                "",
            ]
        )
    lines.append("## CANDIDATE DECISIONS")
    for name in sorted(decisions):
        lines.append(f"{name}: {decisions[name]}")
    lines.extend(
        [
            "",
            "## BEST INDEPENDENT CANDIDATES",
            f"best dynamics: {best.get('dynamics')}",
            f"best geometry: {best.get('geometry')}",
            f"best observation: {best.get('observation')}",
            "",
            "## COMPLEMENTARITY",
            json.dumps(result.get("complementarity", {}), indent=2, sort_keys=True),
            "",
            "## FAILURE STATES",
            json.dumps(result.get("failure_states", {}), indent=2, sort_keys=True),
            "",
            "## SHADOW ITERATION STUDY",
            f"performed: {'YES' if shadow.get('performed') else 'NO'}",
            f"details: {json.dumps(shadow, sort_keys=True)}",
            "",
            "## STEREO CONSENSUS FEASIBILITY",
            "L+R -> shared Z -> L_hat/R_hat",
            f"scientifically justified: {consensus.get('scientifically_justified', 'MAYBE')}",
            f"recommended: {consensus.get('recommended', 'P1c')}",
            f"reason: {consensus.get('reason')}",
            "implementation performed: NO",
            "",
            "## FAMILY DECISIONS",
            f"DYNAMICS: {family.get('dynamics')}",
            f"GEOMETRY: {family.get('geometry')}",
            f"OBSERVATION: {family.get('observation')}",
            "",
            "## OVERALL DECISION",
            str(result.get("overall_decision")),
            "",
            "## BASELINE INVARIANCE",
            json.dumps(invariance, indent=2, sort_keys=True),
            "",
            "## FINAL STATUS",
            f"P1B_COMPLETE: {'YES' if result.get('complete') else 'NO'}",
            f"U1_RESCUED: {result.get('status', {}).get('u1')}",
            f"U2_RESCUED: {result.get('status', {}).get('u2')}",
            f"U3_RESCUED: {result.get('status', {}).get('u3')}",
            f"U4_IMPROVED: {result.get('status', {}).get('u4')}",
            f"MULTI_CHANNEL_RELIABILITY_SUPPORTED: {result.get('status', {}).get('multi_channel')}",
            f"U4_OR_U4B_PRIMARY: {result.get('status', {}).get('u4_primary')}",
            "READY_FOR_NEW_CALIBRATION: NO",
            "READY_FOR_P3: NO",
            "READY_FOR_PROBABILISTIC_GAUSSIAN: NO",
            "FINAL_SCARED_UNTOUCHED: YES",
            "NEXT_SAFE_TASK: review P1b metrics and gate a separately preregistered follow-up",
        ]
    )
    return "\n".join(lines) + "\n"


def run_p1b(
    *,
    repo: Path,
    config_path: Path,
    output_path: Path,
    device: torch.device,
    max_samples: int | None = None,
    enable_shadow: bool | None = None,
) -> dict[str, object]:
    """Execute the complete development-only P1b measurement."""

    raw_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, Mapping):
        raise RuntimeError("P1b config must contain a mapping")
    raw = dict(raw_value)
    _require_config(raw)
    data_cfg = _config_mapping(raw, "data")
    identity_cfg = _config_mapping(raw, "identity")
    evaluation_cfg = _config_mapping(raw, "evaluation")
    baseline_cfg = _config_mapping(raw, "baseline")
    shadow_cfg = _config_mapping(raw, "shadow_iteration_study")

    from reliable_endo_gs.data.scared_manifest import load_scared_split_manifest

    split_path = repo / str(data_cfg["split"])
    data_config_path = repo / str(data_cfg["config"])
    split = load_scared_split_manifest(split_path)
    sequences = tuple(split.train) + tuple(split.validation)
    if tuple(split.test) or P1B_FINAL_SEQUENCES.intersection(tuple(split.test)):
        raise RuntimeError("P1b development split must not contain test/final sequences")
    validate_p1b_sequences(sequences)
    configured_sequences = tuple(str(item) for item in data_cfg["development_sequences"])
    if configured_sequences != P1B_DEVELOPMENT_SEQUENCES:
        raise RuntimeError("P1b config development sequence order mismatch")
    data_root = _resolve_required_env(str(identity_cfg["data_root_env"]))
    manifest_path = _resolve_required_env(str(identity_cfg["manifest_env"]))
    checkpoint_path = _resolve_required_env(str(identity_cfg["checkpoint_env"]))
    identities = _verify_identities(
        repo,
        raw,
        checkpoint_path,
        data_root,
        manifest_path,
        split_path,
        data_config_path,
    )
    if output_path.exists():
        raise RuntimeError(f"P1b refuses to overwrite output directory: {output_path}")
    output_path.mkdir(parents=True)
    (output_path / "samples").mkdir()
    (output_path / "metrics").mkdir()
    _write_json(output_path / "protocol.json", raw)
    _write_json(output_path / "source_identity.json", identities)

    from torch.utils.data import ConcatDataset, DataLoader

    from reliable_endo_gs.data.stage2 import ScaredStage2Dataset, stage2_collate_fn

    train_dataset = ScaredStage2Dataset(
        scared_root=data_root,
        split_manifest=split_path,
        split="train",
        phase=str(data_cfg["train_phase"]),
        test_every=int(data_cfg["test_every"]),
    )
    holdout_dataset = ScaredStage2Dataset(
        scared_root=data_root,
        split_manifest=split_path,
        split="validation",
        phase=str(data_cfg["validation_phase"]),
        test_every=int(data_cfg["test_every"]),
    )
    if tuple(train_dataset.keyframe_entries) != P1B_FIT_SEQUENCES:
        raise RuntimeError("P1b fit sequence contract mismatch")
    if tuple(holdout_dataset.keyframe_entries) != P1B_HOLDOUT_SEQUENCES:
        raise RuntimeError("P1b holdout sequence contract mismatch")
    dataset = ConcatDataset((train_dataset, holdout_dataset))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=stage2_collate_fn,
    )

    model = _load_model(repo, checkpoint_path, device)
    specs = {spec.name: spec for spec in candidate_specs()}
    candidate_rows_by_sequence: dict[str, dict[str, list[Mapping[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    complementarity_rows: dict[str, dict[str, list[Mapping[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cost_rows: list[dict[str, object]] = []
    shadow_rows: list[dict[str, object]] = []
    baseline_invariance: dict[str, object] = {
        "stage2_disparity_max_abs_diff": 0.0,
        "stage2_disparity_torch_equal": True,
        "gaussian_parameter_snapshot": None,
        "renderer_invocations_by_p1b": 0,
        "renderer_mutated": False,
    }
    sample_count = 0
    expected_count = len(dataset)
    shadow_enabled = (
        bool(shadow_cfg.get("enabled", False)) if enable_shadow is None else enable_shadow
    )
    max_shadow_iterations = int(shadow_cfg.get("max_iterations", 6))
    shadow_k = int(shadow_cfg.get("requested_iterations", max_shadow_iterations))
    if shadow_enabled and shadow_k > max_shadow_iterations:
        raise RuntimeError("requested shadow iterations exceed preregistered maximum")
    high_error_threshold = float(evaluation_cfg["high_error_threshold_disparity_px"])
    coverage_levels = tuple(float(value) for value in evaluation_cfg["coverage_levels"])
    bootstrap_repeats = int(evaluation_cfg["bootstrap_repeats"])
    bootstrap_seed = int(evaluation_cfg["bootstrap_seed"])
    region_grid = tuple(int(value) for value in evaluation_cfg["region_grid"])
    if coverage_levels != P1B_COVERAGE_LEVELS or region_grid != (4, 4):
        raise RuntimeError("P1b metric grid/coverage differs from the preregistered protocol")

    with (output_path / "metrics/per_frame.jsonl").open("w", encoding="utf-8") as metrics_file:
        with torch.no_grad():
            for batch in loader:
                sequence = _sequence_from_sample(batch["name"][0])
                if sequence not in P1B_DEVELOPMENT_SEQUENCES:
                    raise RuntimeError(f"unexpected P1b sequence {sequence}")
                batch = _move_batch(batch, device)
                left_image = batch["lmain"]["img"]
                right_image = batch["rmain"]["img"]

                _sync(device)
                native_start = time.perf_counter()
                baseline_snapshot = _native_snapshot(model, batch)
                _sync(device)
                native_seconds = time.perf_counter() - native_start
                baseline_disparity = baseline_snapshot["flow_pred"]

                _sync(device)
                left_memory_start = _vram_start(device)
                left_start = time.perf_counter()
                left_iterations = extract_model_iterations(
                    model,
                    left_image,
                    right_image,
                    iterations=int(baseline_cfg["validation_iterations"]),
                )
                _sync(device)
                left_seconds = time.perf_counter() - left_start
                left_vram = _vram_finish(device, left_memory_start)
                _sync(device)
                right_memory_start = _vram_start(device)
                right_start = time.perf_counter()
                right_iterations = extract_model_iterations(
                    model,
                    right_image,
                    left_image,
                    iterations=int(baseline_cfg["validation_iterations"]),
                )
                _sync(device)
                right_seconds = time.perf_counter() - right_start
                right_vram = _vram_finish(device, right_memory_start)

                baseline_diff = _max_abs_difference(baseline_disparity, left_iterations[:, -1:])
                baseline_invariance["stage2_disparity_max_abs_diff"] = max(
                    float(baseline_invariance["stage2_disparity_max_abs_diff"]), baseline_diff
                )
                if baseline_diff != 0.0:
                    baseline_invariance["stage2_disparity_torch_equal"] = False

                left_disparity = left_iterations[:, -1:]
                right_disparity = right_iterations[:, -1:]
                left_valid = torch.isfinite(left_disparity) & (left_disparity > 0.0)
                right_valid = torch.isfinite(right_disparity) & (right_disparity > 0.0)
                post_memory_start = _vram_start(device)
                trajectory_start = time.perf_counter()
                trajectory = compute_trajectory_features(left_iterations, left_valid)
                _sync(device)
                trajectory_seconds = time.perf_counter() - trajectory_start
                geometry_start = time.perf_counter()
                geometry = compute_geometry_features(
                    left_disparity,
                    right_disparity,
                    left_valid,
                    right_valid,
                )
                _sync(device)
                geometry_seconds = time.perf_counter() - geometry_start
                observation_start = time.perf_counter()
                observation = compute_photometric_features(
                    left_image,
                    right_image,
                    left_disparity,
                    right_disparity,
                    geometry,
                    left_valid,
                    right_valid,
                )
                _sync(device)
                observation_seconds = time.perf_counter() - observation_start
                postprocess_vram = _vram_finish(device, post_memory_start)

                signals = dict(trajectory.signals)
                signals.update(geometry.signals)
                signals.update(observation.signals)
                absolute_error, target_valid = oracle_disparity_error(
                    left_disparity,
                    batch["lmain"]["disp"],
                    batch["lmain"]["mask"] >= 0.5,
                )
                frame_metrics: dict[str, Mapping[str, object]] = {}
                for name, spec in specs.items():
                    if name not in signals:
                        raise RuntimeError(f"P1b signal implementation omitted {name}")
                    frame_metrics[name] = evaluate_frame_signal(
                        signals[name],
                        absolute_error,
                        target_valid,
                        spec,
                        high_error_threshold=high_error_threshold,
                        coverage_levels=coverage_levels,
                        grid=region_grid,
                    )
                    candidate_rows_by_sequence[sequence][name].append(frame_metrics[name])
                    complementarity_rows[sequence][name].append(frame_metrics[name])

                sample_id = str(batch["name"][0])
                sample_dir = output_path / "samples" / sequence
                sample_dir.mkdir(parents=True, exist_ok=True)
                sample_payload: dict[str, np.ndarray] = {
                    "disparity": left_disparity[0].detach().cpu().numpy(),
                    "disparity_iterations": left_iterations[0].detach().cpu().numpy(),
                    "right_disparity_iterations": right_iterations[0].detach().cpu().numpy(),
                    "inference_valid_mask": left_valid[0].detach().cpu().numpy(),
                    "right_inference_valid_mask": right_valid[0].detach().cpu().numpy(),
                    "ground_truth_disparity": batch["lmain"]["disp"][0].detach().cpu().numpy(),
                    "ground_truth_valid_mask": target_valid[0].detach().cpu().numpy(),
                    "absolute_disparity_error": absolute_error[0].detach().cpu().numpy(),
                    "u3b_strict_valid_mask": geometry.support.valid_mask[0].cpu().numpy(),
                    "u3b_visibility_mask": geometry.visibility_mask[0].cpu().numpy(),
                    "u3b_occlusion_or_unsupported_mask": geometry.occlusion_or_unsupported_mask[0]
                    .cpu()
                    .numpy(),
                    "u4b_left_valid_mask": observation.left_valid_mask[0].cpu().numpy(),
                    "u4b_right_valid_mask": observation.right_valid_mask[0].cpu().numpy(),
                    "u4b_bidirectional_valid_mask": observation.bidirectional_valid_mask[0]
                    .cpu()
                    .numpy(),
                    "u4b_specular_exclusion_mask": observation.specular_exclusion_mask[0]
                    .cpu()
                    .numpy(),
                }
                for name, signal in signals.items():
                    sample_payload[name] = signal.score[0].detach().cpu().numpy()
                    sample_payload[f"{name}_valid_mask"] = signal.valid_mask[0].cpu().numpy()
                np.savez_compressed(
                    sample_dir / f"{sample_id.rsplit('/', 1)[-1]}.npz", **sample_payload
                )

                if sample_count == 0:
                    before = baseline_snapshot
                    after = _native_snapshot(model, batch)
                    baseline_invariance["gaussian_parameter_snapshot"] = _snapshot_differences(
                        before, after
                    )

                shadow_payload: dict[str, object] | None = None
                if shadow_enabled:
                    _sync(device)
                    shadow_start = time.perf_counter()
                    shadow_iterations = extract_model_iterations(
                        model, left_image, right_image, iterations=shadow_k
                    )
                    _sync(device)
                    shadow_seconds = time.perf_counter() - shadow_start
                    if shadow_iterations.shape[1] != shadow_k:
                        raise RuntimeError("upstream shadow rollout returned an unexpected K")
                    if _max_abs_difference(shadow_iterations[:, :3], left_iterations) != 0.0:
                        raise RuntimeError("shadow rollout changed the first three predictions")
                    diagnostics = shadow_diagnostics(shadow_iterations)
                    shadow_payload = {
                        "sample_id": sample_id,
                        "seconds": shadow_seconds,
                        "diagnostics": {
                            key: float(value.mean().item()) for key, value in diagnostics.items()
                        },
                    }
                    shadow_rows.append(shadow_payload)

                cost_rows.append(
                    {
                        "sample_id": sample_id,
                        "sequence": sequence,
                        "native_baseline_seconds": native_seconds,
                        "left_forward_seconds": left_seconds,
                        "right_forward_seconds": right_seconds,
                        "left_forward_vram": left_vram,
                        "right_forward_vram": right_vram,
                        "postprocess_seconds": {
                            "dynamics": trajectory_seconds,
                            "geometry": geometry_seconds,
                            "observation": observation_seconds,
                        },
                        "postprocess_vram": postprocess_vram,
                        "coverage": {
                            name: float(frame_metrics[name]["valid_fraction"])
                            for name in frame_metrics
                        },
                        "shadow": shadow_payload,
                    }
                )
                metrics_file.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "sequence": sequence,
                            "metrics": frame_metrics,
                            "cost": cost_rows[-1],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                sample_count += 1
                if max_samples is not None and sample_count >= max_samples:
                    break

    per_sequence: dict[str, dict[str, object]] = {}
    for sequence, rows_by_name in candidate_rows_by_sequence.items():
        per_sequence[sequence] = {
            name: summarize_candidate_rows(
                rows,
                bootstrap_repeats=bootstrap_repeats,
                bootstrap_seed=bootstrap_seed,
            )
            for name, rows in rows_by_name.items()
        }
    candidate_decisions = {
        name: classify_candidate(
            {
                sequence: per_sequence.get(sequence, {}).get(name, {})
                for sequence in P1B_HOLDOUT_SEQUENCES
            }
        )
        for name in specs
    }
    macro: dict[str, object] = {}
    for name in specs:
        sequence_summaries = [
            per_sequence[sequence][name]
            for sequence in per_sequence
            if name in per_sequence[sequence]
        ]
        macro[name] = {
            "sequences": len(sequence_summaries),
            "region_spearman_macro": _mean(
                [
                    float(summary["region"]["spearman"])
                    for summary in sequence_summaries
                    if summary["region"].get("spearman") is not None
                ]
            ),
            "pixel_spearman_macro": _mean(
                [
                    float(summary["pixel"]["spearman_mean"])
                    for summary in sequence_summaries
                    if summary["pixel"].get("spearman_mean") is not None
                ]
            ),
            "frame_rho_macro": _mean(
                [
                    float(summary["frame"]["rho_mean"])
                    for summary in sequence_summaries
                    if summary["frame"].get("rho_mean") is not None
                ]
            ),
        }

    sequence_bootstrap = {
        name: bootstrap_mean(
            [
                float(per_sequence[sequence][name]["region"]["spearman"])
                for sequence in per_sequence
                if per_sequence[sequence][name]["region"].get("spearman") is not None
            ],
            repeats=bootstrap_repeats,
            seed=bootstrap_seed + 100,
        )
        for name in specs
    }

    best = _best_by_family(per_sequence, specs)
    family = _family_decisions(candidate_decisions, best=best)
    overall = _overall_decision(family, candidate_decisions)
    complementarity: dict[str, object] = {"best_candidates": best, "pairwise_rank_correlations": {}}
    best_names = [name for name in best.values() if name is not None]
    for sequence, rows_by_name in complementarity_rows.items():
        complementarity["pairwise_rank_correlations"][sequence] = _pairwise_region_correlations(
            rows_by_name, best_names
        )
    complementarity["pairwise_rank_correlations"]["development_all"] = (
        _pairwise_region_correlations(
            {
                name: [
                    row
                    for sequence in P1B_DEVELOPMENT_SEQUENCES
                    for row in complementarity_rows.get(sequence, {}).get(name, ())
                ]
                for name in best_names
            },
            best_names,
        )
    )
    failure_states = _state_analysis(
        candidate_rows_by_sequence,
        best,
        high_error_threshold=high_error_threshold,
    )
    cost_summary = {
        "frames": len(cost_rows),
        "native_baseline_seconds_mean": _mean(
            [float(row["native_baseline_seconds"]) for row in cost_rows]
        ),
        "left_forward_seconds_mean": _mean(
            [float(row["left_forward_seconds"]) for row in cost_rows]
        ),
        "right_forward_seconds_mean": _mean(
            [float(row["right_forward_seconds"]) for row in cost_rows]
        ),
        "postprocess_seconds_mean": {
            family_name: _mean(
                [float(row["postprocess_seconds"][family_name]) for row in cost_rows]
            )
            for family_name in ("dynamics", "geometry", "observation")
        },
        "vram_delta_mean_bytes": {
            "left_forward_peak_allocated": _mean(
                [
                    float(row["left_forward_vram"]["peak_allocated_delta_bytes"])
                    for row in cost_rows
                    if row["left_forward_vram"]["peak_allocated_delta_bytes"] is not None
                ]
            ),
            "right_forward_peak_allocated": _mean(
                [
                    float(row["right_forward_vram"]["peak_allocated_delta_bytes"])
                    for row in cost_rows
                    if row["right_forward_vram"]["peak_allocated_delta_bytes"] is not None
                ]
            ),
            "postprocess_peak_allocated": _mean(
                [
                    float(row["postprocess_vram"]["peak_allocated_delta_bytes"])
                    for row in cost_rows
                    if row["postprocess_vram"]["peak_allocated_delta_bytes"] is not None
                ]
            ),
        },
        "shadow_extra_forward_seconds_mean": _mean(
            [float(row["shadow"]["seconds"]) for row in cost_rows if row["shadow"] is not None]
        ),
    }
    source_after = _source_state(repo)
    source_before = identities
    source_unchanged = source_before.get("source_sha") == source_after.get("source_sha")

    def rescue_status(names: Sequence[str]) -> str:
        results = [candidate_decisions[name] for name in names]
        if "PASS" in results:
            return "YES"
        if "MARGINAL" in results:
            return "PARTIAL"
        return "NO"

    status = {
        "u1": rescue_status(
            (
                "u1b_path",
                "u1b_decay",
                "u1b_decay_deviation",
                "u1b_acceleration",
                "u1b_directional_disagreement",
                "u1b_directional_agreement",
            )
        ),
        "u2": rescue_status(
            ("u2b_normalized_dispersion", "u2b_path_efficiency", "u2b_oscillation")
        ),
        "u3": rescue_status(("u3b_valid_absolute", "u3b_relative", "u3b_visibility_aware")),
        "u4": "YES"
        if any(candidate_decisions.get(name) == "PASS" for name in specs if name.startswith("u4b"))
        else "NO",
        "multi_channel": "YES"
        if sum(value.endswith("-A") for value in family.values()) >= 2
        else "NO",
        "u4_primary": best.get("observation") or "u4_historical",
    }
    result: dict[str, object] = {
        "complete": sample_count == expected_count and source_unchanged,
        "samples": sample_count,
        "expected_samples": expected_count,
        "final_samples_accessed": "NO",
        "source": {"before": source_before, "after": source_after, "unchanged": source_unchanged},
        "identity": identities,
        "protocol": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "remote_sha": source_before.get("remote_sha"),
        },
        "per_sequence": per_sequence,
        "macro": macro,
        "bootstrap": {"sequence_cluster": sequence_bootstrap},
        "decisions": candidate_decisions,
        "best_candidates": best,
        "family_decisions": family,
        "overall_decision": overall,
        "complementarity": complementarity,
        "failure_states": failure_states,
        "cost": cost_summary,
        "cost_rows": cost_rows,
        "baseline_invariance": baseline_invariance,
        "shadow_iteration_study": {
            "performed": shadow_enabled,
            "K": shadow_k if shadow_enabled else None,
            "rows": shadow_rows,
            "extra_compute_signal": True,
            "substituted_into_reconstruction": False,
        },
        "stereo_consensus_feasibility": {
            "scientifically_justified": "MAYBE",
            "recommended": "P1c",
            "reason": "shared-scene residuals are distinct from U4 but need a separate capacity and leakage gate",
            "implementation": "NO",
        },
        "status": status,
    }
    _write_json(output_path / "metrics/per_sequence.json", per_sequence)
    _write_json(output_path / "metrics/macro.json", macro)
    _write_json(output_path / "metrics/bootstrap.json", {"sequence_cluster": sequence_bootstrap})
    _write_json(output_path / "metrics/decisions.json", candidate_decisions)
    _write_json(output_path / "metrics/complementarity.json", complementarity)
    _write_json(output_path / "metrics/failure_states.json", failure_states)
    _write_json(output_path / "metrics/cost.json", cost_summary)
    _write_json(output_path / "metrics/result.json", result)
    (output_path / "P1b_report.md").write_text(_render_report(result), encoding="utf-8")
    return result
