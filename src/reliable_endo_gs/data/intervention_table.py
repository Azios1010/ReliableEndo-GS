"""Immutable offline outcome-table schema; medical images remain external."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.contracts.interventions import ACTION_NAMES, COMPONENT_NAMES


@dataclass(frozen=True)
class OutcomeTable:
    """CPU features [R,F], changes [R,A,C], bool observation/eligibility [R,A]."""

    metadata: dict[str, Any]
    records: list[dict[str, Any]]
    features: torch.Tensor
    changes: torch.Tensor
    observed: torch.Tensor
    eligible: torch.Tensor
    costs: torch.Tensor
    slices: tuple[tuple[int, int], ...]


def load_outcome_table(
    directory: Path, expected_role: str, *, feature_names: tuple[str, ...]
) -> OutcomeTable:
    """Reject incomplete, duplicated, unpinned, or contaminated tables."""
    metadata = json.loads((directory / "result.json").read_text())
    content = (directory / "samples.jsonl").read_bytes()
    if (
        metadata.get("status") != "metric_outcome_table_complete"
        or metadata.get("table_schema") != "metric_outcome_table.v1"
        or metadata.get("collection_role") != expected_role
        or metadata.get("outcome_sha256") != hashlib.sha256(content).hexdigest()
        or metadata.get("feature_names") != list(feature_names)
        or metadata.get("component_names") != list(COMPONENT_NAMES)
        or metadata.get("action_names") != list(ACTION_NAMES)
        or metadata.get("weights_unchanged") is not True
        or metadata.get("teacher_inference_executed") is not False
        or metadata.get("training_executed") is not False
    ):
        raise ValueError("invalid frozen metric-outcome table identity")
    records = [json.loads(line) for line in content.splitlines()]
    ids = [r["sample_id"] for r in records]
    if (
        not ids
        or len(ids) != len(set(ids))
        or metadata.get("samples") != len(ids)
        or ids != metadata.get("sample_ids")
        or any(sid.split("/")[0] not in {f"dataset_{i}" for i in range(1, 8)} for sid in ids)
        or metadata.get("sample_ids_sha256") != hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    ):
        raise ValueError("invalid original-SCARED table samples")
    features, changes, observed, eligibility, costs, slices = [], [], [], [], [], []
    start = 0
    for record in records:
        f = torch.tensor(record["features_raw"], dtype=torch.float32)
        valid_regions = record["valid_regions"]
        if (
            f.ndim != 2
            or f.shape[1] != len(feature_names)
            or not bool(torch.isfinite(f).all())
            or f.shape[0] != len(valid_regions)
            or any(type(v) is not bool for v in valid_regions)
        ):
            raise ValueError("invalid pre-action evidence schema")
        values = torch.zeros(f.shape[0], len(ACTION_NAMES), len(COMPONENT_NAMES))
        mask = torch.zeros(f.shape[0], len(ACTION_NAMES), dtype=torch.bool)
        cost = torch.zeros(f.shape[0], len(ACTION_NAMES))
        seen = set()
        for row in record["outcomes"]:
            ri, name = row["region_id"], row["action"]
            if type(ri) is not int or not 0 <= ri < f.shape[0] or name not in ACTION_NAMES:
                raise ValueError("unknown action/region in table")
            ai = ACTION_NAMES.index(name)
            if (ri, ai) in seen or not valid_regions[ri]:
                raise ValueError("duplicate or ineligible intervention")
            seen.add((ri, ai))
            samples = row["action_ms_samples"]
            if (
                len(samples) != metadata["timing_repeats"]
                or not all(
                    isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in samples
                )
                or row["action_ms_max"] != max(samples)
            ):
                raise ValueError("invalid measured action timing samples")
            cost[ri, ai] = max(samples)
            vector = row["metric_changes"]
            available = row["metric_observation_available"]
            if type(available) is not bool or available != (vector is not None):
                raise ValueError("metric availability must be explicit")
            if available:
                if len(vector) != len(COMPONENT_NAMES) or not all(math.isfinite(x) for x in vector):
                    raise ValueError("invalid component vector")
                values[ri, ai] = torch.tensor(vector)
                mask[ri, ai] = True
                if name == "gaussian" and bool(
                    values[ri, ai, [0, 1, 2, 3, 4, 11]].abs().gt(1e-6).any()
                ):
                    raise ValueError("Gaussian repair violated pinned disparity invariance")
        expected = {(ri, ai) for ri, ok in enumerate(valid_regions) if ok for ai in range(3)}
        if seen != expected:
            raise ValueError("missing interventions; partial tables cannot train")
        for key in ("base_ms", "base_ms_without_matching", "feature_ms"):
            if not math.isfinite(record[key]) or record[key] < 0:
                raise ValueError("invalid base/feature timing")
        features.append(f)
        changes.append(values)
        observed.append(mask)
        costs.append(cost)
        eligibility.append(torch.tensor(valid_regions)[:, None].expand(-1, len(ACTION_NAMES)))
        slices.append((start, start + f.shape[0]))
        start += f.shape[0]
    return OutcomeTable(
        metadata,
        records,
        torch.cat(features),
        torch.cat(changes),
        torch.cat(observed),
        torch.cat(eligibility),
        torch.cat(costs),
        tuple(slices),
    )
