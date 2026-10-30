"""Resume immutable outcome shards without mixing label generators or splits."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.routing.cost import capture_runtime_identity
from reliable_endo_gs.training.metric_quality import OfflineMetricTruth
from reliable_endo_gs.training.selector_outcomes import (
    SelectorOutcomes,
    collect_selector_outcomes,
    load_selector_outcomes,
)
from reliable_endo_gs.training.student_learning import StudentLearner
from reliable_endo_gs.utils.io import write_json_atomic


def collect_selector_shards(
    learner: StudentLearner,
    sample_ids: tuple[str, ...],
    sample_provider: Callable[[str], ScaredA00Sample],
    gt_provider: Callable[[str], OfflineMetricTruth],
    *,
    provenance: Mapping[str, object],
    output_dir: Path,
    shard_size: int = 64,
    resume: bool = False,
    store_feature_bank: bool = False,
) -> dict[str, Any]:
    """Completed shards are verified/reused; interrupted shards are archived."""
    repo = Path(__file__).resolve().parents[3]
    if (
        output_dir.resolve() == repo
        or repo in output_dir.resolve().parents
        or not sample_ids
        or len(set(sample_ids)) != len(sample_ids)
        or type(shard_size) is not int
        or not 1 <= shard_size <= 256
        or learner.intervention_config is None
    ):
        raise ValueError("invalid external frozen shard collection")
    device = next(learner.model.parameters()).device
    seal = {
        **dict(provenance),
        "schema": "selector_shard_collection.v1",
        "sample_ids": list(sample_ids),
        "shard_size": shard_size,
        "action_identity": learner.frozen_action_identity(),
        "intervention_config": learner.intervention_config.to_dict(),
        "runtime": capture_runtime_identity(device),
        **({"store_feature_bank": True} if store_feature_bank else {}),
    }
    seal = json.loads(json.dumps(seal, allow_nan=False))
    normalized_provenance = json.loads(json.dumps(dict(provenance), allow_nan=False))
    if output_dir.exists():
        if (
            not resume
            or not (output_dir / "collection.json").is_file()
            or json.loads((output_dir / "collection.json").read_text()) != seal
        ):
            raise ValueError(
                "resume requires exact source/checkpoint/cohort/runtime shard identity"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(output_dir / "collection.json", seal)
    tables = []
    reused = 0
    completed = 0
    for start in range(0, len(sample_ids), shard_size):
        ids = sample_ids[start : start + shard_size]
        directory = output_dir / f"shard-{start:06d}"
        if directory.exists() and not (directory / "result.json").is_file():
            if not resume:
                raise ValueError("partial shard requires explicit resume")
            archived = output_dir / (directory.name + "-interrupted-" + uuid.uuid4().hex[:12])
            directory.rename(archived)
            print(
                json.dumps({"event": "selector_shard_archived", "path": str(archived)}), flush=True
            )
        if directory.exists():
            reused += len(ids)
        else:
            collect_selector_outcomes(
                learner,
                (sample_provider(s) for s in ids),
                gt_provider,
                collection_role="train",
                provenance=provenance,
                output_dir=directory,
                **({"store_feature_bank": True} if store_feature_bank else {}),
            )
        table = load_selector_outcomes(directory, "train", allow_empty=True)
        if (
            table.metadata["sample_ids"] != list(ids)
            or table.metadata["action_identity"] != seal["action_identity"]
            or table.metadata["intervention_config"] != seal["intervention_config"]
            or any(table.metadata.get(k) != v for k, v in normalized_provenance.items())
        ):
            raise ValueError("frozen shard provenance mismatch")
        tables.append(table)
        completed += len(ids)
        print(
            json.dumps(
                {
                    "event": "selector_shards_progress",
                    "completed_samples": completed,
                    "total_samples": len(sample_ids),
                    "reused_samples": reused,
                }
            ),
            flush=True,
        )
    # Rebuild the aggregate atomically from verified shards, preserving all unknown rows.
    content = "".join(
        json.dumps(row, allow_nan=False) + "\n" for table in tables for row in table.records
    )
    from reliable_endo_gs.utils.io import _write_text_atomic

    _write_text_atomic(output_dir / "samples.jsonl", content)
    result = {
        **tables[0].metadata,
        "sample_ids": list(sample_ids),
        "samples_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "shards": [
            {"path": f"shard-{i * shard_size:06d}", "samples_sha256": t.metadata["samples_sha256"]}
            for i, t in enumerate(tables)
        ],
    }
    write_json_atomic(output_dir / "result.json", result)
    load_selector_outcomes(output_dir, "train", allow_empty=True)
    return {
        "completed_samples": completed,
        "reused_samples": reused,
        "shards": len(tables),
        "outcome_sha256": result["samples_sha256"],
    }


def load_verified_selector_shards(directory: Path) -> SelectorOutcomes:
    """Verify aggregate against every immutable shard before any fit-only reuse."""
    table = load_selector_outcomes(directory, "train", allow_empty=True)
    rows = []
    for member in table.metadata.get("shards", []):
        name = member.get("path")
        if not isinstance(name, str) or Path(name).name != name or not name.startswith("shard-"):
            raise ValueError("unsafe outcome shard identity")
        shard = load_selector_outcomes(directory / name, "train", allow_empty=True)
        if member.get("samples_sha256") != shard.metadata["samples_sha256"] or any(
            shard.metadata.get(k) != table.metadata.get(k)
            for k in (
                "action_identity",
                "operative_action_identity",
                "candidate_protocol_identity",
                "intervention_config",
                "feature_names",
                "feature_bank_names",
                "deployment_costs_ms",
                "checkpoint_sha256",
                "source_sha256",
                "partition_sha256",
                "study_config_hash",
                "gate_a_report_sha256",
            )
        ):
            raise ValueError("shared outcome shard identity mismatch")
        rows.extend(shard.records)
    if not table.metadata.get("shards") or rows != table.records:
        raise ValueError("shared aggregate differs from verified outcome shards")
    return table
