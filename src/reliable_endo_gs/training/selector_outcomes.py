"""One frozen label generator, deployment-only features and separate offline GT."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from reliable_endo_gs.actions.repair import eligible
from reliable_endo_gs.config.intervention import InterventionConfig
from reliable_endo_gs.data.scared_a00 import ScaredA00Sample
from reliable_endo_gs.regions.masks import MaskRegion
from reliable_endo_gs.routing.balanced import BalancedSelector
from reliable_endo_gs.routing.metric_selection import module_state_sha256
from reliable_endo_gs.routing.tokens import selector_feature_names
from reliable_endo_gs.training.balanced_interventions import BalancedHarness
from reliable_endo_gs.training.metric_quality import (
    OfflineMetricTruth,
    metric_risk_targets,
    metric_targets,
)
from reliable_endo_gs.training.student_learning import StudentLearner, evaluation_mode
from reliable_endo_gs.utils.io import write_json_atomic


@dataclass(frozen=True)
class SelectorOutcomes:
    """Known CPU features [R,F], utilities/local/harm [R,3]; full logs retain unknowns."""

    metadata: dict[str, Any]
    records: list[dict[str, Any]]
    features: torch.Tensor
    global_gain: torch.Tensor
    local_gain: torch.Tensor
    harms: torch.Tensor
    sample_ids: tuple[str, ...]
    risk_components: torch.Tensor | None = None

    def observation_summary(self) -> dict[str, Any]:
        """Report exclusion by keyframe without interpreting unknown as safe.

        Historical tables lacking diagnostic fields remain readable; their
        missing-reason counts are explicitly recorded as unavailable evidence.
        """
        counts: dict[str, Counter[str]] = {}
        unknown_reasons: dict[str, Counter[str]] = {}
        for record in self.records:
            key = record["sample_id"].rsplit("/", 1)[0]
            count = counts.setdefault(key, Counter())
            count["images"] += 1
            regions = record.get("interventions", [])
            count["regions"] += len(regions)
            known = sum(all(a["selector_label_available"] for a in r["actions"]) for r in regions)
            count["known_regions"] += known
            count["unknown_regions"] += len(regions) - known
            count["images_without_known_regions"] += known == 0
            for region in regions:
                for action in region["actions"]:
                    if not action["selector_label_available"]:
                        diagnostics = action.get("metric_diagnostics") or {}
                        unknown_reasons.setdefault(action["action"], Counter()).update(
                            diagnostics.get("metric_unavailable_reasons")
                            or ["not_recorded_in_artifact"]
                        )
        totals: Counter[str] = Counter()
        for count in counts.values():
            totals.update(count)
        return {
            **dict(totals),
            "known_training_rows": len(self.sample_ids),
            "by_keyframe": {k: dict(v) for k, v in sorted(counts.items())},
            "unknown_reasons": {k: dict(v) for k, v in sorted(unknown_reasons.items())},
            "unknown_is_safe": False,
            "coverage_is_deployment_feature": False,
        }

    def view(self, ids: tuple[str, ...]) -> SelectorOutcomes:
        selected = set(ids)
        if (
            not ids
            or len(ids) != len(selected)
            or not selected <= {r["sample_id"] for r in self.records}
        ):
            raise ValueError("internal TRAIN view contains unknown sample identities")
        index = torch.tensor(
            [i for i, s in enumerate(self.sample_ids) if s in selected], dtype=torch.long
        )
        if not len(index):
            raise ValueError("internal TRAIN role has no known metric labels")
        return SelectorOutcomes(
            self.metadata,
            [r for r in self.records if r["sample_id"] in selected],
            self.features[index],
            self.global_gain[index],
            self.local_gain[index],
            self.harms[index],
            tuple(self.sample_ids[i] for i in index.tolist()),
            None if self.risk_components is None else self.risk_components[index],
        )


def collect_selector_outcomes(
    learner: StudentLearner,
    samples: Iterable[ScaredA00Sample],
    gt_provider: Callable[[str], OfflineMetricTruth],
    *,
    collection_role: str,
    provenance: Mapping[str, object],
    output_dir: Path,
) -> dict[str, Any]:
    """Execute shared real STOP/STEREO/GAUSSIAN; no selector or optimizer update.

    Proposals/features are built before the GT callback. Only the offline harness
    receives truth; neither DeploymentStereoBatch nor actions receive it.
    """
    cfg = learner.intervention_config
    if (
        collection_role not in ("train", "development")
        or cfg is None
        or cfg.metric_quality is None
        or not isinstance(learner.harness, BalancedHarness)
        or not isinstance(learner.selector, BalancedSelector)
    ):
        raise ValueError("frozen outcomes require the physical balanced selector protocol")
    repo = Path(__file__).resolve().parents[3]
    if output_dir.resolve() == repo or repo in output_dir.resolve().parents:
        raise ValueError("selector outcomes must stay outside repository")
    required = {
        "checkpoint_sha256",
        "checkpoint_source_sha256",
        "source_sha256",
        "training_manifest_hash",
        "development_manifest_hash",
        "role_hash",
        "target_mask_index_sha256",
    }
    if not required <= provenance.keys():
        raise ValueError("missing frozen outcome provenance")
    output_dir.mkdir(parents=True, exist_ok=False)
    identity = learner.frozen_action_identity()
    selector_hash = module_state_sha256(learner.selector)
    flags = [
        (p, p.requires_grad)
        for m in (learner.model, learner.actions, learner.selector)
        for p in m.parameters()
    ]
    ids = []
    try:
        for p, _ in flags:
            p.requires_grad_(False)
        with (
            torch.no_grad(),
            evaluation_mode(learner.model, learner.actions, learner.selector),
            (output_dir / "samples.jsonl").open("x") as stream,
        ):
            for sample in samples:
                sid = sample.sample_id
                if (
                    sample.supervision is not None
                    or sample.data_role != "final_train"
                    or sid.split("/")[0] not in {f"dataset_{i}" for i in range(1, 8)}
                    or sid in ids
                ):
                    raise ValueError(
                        "frozen outcome inputs must be unique GT-free original TRAIN samples"
                    )
                ids.append(sid)
                state = learner.state(sample)
                regions = tuple(
                    r for r in learner.candidate_regions(state, sid) if eligible(state, r)
                )
                features = (
                    learner.selection_features(state, regions)
                    if regions
                    else state.batch.left.new_empty((0, len(selector_feature_names(cfg))))
                )
                truth = gt_provider(sid)
                truth.validate(sid, state.batch.left[:, :1].shape)
                truth = replace(
                    truth,
                    disparity=truth.disparity.to(features.device),
                    valid_mask=truth.valid_mask.to(features.device),
                )
                tables = [
                    learner.harness.collect(
                        state, r, learner.actions, learner.model, metric_truth=truth
                    )
                    for r in regions
                ]
                record = {
                    "sample_id": sid,
                    "gt_resampling_identity": truth.resampling_identity,
                    "features_raw": features.cpu().tolist(),
                    "valid_regions": [True] * len(regions),
                    "interventions": [
                        {
                            "region_id": r.index,
                            "core_yxyx": list(r.core),
                            "write_pixel_count": r.pixel_count,
                            "write_flat_indices": list(r.flat_indices)
                            if isinstance(r, MaskRegion)
                            else None,
                            "actions": [
                                {
                                    "action": row.action.name,
                                    "global_gain": row.utility if row.metric_observed else None,
                                    "local_gain": row.local_gain if row.metric_observed else None,
                                    "harm": row.harmful if row.metric_observed else None,
                                    **(
                                        {
                                            "risk_local": row.risk_local,
                                            "risk_global": row.risk_global,
                                        }
                                        if cfg.selector_risk_mode == "local_global"
                                        else {}
                                    ),
                                    "metric_changes": row.metric_changes,
                                    "selector_label_available": row.metric_observed,
                                    "metric_diagnostics": row.metric_diagnostics,
                                    "cost_ms": row.cost_ms,
                                }
                                for row in table
                            ],
                        }
                        for r, table in zip(regions, tables, strict=True)
                    ],
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                if len(ids) == 1 or len(ids) % 8 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "frozen_selector_outcomes_progress",
                                "role": collection_role,
                                "samples": len(ids),
                            }
                        ),
                        flush=True,
                    )
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
    if (
        not ids
        or learner.frozen_action_identity() != identity
        or module_state_sha256(learner.selector) != selector_hash
    ):
        raise ValueError("empty collection or frozen model/action/selector mutation")
    result = {
        **dict(provenance),
        "schema": "frozen_selector_outcomes.v1",
        "collection_role": collection_role,
        "sample_ids": ids,
        "action_identity": identity,
        "intervention_config": cfg.to_dict(),
        "feature_names": list(selector_feature_names(cfg)),
        "samples_sha256": hashlib.sha256((output_dir / "samples.jsonl").read_bytes()).hexdigest(),
        "weights_unchanged": True,
        "teacher_inference_executed": False,
        "selector_executed": False,
    }
    write_json_atomic(output_dir / "result.json", result)
    return result


def load_selector_outcomes(
    directory: Path, expected_role: str, *, allow_empty: bool = False
) -> SelectorOutcomes:
    """Verify immutable identity and canonical labels, preserving unknown log rows."""
    meta = json.loads((directory / "result.json").read_text())
    content = (directory / "samples.jsonl").read_bytes()
    cfg = InterventionConfig(**meta["intervention_config"])
    if (
        meta.get("schema") != "frozen_selector_outcomes.v1"
        or meta.get("collection_role") != expected_role
        or meta.get("weights_unchanged") is not True
        or meta.get("selector_executed") is not False
        or meta.get("teacher_inference_executed") is not False
        or meta.get("samples_sha256") != hashlib.sha256(content).hexdigest()
        or meta.get("feature_names") != list(selector_feature_names(cfg))
    ):
        raise ValueError("invalid frozen selector outcome identity")
    records = [json.loads(s) for s in content.splitlines()]
    ids = [r["sample_id"] for r in records]
    if (
        ids != meta["sample_ids"]
        or len(ids) != len(set(ids))
        or not ids
        or any(s.split("/")[0] not in {f"dataset_{i}" for i in range(1, 8)} for s in ids)
    ):
        raise ValueError("invalid frozen original-SCARED sample identities")
    features = []
    gains = []
    locals_ = []
    harms = []
    known_ids = []
    components = []
    for record in records:
        raw = record["features_raw"]
        if len(raw) != len(record["interventions"]) or len(raw) != len(record["valid_regions"]):
            raise ValueError("frozen feature/intervention correspondence mismatch")
        for f, region in zip(raw, record["interventions"], strict=True):
            if len(f) != len(selector_feature_names(cfg)) or not bool(
                torch.isfinite(torch.tensor(f)).all()
            ):
                raise ValueError("invalid deployment feature vector")
            rows = region["actions"]
            if [r["action"] for r in rows] != ["STOP", "STEREO", "GAUSSIAN"] or any(
                type(r["selector_label_available"]) is not bool for r in rows
            ):
                raise ValueError("invalid real action identity/availability")
            if (
                rows[0]["selector_label_available"] is not True
                or rows[0]["metric_changes"] is not None
            ):
                raise ValueError(
                    "STOP must be observed with zero gain/risk and no metric change vector"
                )
            triples = [(0.0, 0.0, False)]
            known = True
            for ai, row in enumerate(rows):
                if not row["selector_label_available"]:
                    if any(
                        row[k] is not None
                        for k in ("global_gain", "local_gain", "harm", "metric_changes")
                    ):
                        raise ValueError("unknown metric outcomes must carry no labels")
                    if row.get("risk_local") is not None or row.get("risk_global") is not None:
                        raise ValueError("unknown outcomes cannot carry local/global risk labels")
                    known = False
                    continue
                if ai:
                    vector = row["metric_changes"]
                    if (
                        not isinstance(vector, list)
                        or len(vector) != 12
                        or not bool(torch.isfinite(torch.tensor(vector)).all())
                    ):
                        raise ValueError("invalid physical metric changes")
                    triples.append(metric_targets(vector, cfg, ai))
                expected = triples[-1]
                if (
                    row["global_gain"] != expected[0]
                    or row["local_gain"] != expected[1]
                    or type(row["harm"]) is not bool
                    or row["harm"] != expected[2]
                ):
                    raise ValueError("frozen labels differ from canonical utility/guards")
                events = (
                    (False, False, False)
                    if ai == 0
                    else metric_risk_targets(row["metric_changes"], cfg, ai)
                )
                for key, value in zip(("risk_local", "risk_global"), events[:2], strict=True):
                    if key in row and (type(row[key]) is not bool or row[key] != value):
                        raise ValueError(
                            "recorded event risk differs from canonical physical guards"
                        )
            if not known:
                continue
            features.append(f)
            gains.append([t[0] for t in triples])
            locals_.append([t[1] for t in triples])
            harms.append([float(t[2]) for t in triples])
            known_ids.append(record["sample_id"])
            components.append(
                [[0.0, 0.0]]
                + [list(metric_risk_targets(rows[a]["metric_changes"], cfg, a)[:2]) for a in (1, 2)]
            )
    if not features and not allow_empty:
        raise ValueError("frozen table has no fully observed regions")
    return SelectorOutcomes(
        meta,
        records,
        torch.tensor(features, dtype=torch.float32).reshape(-1, len(selector_feature_names(cfg))),
        torch.tensor(gains, dtype=torch.float32).reshape(-1, 3),
        torch.tensor(locals_, dtype=torch.float32).reshape(-1, 3),
        torch.tensor(harms, dtype=torch.float32).reshape(-1, 3),
        tuple(known_ids),
        torch.tensor(components, dtype=torch.float32).reshape(-1, 3, 2),
    )
