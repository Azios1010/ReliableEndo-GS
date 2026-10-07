"""Read-only cache screening preflight; original target identities are retained."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from reliable_endo_gs.config.intervention import load_intervention_config
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, load_a01_manifest
from reliable_endo_gs.runtime.identity import source_sha256
from reliable_endo_gs.teachers import TargetCache, TargetCacheKey, load_a01_roles
from reliable_endo_gs.teachers.quality import screen_target
from reliable_endo_gs.teachers.workflow import sample_cache_key
from reliable_endo_gs.utils.io import write_json_atomic


def audit_target_quality(
    *,
    data_root: Path,
    manifest_path: Path,
    roles_path: Path,
    cache_root: Path,
    config_path: Path,
    output_dir: Path,
    sample_ids: Sequence[str] = (),
    max_samples: int = 64,
) -> dict[str, object]:
    """CPU GT-free audit; neither cache misses nor screening trigger inference."""
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    repo = Path(__file__).resolve().parents[3]
    if repo == output_dir.resolve() or repo in output_dir.resolve().parents:
        raise ValueError("quality audit artifacts must be outside the repository")
    config = load_intervention_config(config_path)
    roles = load_a01_roles(roles_path)
    manifest = load_a01_manifest(manifest_path, roles=roles)
    adapter = ScaredA00Adapter(data_root, manifest, roles=roles)
    cache = TargetCache(cache_root, roles=roles, read_only=True)
    keys: dict[str, TargetCacheKey] = {}
    for entry in cache.entries:
        entry_key = TargetCacheKey.from_dict(cast(Mapping[str, object], entry["key"]))
        if entry_key.sample_id in keys:
            raise ValueError("ambiguous teacher identities for one sample; choose a pinned cache")
        keys[entry_key.sample_id] = entry_key
    if sample_ids:
        ids = list(sample_ids)
    else:
        # Deterministic pilot across folds, without creating/changing a split.
        groups = [
            [sid for sid in sorted(keys) if sid.startswith(f"dataset_{i}/")] for i in range(1, 8)
        ]
        ids = []
        for position in range(max_samples):
            group = groups[position % 7]
            if not group:
                continue
            take = (max_samples + 6 - position % 7) // 7
            index = min(len(group) - 1, (position // 7) * len(group) // max(1, take))
            if group[index] not in ids:
                ids.append(group[index])
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("quality audit needs unique nonempty sample IDs")
    rows = []
    for sid in ids:
        if sid.split("/")[0] not in {f"dataset_{i}" for i in range(1, 8)}:
            raise ValueError("target-quality pilot cannot consume final_test")
        key = keys.get(sid)
        if key is None:
            raise ValueError(f"missing cached sample: {sid}")
        target = cache.get(key)
        if target is None:
            raise ValueError(f"cache identity/checksum failure: {sid}")
        sample = adapter.load(
            sid,
            data_role="final_train",
            include_supervision=False,
            output_hw=(target.disparity.shape[-2], target.disparity.shape[-1]),
        )
        if sample.supervision is not None:
            raise ValueError("GT leakage in quality audit")
        actual_key = sample_cache_key(
            sample,
            teacher_identity_hash=key.teacher_identity_hash,
            split_identity=key.split_identity,
            roles=roles,
        )
        if actual_key != key:
            raise ValueError(f"cached target does not match actual image/calibration: {sid}")
        quality = screen_target(sample.batch, target, config)
        rows.append(
            {
                "sample_id": sid,
                "cache_key": key.digest,
                "declared_pixels": quality.declared_count,
                "accepted_pixels": quality.accepted_count,
                "outside_right_pixels": quality.outside_right_count,
                "below_near_pixels": quality.below_near_count,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    result: dict[str, object] = {
        "status": "target_quality_audit_complete",
        "samples": len(rows),
        "source_sha256": source_sha256(repo),
        "manifest_hash": manifest.manifest_hash,
        "role_hash": roles.role_hash,
        "config": config.to_dict(),
        "config_hash": config.config_hash,
        "teacher_inference_executed": False,
        "GT_loaded": False,
        "cache_modified": False,
        "empty_KD_frames": sum(row["accepted_pixels"] == 0 for row in rows),
        "rows": rows,
        "limitations": [
            "FOV/render-support screening does not prove every accepted disparity correct.",
            "No photometric threshold or calibrated confidence is inferred from GT.",
        ],
    }
    write_json_atomic(output_dir / "result.json", result)
    return result
