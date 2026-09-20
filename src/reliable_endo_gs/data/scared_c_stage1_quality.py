"""Traceable exclusions for unusable Stage1 GT, without changing the source split.

The audit reads cached float32 disparity and bool masks of shape ``[H, W]``.
It does not infer that an unusable target is a corrupt source frame: source
verification is separate. Unreadable or structurally corrupt cache entries
are errors and must be repaired, rather than silently excluded.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np

from reliable_endo_gs.data.scared_c_stage1_cache import Stage1GTCache
from reliable_endo_gs.data.scared_c_stage1_split import (
    Stage1FrameSplit,
    load_stage1_frame_split,
)


class Stage1GTQualityError(ValueError):
    """Raised when an audit or its provenance cannot safely be used."""


def _hash_payload(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Stage1SampleQuality:
    """Counts over one cached ``[H, W]`` target, retaining its original partition."""

    sample_id: str
    partition: Literal["train", "validation"]
    total_pixel_count: int
    valid_pixel_count: int
    invalid_supervised_disparity_count: int

    @property
    def exclusion_reason(self) -> str | None:
        if self.valid_pixel_count == 0:
            return "zero_valid_mask"
        if self.invalid_supervised_disparity_count:
            return "invalid_supervised_disparity"
        return None

    def to_payload(self) -> dict[str, object]:
        """Return JSON-compatible pixel support and the deterministic decision."""

        return {
            "sample_id": self.sample_id,
            "partition": self.partition,
            "total_pixel_count": self.total_pixel_count,
            "valid_pixel_count": self.valid_pixel_count,
            "invalid_supervised_disparity_count": self.invalid_supervised_disparity_count,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True, slots=True)
class Stage1GTQualityManifest:
    """Immutable audit results tied to the original split and cache manifest."""

    input_split_manifest_sha256: str
    cache_manifest_sha256: str
    samples: tuple[Stage1SampleQuality, ...]

    @property
    def train_sample_ids(self) -> tuple[str, ...]:
        return tuple(
            sample.sample_id
            for sample in self.samples
            if sample.partition == "train" and sample.exclusion_reason is None
        )

    @property
    def validation_sample_ids(self) -> tuple[str, ...]:
        return tuple(
            sample.sample_id
            for sample in self.samples
            if sample.partition == "validation" and sample.exclusion_reason is None
        )

    @property
    def excluded(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                sample.sample_id: reason
                for sample in self.samples
                if (reason := sample.exclusion_reason) is not None
            }
        )

    @property
    def coverage_by_keyframe(self) -> Mapping[str, Mapping[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for sample in self.samples:
            keyframe = "/".join(sample.sample_id.split("/")[1:3])
            entry = counts.setdefault(
                keyframe,
                {
                    f"{partition}_{kind}": 0
                    for partition in ("train", "validation")
                    for kind in ("source", "retained", "excluded")
                },
            )
            entry[f"{sample.partition}_source"] += 1
            kind = "retained" if sample.exclusion_reason is None else "excluded"
            entry[f"{sample.partition}_{kind}"] += 1
        return MappingProxyType({key: MappingProxyType(value) for key, value in counts.items()})

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "manifest_kind": "stage1_scared_c_gt_quality",
            "input_split_manifest_sha256": self.input_split_manifest_sha256,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "policy": "exclude_zero_valid_or_invalid_supervised_disparity_v1",
            "samples": [sample.to_payload() for sample in self.samples],
            "train_sample_ids": list(self.train_sample_ids),
            "validation_sample_ids": list(self.validation_sample_ids),
            "excluded": dict(self.excluded),
            "coverage_by_keyframe": {
                key: dict(value) for key, value in self.coverage_by_keyframe.items()
            },
        }

    @property
    def manifest_sha256(self) -> str:
        return _hash_payload(self._payload_without_hash())

    def to_payload(self) -> dict[str, object]:
        """Return the canonical audit content, including its SHA-256 identity."""

        payload = self._payload_without_hash()
        payload["manifest_sha256"] = _hash_payload(payload)
        return payload


def _open_cache(
    cache_root: Path, split: Stage1FrameSplit, split_manifest_path: Path
) -> tuple[Stage1GTCache, str]:
    persisted_split = load_stage1_frame_split(split_manifest_path)
    if persisted_split.to_payload() != split.to_payload():
        raise Stage1GTQualityError("audit split differs from the persisted source split")
    cache_hash = hashlib.sha256((cache_root / "manifest.json").read_bytes()).hexdigest()
    cache = Stage1GTCache(
        cache_root,
        expected_sample_ids=split.source_sample_ids,
        expected_split_manifest_sha256=split.manifest_sha256,
        expected_split_manifest_path=split_manifest_path,
    )
    return cache, cache_hash


def _validate_manifest(manifest: Stage1GTQualityManifest, split: Stage1FrameSplit) -> None:
    if manifest.input_split_manifest_sha256 != split.manifest_sha256:
        raise Stage1GTQualityError("quality manifest belongs to a different source split")
    if tuple(sample.sample_id for sample in manifest.samples) != split.source_sample_ids:
        raise Stage1GTQualityError("quality manifest must cover each source sample exactly once")
    train_ids = set(split.train_sample_ids)
    for sample in manifest.samples:
        expected_partition = "train" if sample.sample_id in train_ids else "validation"
        if sample.partition != expected_partition:
            raise Stage1GTQualityError(f"quality manifest reassigns partition: {sample.sample_id}")
        counts = (
            sample.total_pixel_count,
            sample.valid_pixel_count,
            sample.invalid_supervised_disparity_count,
        )
        if (
            any(type(count) is not int for count in counts)
            or not 0 <= counts[2] <= counts[1] <= counts[0]
            or counts[0] <= 0
        ):
            raise Stage1GTQualityError(f"invalid pixel support counts: {sample.sample_id}")
    for partition, retained in (
        ("train", manifest.train_sample_ids),
        ("validation", manifest.validation_sample_ids),
    ):
        if not retained:
            raise Stage1GTQualityError(f"quality audit leaves no usable {partition} samples")


def audit_stage1_gt_cache(
    cache_root: Path,
    split: Stage1FrameSplit,
    *,
    split_manifest_path: Path,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> Stage1GTQualityManifest:
    """Scan all ``[H, W]`` GT entries and exclude only unusable supervision.

    Valid disparity must be finite and positive wherever the original mask
    is true. Values outside that mask do not affect the exclusion decision.
    No pixels, original partitions, or cache files are modified. Cache read,
    dtype, shape, and calibration errors propagate instead of being excluded.
    ``progress(completed, total)`` runs in the calling thread after each result,
    including when multiple IO workers are used.
    """

    if type(workers) is not int or workers < 1:
        raise Stage1GTQualityError("audit workers must be a positive integer")
    cache, cache_hash = _open_cache(cache_root, split, split_manifest_path)
    train_ids = set(split.train_sample_ids)

    def inspect(sample_id: str) -> Stage1SampleQuality:
        disparity, mask, _, _ = cache.load(sample_id)
        supervised = disparity[mask]
        invalid_count = int(np.count_nonzero(~np.isfinite(supervised) | (supervised <= 0)))
        return Stage1SampleQuality(
            sample_id=sample_id,
            partition="train" if sample_id in train_ids else "validation",
            total_pixel_count=int(mask.size),
            valid_pixel_count=int(supervised.size),
            invalid_supervised_disparity_count=invalid_count,
        )

    def collect(results: Iterable[Stage1SampleQuality]) -> tuple[Stage1SampleQuality, ...]:
        collected: list[Stage1SampleQuality] = []
        total = len(split.source_sample_ids)
        for completed, sample in enumerate(results, start=1):
            collected.append(sample)
            if progress is not None:
                progress(completed, total)
        return tuple(collected)

    if workers == 1:
        samples = collect(inspect(sample_id) for sample_id in split.source_sample_ids)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            samples = collect(executor.map(inspect, split.source_sample_ids))
    if hashlib.sha256((cache_root / "manifest.json").read_bytes()).hexdigest() != cache_hash:
        raise Stage1GTQualityError("cache manifest changed during quality audit")
    result = Stage1GTQualityManifest(split.manifest_sha256, cache_hash, samples)
    _validate_manifest(result, split)
    return result


def write_stage1_gt_quality_manifest(path: Path, manifest: Stage1GTQualityManifest) -> Path:
    """Write an audit once, accepting an identical existing manifest only."""

    payload = manifest.to_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Stage1GTQualityError(f"cannot read existing quality manifest: {path}") from error
        if existing != payload:
            raise Stage1GTQualityError(
                f"refusing to overwrite incompatible quality manifest: {path}"
            ) from None
    return path


def load_stage1_gt_quality_manifest(
    path: Path,
    *,
    split: Stage1FrameSplit,
    cache_root: Path,
    split_manifest_path: Path,
) -> Stage1GTQualityManifest:
    """Validate a persisted audit, its split, and its cache-manifest byte hash.

    This does not reread NPZ contents. Call :func:`audit_stage1_gt_cache` to
    verify current target arrays before a new run.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage1GTQualityError(f"unable to read quality manifest: {path}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise Stage1GTQualityError("quality manifest must be an object with sample records")
    samples: list[Stage1SampleQuality] = []
    for raw in payload["samples"]:
        if not isinstance(raw, dict):
            raise Stage1GTQualityError("quality manifest contains a non-object sample")
        try:
            sample = Stage1SampleQuality(
                sample_id=raw["sample_id"],
                partition=raw["partition"],
                total_pixel_count=raw["total_pixel_count"],
                valid_pixel_count=raw["valid_pixel_count"],
                invalid_supervised_disparity_count=raw["invalid_supervised_disparity_count"],
            )
        except KeyError as error:
            raise Stage1GTQualityError(
                "quality manifest has an incomplete sample record"
            ) from error
        samples.append(sample)
    for name in ("input_split_manifest_sha256", "cache_manifest_sha256"):
        value = payload.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise Stage1GTQualityError(f"quality manifest has invalid {name}")
    result = Stage1GTQualityManifest(
        payload["input_split_manifest_sha256"], payload["cache_manifest_sha256"], tuple(samples)
    )
    _validate_manifest(result, split)
    if result.to_payload() != payload:
        raise Stage1GTQualityError(
            "quality manifest hash, policy, decisions, or coverage are invalid"
        )
    _, cache_hash = _open_cache(cache_root, split, split_manifest_path)
    if result.cache_manifest_sha256 != cache_hash:
        raise Stage1GTQualityError("quality manifest belongs to a different cache manifest")
    return result
