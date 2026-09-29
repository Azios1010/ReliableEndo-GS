"""Executable A01 teacher/cache workflow over the first-party A00 adapter."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import torch

from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera
from reliable_endo_gs.data.scared_a00 import ScaredA00Adapter, ScaredA00Sample
from reliable_endo_gs.teachers.base import (
    Teacher,
    TeacherContractError,
    TeacherPrediction,
    TeacherUnavailableError,
)
from reliable_endo_gs.teachers.cache import (
    DEFAULT_TARGET_SCHEMA,
    TargetCache,
    TargetCacheKey,
    TargetCacheResult,
)
from reliable_endo_gs.teachers.roles import DEFAULT_A01_SCARED_ROLES, A01ScaredDataRoles
from reliable_endo_gs.teachers.targets import (
    AlignedTeacherTarget,
    RectifiedAlignment,
    align_teacher_prediction,
)


class A01WorkflowError(RuntimeError):
    """Raised when an A01 command cannot complete safely."""


@dataclass(frozen=True, slots=True)
class A01TargetResult:
    sample_id: str
    data_role: str
    target: AlignedTeacherTarget
    cache_hit: bool
    cache_key: TargetCacheKey


@dataclass(frozen=True, slots=True)
class A01CacheReport:
    processed: int
    cache_hits: int
    computed: int
    failed: tuple[str, ...]
    teacher_identity_hash: str
    cache_root: Path


def _hash_mapping(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _teacher_identity_hash(teacher: Teacher) -> str:
    identity = getattr(teacher, "identity", None)
    identity_hash = getattr(identity, "identity_hash", None)
    if not isinstance(identity_hash, str) or len(identity_hash) != 64:
        raise A01WorkflowError("teacher adapter must expose a hashed TeacherIdentity")
    return identity_hash


def _alignment(
    sample: ScaredA00Sample, *, teacher_hw: tuple[int, int] | None = None
) -> RectifiedAlignment:
    height, width = sample.batch.left.shape[-2:]
    resolved_teacher_hw = (int(height), int(width)) if teacher_hw is None else teacher_hw
    return RectifiedAlignment(
        input_hw=(int(height), int(width)),
        teacher_hw=resolved_teacher_hw,
        crop_hw=(int(height), int(width)),
        image_transform_id=f"a00:{sample.calibration_metadata.get('protocol_identity', 'unknown')}",
        calibration_transform_id=f"a00:{sample.calibration_metadata.get('protocol_identity', 'unknown')}",
    )


def target_cache_key(
    sample: ScaredA00Sample,
    teacher: Teacher,
    *,
    split_identity: str,
    roles: A01ScaredDataRoles = DEFAULT_A01_SCARED_ROLES,
) -> TargetCacheKey:
    """Build the complete cache identity for one A00 sample."""

    alignment = _alignment(sample)
    image_hash = _hash_mapping(
        {
            "alignment": alignment.to_metadata(),
            "image_range": sample.calibration_metadata.get("image_range"),
            "input_hw": sample.calibration_metadata.get("input_hw"),
            "source_file_hashes": sample.calibration_metadata.get("source_file_hashes"),
        }
    )
    calibration_hash = _hash_mapping(dict(sample.calibration_metadata))
    identity_hash = _teacher_identity_hash(teacher)
    return TargetCacheKey(
        sample_id=sample.sample_id,
        split_identity=split_identity,
        data_role=sample.data_role,
        image_transform_hash=image_hash,
        calibration_transform_hash=calibration_hash,
        teacher_identity_hash=identity_hash,
        target_schema_hash=DEFAULT_TARGET_SCHEMA.schema_hash,
        role_contract_hash=roles.role_hash,
    )


def _batch_on_device(
    batch: DeploymentStereoBatch, device: torch.device | str
) -> DeploymentStereoBatch:
    """Move an A00 deployment batch, including calibration, as one unit."""

    resolved = torch.device(device)
    camera = batch.camera
    moved_camera = RectifiedStereoCamera(
        camera.K_L.to(device=resolved),
        camera.K_R.to(device=resolved),
        camera.T_R_from_L.to(device=resolved),
        True,
    )
    return DeploymentStereoBatch(
        batch.left.to(device=resolved),
        batch.right.to(device=resolved),
        moved_camera,
        batch.support.to(device=resolved),
    )


def infer_and_align(
    sample: ScaredA00Sample,
    teacher: Teacher,
    *,
    device: torch.device | str | None = None,
) -> AlignedTeacherTarget:
    """Run one frozen teacher and convert output to the A00 target grid.

    ``device`` is explicit because SCARED decoding is CPU-side while real
    FoundationStereo/DEFOM inference is normally GPU-side.
    """

    if not callable(getattr(teacher, "infer", None)):
        raise A01WorkflowError("teacher must expose infer(left, right)")
    batch = sample.batch if device is None else _batch_on_device(sample.batch, device)
    with torch.inference_mode():
        prediction = teacher.infer(batch.left, batch.right)
    if not isinstance(prediction, TeacherPrediction):
        raise A01WorkflowError(
            "teacher inference must return TeacherPrediction after output conversion"
        )
    teacher_hw = (
        int(prediction.disparity.shape[-2]),
        int(prediction.disparity.shape[-1]),
    )
    return align_teacher_prediction(
        prediction,
        _alignment(sample, teacher_hw=teacher_hw),
        input_support=batch.support,
        calibration_metadata=sample.calibration_metadata,
    )


def collect_one_target(
    sample: ScaredA00Sample,
    teacher: Teacher,
    cache: TargetCache,
    *,
    split_identity: str,
    roles: A01ScaredDataRoles = DEFAULT_A01_SCARED_ROLES,
    device: torch.device | str | None = None,
) -> A01TargetResult:
    """Resume one cache item or compute it exactly once."""

    key = target_cache_key(sample, teacher, split_identity=split_identity, roles=roles)

    def produce() -> AlignedTeacherTarget:
        target = infer_and_align(sample, teacher, device=device)
        return replace(
            target,
            metadata={
                **dict(target.metadata),
                "target_cache_key": key.to_dict(),
                "target_cache_digest": key.digest,
            },
        )

    result: TargetCacheResult = cache.get_or_compute(key, produce)
    return A01TargetResult(sample.sample_id, sample.data_role, result.target, result.cache_hit, key)


def collect_targets(
    adapter: ScaredA00Adapter,
    teacher: Teacher,
    cache: TargetCache,
    *,
    split_identity: str,
    roles: A01ScaredDataRoles = DEFAULT_A01_SCARED_ROLES,
    stop_on_error: bool = True,
    device: torch.device | str | None = None,
    output_hw: tuple[int, int] | None = None,
) -> A01CacheReport:
    """Collect/resume all manifest samples with provenance-bound keys."""

    if cache.roles.role_hash != roles.role_hash:
        raise A01WorkflowError("target cache role contract does not match active A01 roles")
    identity_hash = _teacher_identity_hash(teacher)
    processed = cache_hits = computed = 0
    failures: list[str] = []
    for sample in adapter.iter_samples(include_supervision=False, output_hw=output_hw):
        try:
            result = collect_one_target(
                sample,
                teacher,
                cache,
                split_identity=split_identity,
                roles=roles,
                device=device,
            )
            processed += 1
            if result.cache_hit:
                cache_hits += 1
            else:
                computed += 1
        except (
            A01WorkflowError,
            TeacherContractError,
            TeacherUnavailableError,
            ValueError,
            OSError,
        ) as error:
            failures.append(f"{sample.sample_id}: {type(error).__name__}: {error}")
            if stop_on_error:
                raise A01WorkflowError(failures[-1]) from error
    return A01CacheReport(
        processed, cache_hits, computed, tuple(failures), identity_hash, cache.root
    )


def validate_cache(
    cache: TargetCache,
    *,
    expected_sample_ids: Sequence[str] | None = None,
    expected_key_digests: Sequence[str] | None = None,
) -> dict[str, object]:
    """Read cache items and optionally require explicit manifest coverage."""

    valid = 0
    stale: list[str] = []
    valid_sample_ids: set[str] = set()
    valid_digests: set[str] = set()
    for entry in cache.entries:
        raw_key = entry.get("key")
        if not isinstance(raw_key, Mapping):
            stale.append(str(entry.get("digest", "unknown")))
            continue
        key = TargetCacheKey.from_dict(raw_key)
        if cache.get(key) is None:
            stale.append(key.sample_id)
        else:
            valid += 1
            valid_sample_ids.add(key.sample_id)
            valid_digests.add(key.digest)
    expected = None if expected_sample_ids is None else set(expected_sample_ids)
    missing_expected = [] if expected is None else sorted(expected - valid_sample_ids)
    expected_digests = None if expected_key_digests is None else set(expected_key_digests)
    missing_expected_digests = (
        [] if expected_digests is None else sorted(expected_digests - valid_digests)
    )
    return {
        "cache_root": cache.root.as_posix(),
        "entries": len(cache.entries),
        "valid": valid,
        "stale_or_missing": stale,
        "expected_count": None if expected is None else len(expected),
        "missing_expected": missing_expected,
        "expected_key_count": None if expected_digests is None else len(expected_digests),
        "missing_expected_keys": missing_expected_digests,
        "role_contract_hash": cache.roles.role_hash,
        "target_schema_hash": cache.target_schema.schema_hash,
    }


def load_teacher_factory(spec: str, *, kwargs: Mapping[str, object] | None = None) -> Teacher:
    """Load an explicitly injected ``module:function`` teacher factory.

    Importing this helper does not import either external teacher repository;
    the import occurs only after a command explicitly requests the factory.
    """

    if ":" not in spec:
        raise A01WorkflowError("teacher factory must use module:function syntax")
    module_name, function_name = spec.split(":", maxsplit=1)
    if not module_name or not function_name:
        raise A01WorkflowError("teacher factory must use module:function syntax")
    try:
        factory = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as error:
        raise A01WorkflowError(
            f"unable to import injected teacher factory {spec!r}: {error}"
        ) from error
    if not callable(factory):
        raise A01WorkflowError(f"injected teacher factory {spec!r} is not callable")
    teacher = cast(Teacher, factory(**dict(kwargs or {})))
    if not callable(getattr(teacher, "infer", None)) or not hasattr(teacher, "identity"):
        raise A01WorkflowError("injected teacher factory must return the A01 Teacher protocol")
    return teacher


__all__ = [
    "A01CacheReport",
    "A01TargetResult",
    "A01WorkflowError",
    "collect_one_target",
    "collect_targets",
    "infer_and_align",
    "load_teacher_factory",
    "target_cache_key",
    "validate_cache",
]
