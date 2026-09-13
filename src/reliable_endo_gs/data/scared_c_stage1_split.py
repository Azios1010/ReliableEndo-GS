"""Deterministic frame-level selection and splitting for Stage1 SCARED-C.

The split is deliberately independent of the role manifest.  The role
manifest protects dataset-level boundaries, while this module records the
contiguous temporal train/validation partition used when every corrected
video keyframe in datasets 1--3 is eligible for Stage1 training.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from reliable_endo_gs.data.scared_c import (
    CORRECTED_VIDEO_MODE,
    SCARED_C_UPSTREAM_REVISION,
    ScaredCSampleRecord,
)

STAGE1_FULL_SPLIT_SCHEMA = 1
STAGE1_FULL_SPLIT_NAME = "stage1_scared_c_full_v1"
STAGE1_TRAIN_DATASET_IDS = ("dataset_1", "dataset_2", "dataset_3")
STAGE1_FORBIDDEN_DATASET_IDS = ("dataset_6", "dataset_7")

_DATASET_ID_RE = re.compile(r"^dataset_([1-9][0-9]*)$")
_KEYFRAME_ENTRY_RE = re.compile(r"^(dataset_[1-9][0-9]*)/(keyframe_[1-9][0-9]*)$")
_SAMPLE_ID_RE = re.compile(r"^scared_c/(dataset_[1-9][0-9]*)/(keyframe_[1-9][0-9]*)/([1-9][0-9]*)$")


class Stage1FrameSplitError(ValueError):
    """Raised when a Stage1 frame selection or split is unsafe."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash_payload(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def normalize_dataset_ids(values: Sequence[str], *, name: str = "dataset_ids") -> tuple[str, ...]:
    """Validate and deterministically normalize explicit dataset IDs."""

    if not values or not all(isinstance(value, str) for value in values):
        raise Stage1FrameSplitError(f"{name} must be a non-empty sequence of strings")
    normalized = tuple(value.strip() for value in values)
    if any(_DATASET_ID_RE.fullmatch(value) is None for value in normalized):
        raise Stage1FrameSplitError(f"{name} contains an invalid dataset ID")
    if len(set(normalized)) != len(normalized):
        raise Stage1FrameSplitError(f"{name} contains duplicate dataset IDs")
    return tuple(sorted(normalized, key=lambda value: (int(value.removeprefix("dataset_")), value)))


def normalize_keyframe_entries(
    values: Sequence[str], *, name: str = "keyframe_entries"
) -> tuple[str, ...]:
    """Validate explicit ``dataset_N/keyframe_M`` selectors."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise Stage1FrameSplitError(f"{name} must be a sequence of strings")
    if not values or not all(isinstance(value, str) for value in values):
        raise Stage1FrameSplitError(f"{name} must be a non-empty sequence of strings")
    normalized = tuple(value.strip().replace("\\", "/") for value in values)
    if any(_KEYFRAME_ENTRY_RE.fullmatch(value) is None for value in normalized):
        raise Stage1FrameSplitError(f"{name} must contain only dataset_N/keyframe_M selectors")
    if len(set(normalized)) != len(normalized):
        raise Stage1FrameSplitError(f"{name} contains duplicate keyframe selectors")
    return tuple(
        sorted(
            normalized,
            key=lambda value: tuple(int(part) for part in re.findall(r"[1-9][0-9]*", value)),
        )
    )


def normalize_sample_ids(values: Sequence[str], *, name: str = "sample_ids") -> tuple[str, ...]:
    """Validate explicit corrected-video sample IDs."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise Stage1FrameSplitError(f"{name} must be a sequence of strings")
    if not values or not all(isinstance(value, str) for value in values):
        raise Stage1FrameSplitError(f"{name} must be a non-empty sequence of strings")
    normalized = tuple(value.strip().replace("\\", "/") for value in values)
    if any(_SAMPLE_ID_RE.fullmatch(value) is None for value in normalized):
        raise Stage1FrameSplitError(
            f"{name} must contain corrected-video IDs of the form "
            "scared_c/dataset_N/keyframe_M/frame"
        )
    if len(set(normalized)) != len(normalized):
        raise Stage1FrameSplitError(f"{name} contains duplicate sample IDs")
    return normalized


def _selector_count(
    dataset_ids: Sequence[str] | None,
    keyframe_entries: Sequence[str] | None,
    sample_ids: Sequence[str] | None,
) -> int:
    return sum(
        value is not None and len(value) > 0
        for value in (dataset_ids, keyframe_entries, sample_ids)
    )


def select_stage1_records(
    records: Sequence[ScaredCSampleRecord],
    *,
    dataset_ids: Sequence[str] | None = None,
    keyframe_entries: Sequence[str] | None = None,
    sample_ids: Sequence[str] | None = None,
    allowed_dataset_ids: Sequence[str] = STAGE1_TRAIN_DATASET_IDS,
) -> tuple[ScaredCSampleRecord, ...]:
    """Select records using exactly one explicit selector kind.

    ``dataset_ids`` selects every indexed corrected-video frame in those
    datasets; ``keyframe_entries`` selects complete keyframes; and
    ``sample_ids`` selects exact frames.  The mutually-exclusive API prevents
    an accidental union of broad and narrow selectors.
    """

    if _selector_count(dataset_ids, keyframe_entries, sample_ids) != 1:
        raise Stage1FrameSplitError(
            "exactly one non-empty selector is required: dataset_ids, "
            "keyframe_entries, or sample_ids"
        )
    allowed = set(normalize_dataset_ids(tuple(allowed_dataset_ids), name="allowed_dataset_ids"))
    normalized_datasets = (
        set(normalize_dataset_ids(tuple(dataset_ids))) if dataset_ids is not None else None
    )
    normalized_keyframes = (
        set(normalize_keyframe_entries(tuple(keyframe_entries)))
        if keyframe_entries is not None
        else None
    )
    normalized_samples = (
        set(normalize_sample_ids(tuple(sample_ids))) if sample_ids is not None else None
    )
    if normalized_datasets is not None and not normalized_datasets <= allowed:
        raise Stage1FrameSplitError(
            f"dataset selection is outside the allowlist: {sorted(normalized_datasets - allowed)}"
        )

    selected: list[ScaredCSampleRecord] = []
    seen: set[str] = set()
    for record in records:
        if record.mode != CORRECTED_VIDEO_MODE:
            continue
        record_dataset = record.sequence_id
        record_keyframe = f"{record.sequence_id}/{record.keyframe_id}"
        matches = (
            (normalized_datasets is not None and record_dataset in normalized_datasets)
            or (normalized_keyframes is not None and record_keyframe in normalized_keyframes)
            or (normalized_samples is not None and record.sample_id in normalized_samples)
        )
        if not matches:
            continue
        if record_dataset not in allowed:
            raise Stage1FrameSplitError(
                f"selected Stage1 record is outside the dataset allowlist: {record.sample_id}"
            )
        if record.final_role or record_dataset in STAGE1_FORBIDDEN_DATASET_IDS:
            raise Stage1FrameSplitError(f"forbidden Stage1 record selected: {record.sample_id}")
        if record.sample_id in seen:
            raise Stage1FrameSplitError(f"duplicate selected sample ID: {record.sample_id}")
        seen.add(record.sample_id)
        selected.append(record)

    requested_samples = normalized_samples or set()
    missing_samples = requested_samples - seen
    if missing_samples:
        raise Stage1FrameSplitError(
            f"requested sample IDs are absent from the index: {sorted(missing_samples)[:3]}"
        )
    if not selected:
        raise Stage1FrameSplitError("explicit Stage1 selection produced no records")
    return tuple(
        sorted(
            selected,
            key=lambda record: (
                int(record.sequence_id.removeprefix("dataset_")),
                int(record.keyframe_id.removeprefix("keyframe_")),
                int(record.frame_id),
            ),
        )
    )


def _record_fingerprint(records: Sequence[ScaredCSampleRecord]) -> str:
    identity = [
        {
            "sample_id": record.sample_id,
            "dataset_id": record.dataset_id,
            "sequence_id": record.sequence_id,
            "keyframe_id": record.keyframe_id,
            "frame_id": int(record.frame_id),
            "source_type": record.source_type,
            "video_frame_count": record.video_frame_count,
        }
        for record in records
    ]
    return _hash_payload({"records": identity})


@dataclass(frozen=True, slots=True)
class Stage1FrameSplit:
    """Immutable frame split plus all provenance needed to replay it."""

    split_name: str
    dataset_revision: str
    mode: str
    dataset_ids: tuple[str, ...]
    keyframe_entries: tuple[str, ...]
    source_sample_ids: tuple[str, ...]
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    source_frame_counts: Mapping[str, int]
    validation_frame_counts: Mapping[str, int]
    validation_fraction: float
    block_policy: str
    rounding: str
    min_validation_frames_per_keyframe: int
    transfer_keyframe_entries: tuple[str, ...]
    excluded_dataset_ids: tuple[str, ...]
    record_fingerprint: str
    provenance: Mapping[str, object]
    manifest_sha256: str = ""

    @property
    def train_count(self) -> int:
        return len(self.train_sample_ids)

    @property
    def validation_count(self) -> int:
        return len(self.validation_sample_ids)

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": STAGE1_FULL_SPLIT_SCHEMA,
            "manifest_kind": "stage1_scared_c_frame_split",
            "dataset": "scared_c",
            "dataset_revision": self.dataset_revision,
            "split_name": self.split_name,
            "mode": self.mode,
            "selection": {
                "dataset_ids": list(self.dataset_ids),
                "keyframe_entries": list(self.keyframe_entries),
                "sample_ids": list(self.source_sample_ids),
            },
            "source_frame_counts": dict(self.source_frame_counts),
            "validation_frame_counts": dict(self.validation_frame_counts),
            "train_sample_ids": list(self.train_sample_ids),
            "validation_sample_ids": list(self.validation_sample_ids),
            "validation_policy": {
                "fraction": self.validation_fraction,
                "block_policy": self.block_policy,
                "rounding": self.rounding,
                "min_frames_per_keyframe": self.min_validation_frames_per_keyframe,
            },
            "dataset_allowlist": list(self.dataset_ids),
            "excluded_dataset_ids": list(self.excluded_dataset_ids),
            "transfer_keyframe_entries": list(self.transfer_keyframe_entries),
            "record_fingerprint": self.record_fingerprint,
            "provenance": dict(self.provenance),
        }

    def to_payload(self) -> dict[str, object]:
        payload = self._payload_without_hash()
        payload["manifest_sha256"] = self.manifest_sha256 or _hash_payload(payload)
        return payload


def make_stage1_frame_split(
    records: Sequence[ScaredCSampleRecord],
    *,
    dataset_ids: Sequence[str],
    validation_fraction: float = 0.05,
    block_policy: str = "tail",
    rounding: str = "nearest_half_up",
    min_validation_frames_per_keyframe: int = 1,
    transfer_keyframe_entries: Sequence[str] = (),
    excluded_dataset_ids: Sequence[str] = STAGE1_FORBIDDEN_DATASET_IDS,
    provenance: Mapping[str, object] | None = None,
) -> Stage1FrameSplit:
    """Make a deterministic contiguous temporal-block split.

    The final ``round(fraction * N)`` frames of each keyframe are reserved for
    validation.  Rounding is explicitly half-up and never randomizes frame
    order.  The split is made independently per keyframe so no temporal block
    crosses a keyframe boundary.
    """

    normalized_datasets = normalize_dataset_ids(tuple(dataset_ids))
    if tuple(normalized_datasets) != tuple(STAGE1_TRAIN_DATASET_IDS):
        raise Stage1FrameSplitError(
            "the full Stage1 config must allow exactly dataset_1, dataset_2, dataset_3"
        )
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise Stage1FrameSplitError("validation_fraction must be between zero and one")
    if block_policy != "tail":
        raise Stage1FrameSplitError(
            "only the deterministic tail temporal-block policy is supported"
        )
    if rounding != "nearest_half_up":
        raise Stage1FrameSplitError("rounding must be nearest_half_up")
    if (
        isinstance(min_validation_frames_per_keyframe, bool)
        or min_validation_frames_per_keyframe < 1
    ):
        raise Stage1FrameSplitError("min_validation_frames_per_keyframe must be positive")
    normalized_transfer = (
        normalize_keyframe_entries(
            tuple(transfer_keyframe_entries), name="transfer_keyframe_entries"
        )
        if transfer_keyframe_entries
        else ()
    )
    if any(not entry.startswith("dataset_7/") for entry in normalized_transfer):
        raise Stage1FrameSplitError("transfer keyframes must be dataset_7-only")
    normalized_excluded = normalize_dataset_ids(
        tuple(excluded_dataset_ids), name="excluded_dataset_ids"
    )
    if not set(STAGE1_FORBIDDEN_DATASET_IDS) <= set(normalized_excluded):
        raise Stage1FrameSplitError("excluded_dataset_ids must include dataset_6 and dataset_7")

    selected = select_stage1_records(
        records,
        dataset_ids=normalized_datasets,
        allowed_dataset_ids=normalized_datasets,
    )
    source_counts: dict[str, int] = defaultdict(int)
    grouped: dict[str, list[ScaredCSampleRecord]] = defaultdict(list)
    for record in selected:
        source_counts[record.sequence_key] += 1
        grouped[record.sequence_key].append(record)
    train_ids: list[str] = []
    validation_ids: list[str] = []
    validation_counts: dict[str, int] = {}
    for sequence_key in sorted(
        grouped, key=lambda value: tuple(int(part) for part in value.split("_"))
    ):
        group = sorted(grouped[sequence_key], key=lambda record: int(record.frame_id))
        if len(group) < 2:
            raise Stage1FrameSplitError(f"keyframe {sequence_key} cannot be split disjointly")
        validation_count = max(
            min_validation_frames_per_keyframe,
            int(math.floor(len(group) * validation_fraction + 0.5)),
        )
        if validation_count >= len(group):
            raise Stage1FrameSplitError(
                f"validation block consumes keyframe {sequence_key}: "
                f"{validation_count}/{len(group)}"
            )
        split_at = len(group) - validation_count
        train_ids.extend(record.sample_id for record in group[:split_at])
        validation_ids.extend(record.sample_id for record in group[split_at:])
        validation_counts[sequence_key] = validation_count

    source_ids = tuple(record.sample_id for record in selected)
    if set(train_ids) & set(validation_ids):
        raise Stage1FrameSplitError("Stage1 train and validation sample IDs overlap")
    if set(train_ids) | set(validation_ids) != set(source_ids):
        raise Stage1FrameSplitError(
            "Stage1 train/validation split does not cover the source selection"
        )
    keyframe_entries = tuple(
        dict.fromkeys(f"{record.sequence_id}/{record.keyframe_id}" for record in selected)
    )
    split = Stage1FrameSplit(
        split_name=STAGE1_FULL_SPLIT_NAME,
        dataset_revision=SCARED_C_UPSTREAM_REVISION,
        mode=CORRECTED_VIDEO_MODE,
        dataset_ids=normalized_datasets,
        keyframe_entries=keyframe_entries,
        source_sample_ids=source_ids,
        train_sample_ids=tuple(train_ids),
        validation_sample_ids=tuple(validation_ids),
        source_frame_counts=dict(sorted(source_counts.items())),
        validation_frame_counts=dict(sorted(validation_counts.items())),
        validation_fraction=validation_fraction,
        block_policy=block_policy,
        rounding=rounding,
        min_validation_frames_per_keyframe=min_validation_frames_per_keyframe,
        transfer_keyframe_entries=normalized_transfer,
        excluded_dataset_ids=normalized_excluded,
        record_fingerprint=_record_fingerprint(selected),
        provenance={} if provenance is None else dict(provenance),
    )
    return replace(split, manifest_sha256=_hash_payload(split._payload_without_hash()))


def _validate_payload(payload: Mapping[str, object]) -> Stage1FrameSplit:
    if payload.get("schema_version") != STAGE1_FULL_SPLIT_SCHEMA:
        raise Stage1FrameSplitError("unsupported Stage1 frame split schema")
    if payload.get("manifest_kind") != "stage1_scared_c_frame_split":
        raise Stage1FrameSplitError("wrong Stage1 frame split manifest kind")
    if payload.get("dataset") != "scared_c" or payload.get("mode") != CORRECTED_VIDEO_MODE:
        raise Stage1FrameSplitError("Stage1 frame split has the wrong dataset or mode")
    if payload.get("dataset_revision") != SCARED_C_UPSTREAM_REVISION:
        raise Stage1FrameSplitError("Stage1 frame split has the wrong SCARED-C revision")
    if payload.get("split_name") != STAGE1_FULL_SPLIT_NAME:
        raise Stage1FrameSplitError("Stage1 frame split has the wrong split name")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise Stage1FrameSplitError("Stage1 frame split selection must be an object")
    raw_dataset_ids = selection.get("dataset_ids")
    raw_keyframe_entries = selection.get("keyframe_entries")
    raw_source_ids = selection.get("sample_ids")
    if not isinstance(raw_dataset_ids, list) or not all(
        isinstance(v, str) for v in raw_dataset_ids
    ):
        raise Stage1FrameSplitError("Stage1 frame split selection dataset_ids is invalid")
    if not isinstance(raw_keyframe_entries, list) or not all(
        isinstance(v, str) for v in raw_keyframe_entries
    ):
        raise Stage1FrameSplitError("Stage1 frame split selection keyframe_entries is invalid")
    if not isinstance(raw_source_ids, list) or not all(isinstance(v, str) for v in raw_source_ids):
        raise Stage1FrameSplitError("Stage1 frame split source sample IDs are invalid")
    dataset_ids = normalize_dataset_ids(raw_dataset_ids)
    if tuple(dataset_ids) != tuple(STAGE1_TRAIN_DATASET_IDS):
        raise Stage1FrameSplitError("Stage1 frame split dataset allowlist is not dataset_1..3")
    keyframe_entries = normalize_keyframe_entries(raw_keyframe_entries)
    source_ids = normalize_sample_ids(raw_source_ids)
    raw_train = payload.get("train_sample_ids")
    raw_validation = payload.get("validation_sample_ids")
    if not isinstance(raw_train, list) or not all(isinstance(v, str) for v in raw_train):
        raise Stage1FrameSplitError("Stage1 train sample IDs are invalid")
    if not isinstance(raw_validation, list) or not all(isinstance(v, str) for v in raw_validation):
        raise Stage1FrameSplitError("Stage1 validation sample IDs are invalid")
    train_ids = normalize_sample_ids(raw_train, name="train_sample_ids")
    validation_ids = normalize_sample_ids(raw_validation, name="validation_sample_ids")
    all_ids = source_ids + train_ids + validation_ids
    if any(
        _SAMPLE_ID_RE.fullmatch(sample_id).group(1) not in STAGE1_TRAIN_DATASET_IDS
        for sample_id in all_ids
    ):
        raise Stage1FrameSplitError("Stage1 frame split contains a sample outside dataset_1..3")
    if set(train_ids) & set(validation_ids):
        raise Stage1FrameSplitError("Stage1 train and validation sample IDs overlap")
    if set(train_ids) | set(validation_ids) != set(source_ids):
        raise Stage1FrameSplitError("Stage1 frame split does not cover its source selection")
    policy = payload.get("validation_policy")
    if not isinstance(policy, Mapping):
        raise Stage1FrameSplitError("Stage1 validation policy is invalid")
    fraction = policy.get("fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (float, int)):
        raise Stage1FrameSplitError("validation policy fraction is invalid")
    block_policy = policy.get("block_policy")
    rounding = policy.get("rounding")
    minimum = policy.get("min_frames_per_keyframe")
    if not isinstance(block_policy, str) or not isinstance(rounding, str):
        raise Stage1FrameSplitError("validation policy names are invalid")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise Stage1FrameSplitError("validation policy minimum is invalid")
    if not 0.0 < float(fraction) < 1.0 or block_policy != "tail" or rounding != "nearest_half_up":
        raise Stage1FrameSplitError(
            "Stage1 frame split validation policy is not the approved policy"
        )
    raw_source_counts = payload.get("source_frame_counts")
    raw_validation_counts = payload.get("validation_frame_counts")
    if not isinstance(raw_source_counts, Mapping) or not isinstance(raw_validation_counts, Mapping):
        raise Stage1FrameSplitError("frame count maps are invalid")
    try:
        source_counts = {str(key): int(value) for key, value in raw_source_counts.items()}
        validation_counts = {str(key): int(value) for key, value in raw_validation_counts.items()}
    except (TypeError, ValueError) as error:
        raise Stage1FrameSplitError("frame count maps contain a non-integer value") from error
    if any(value < 1 for value in source_counts.values()) or any(
        value < 1 for value in validation_counts.values()
    ):
        raise Stage1FrameSplitError("frame count maps must contain positive values")
    expected_keyframes = {
        f"{dataset.removeprefix('dataset_')}_{keyframe.removeprefix('keyframe_')}"
        for entry in keyframe_entries
        for dataset, keyframe in (entry.split("/", maxsplit=1),)
    }
    if set(source_counts) != expected_keyframes or set(validation_counts) != expected_keyframes:
        raise Stage1FrameSplitError("frame count maps do not match the selected keyframes")
    if sum(source_counts.values()) != len(source_ids) or sum(validation_counts.values()) != len(
        validation_ids
    ):
        raise Stage1FrameSplitError("frame count maps do not match sample-ID counts")
    if any(validation_counts[key] >= source_counts[key] for key in expected_keyframes):
        raise Stage1FrameSplitError("validation consumes an entire keyframe")
    source_frames: dict[str, list[int]] = defaultdict(list)
    train_frames: dict[str, list[int]] = defaultdict(list)
    validation_frames: dict[str, list[int]] = defaultdict(list)
    for sample_id in source_ids:
        match = _SAMPLE_ID_RE.fullmatch(sample_id)
        assert match is not None
        keyframe = (
            f"{match.group(1).removeprefix('dataset_')}_{match.group(2).removeprefix('keyframe_')}"
        )
        source_frames[keyframe].append(int(match.group(3)))
    for sample_id in train_ids:
        match = _SAMPLE_ID_RE.fullmatch(sample_id)
        assert match is not None
        keyframe = (
            f"{match.group(1).removeprefix('dataset_')}_{match.group(2).removeprefix('keyframe_')}"
        )
        train_frames[keyframe].append(int(match.group(3)))
    for sample_id in validation_ids:
        match = _SAMPLE_ID_RE.fullmatch(sample_id)
        assert match is not None
        keyframe = (
            f"{match.group(1).removeprefix('dataset_')}_{match.group(2).removeprefix('keyframe_')}"
        )
        validation_frames[keyframe].append(int(match.group(3)))
    for keyframe in expected_keyframes:
        source_order = sorted(source_frames[keyframe])
        train_order = sorted(train_frames[keyframe])
        validation_order = sorted(validation_frames[keyframe])
        expected_validation_count = max(
            minimum,
            int(math.floor(source_counts[keyframe] * float(fraction) + 0.5)),
        )
        if validation_counts[keyframe] != expected_validation_count:
            raise Stage1FrameSplitError(
                f"validation count for {keyframe} does not match the declared policy"
            )
        split_at = len(source_order) - expected_validation_count
        if train_order != source_order[:split_at] or validation_order != source_order[split_at:]:
            raise Stage1FrameSplitError(
                f"validation IDs for {keyframe} are not the deterministic tail block"
            )
    actual_source_counts: dict[str, int] = defaultdict(int)
    actual_validation_counts: dict[str, int] = defaultdict(int)
    for sample_id in source_ids:
        match = _SAMPLE_ID_RE.fullmatch(sample_id)
        assert match is not None
        actual_source_counts[
            f"{match.group(1).removeprefix('dataset_')}_{match.group(2).removeprefix('keyframe_')}"
        ] += 1
    for sample_id in validation_ids:
        match = _SAMPLE_ID_RE.fullmatch(sample_id)
        assert match is not None
        actual_validation_counts[
            f"{match.group(1).removeprefix('dataset_')}_{match.group(2).removeprefix('keyframe_')}"
        ] += 1
    if (
        dict(actual_source_counts) != source_counts
        or dict(actual_validation_counts) != validation_counts
    ):
        raise Stage1FrameSplitError("frame count maps do not match sample-ID keyframes")
    raw_excluded = payload.get("excluded_dataset_ids")
    raw_transfer = payload.get("transfer_keyframe_entries")
    if not isinstance(raw_excluded, list) or not all(isinstance(v, str) for v in raw_excluded):
        raise Stage1FrameSplitError("excluded_dataset_ids is invalid")
    if not isinstance(raw_transfer, list) or not all(isinstance(v, str) for v in raw_transfer):
        raise Stage1FrameSplitError("transfer_keyframe_entries is invalid")
    excluded = normalize_dataset_ids(raw_excluded, name="excluded_dataset_ids")
    if not set(STAGE1_FORBIDDEN_DATASET_IDS) <= set(excluded):
        raise Stage1FrameSplitError("dataset6 and dataset7 must be excluded")
    transfer = (
        normalize_keyframe_entries(raw_transfer, name="transfer_keyframe_entries")
        if raw_transfer
        else ()
    )
    if any(not entry.startswith("dataset_7/") for entry in transfer):
        raise Stage1FrameSplitError("transfer selectors must be dataset7-only")
    raw_allowlist = payload.get("dataset_allowlist")
    if not isinstance(raw_allowlist, list) or tuple(normalize_dataset_ids(raw_allowlist)) != tuple(
        STAGE1_TRAIN_DATASET_IDS
    ):
        raise Stage1FrameSplitError("dataset_allowlist is not dataset_1..3")
    record_fingerprint = payload.get("record_fingerprint")
    if not isinstance(record_fingerprint, str) or not record_fingerprint:
        raise Stage1FrameSplitError("record_fingerprint is missing")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise Stage1FrameSplitError("provenance must be an object")
    without_hash = dict(payload)
    manifest_hash = without_hash.pop("manifest_sha256", None)
    if not isinstance(manifest_hash, str) or _hash_payload(without_hash) != manifest_hash:
        raise Stage1FrameSplitError("Stage1 frame split manifest SHA256 does not match content")
    return Stage1FrameSplit(
        split_name=str(payload.get("split_name", "")),
        dataset_revision=str(payload.get("dataset_revision", "")),
        mode=CORRECTED_VIDEO_MODE,
        dataset_ids=dataset_ids,
        keyframe_entries=keyframe_entries,
        source_sample_ids=source_ids,
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        source_frame_counts=source_counts,
        validation_frame_counts=validation_counts,
        validation_fraction=float(fraction),
        block_policy=block_policy,
        rounding=rounding,
        min_validation_frames_per_keyframe=minimum,
        transfer_keyframe_entries=transfer,
        excluded_dataset_ids=excluded,
        record_fingerprint=record_fingerprint,
        provenance=dict(provenance),
        manifest_sha256=manifest_hash,
    )


def load_stage1_frame_split(path: str | Path) -> Stage1FrameSplit:
    """Load, hash-check, and validate a generated frame split manifest."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage1FrameSplitError(
            f"unable to read Stage1 frame split {manifest_path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise Stage1FrameSplitError("Stage1 frame split must be a JSON object")
    return _validate_payload(payload)


def write_stage1_frame_split(path: str | Path, split: Stage1FrameSplit) -> Path:
    """Write once; an existing different manifest is never overwritten."""

    manifest_path = Path(path)
    payload = split.to_payload()
    if manifest_path.exists():
        existing = load_stage1_frame_split(manifest_path)
        if existing.manifest_sha256 != payload["manifest_sha256"]:
            raise Stage1FrameSplitError(
                f"refusing to overwrite a different Stage1 frame split: {manifest_path}"
            )
        return manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest_path


__all__ = [
    "STAGE1_FORBIDDEN_DATASET_IDS",
    "STAGE1_FULL_SPLIT_NAME",
    "STAGE1_FULL_SPLIT_SCHEMA",
    "STAGE1_TRAIN_DATASET_IDS",
    "Stage1FrameSplit",
    "Stage1FrameSplitError",
    "load_stage1_frame_split",
    "make_stage1_frame_split",
    "normalize_dataset_ids",
    "normalize_keyframe_entries",
    "normalize_sample_ids",
    "select_stage1_records",
    "write_stage1_frame_split",
]
