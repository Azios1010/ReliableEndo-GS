"""Active SCARED adapter and A00 deployment access.

This module is deliberately independent of ``scared_c``. Historical SCARED-C
access remains a separate adapter and is loaded only when explicitly requested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from reliable_endo_gs.data.scared_a00 import (
    A01Manifest,
    ScaredA00Adapter,
    ScaredA00Dataset,
    ScaredA00ProtocolError,
    ScaredA00Sample,
    ScaredA00SampleError,
    ScaredSupervision,
    build_a01_manifest,
    collate_scared_a00,
    load_a01_manifest,
)
from reliable_endo_gs.data.validation import DatasetValidationReport
from reliable_endo_gs.teachers.roles import DEFAULT_A01_SCARED_ROLES


@dataclass(frozen=True, slots=True)
class ScaredAdapter:
    """Validate/index active SCARED dataset_1..dataset_7 processed roots."""

    name: str = "scared"
    layout_validation_status: str = "implemented_a00_processed_layout_v1"

    def validate_root(self, root: Path) -> DatasetValidationReport:
        root = Path(root)
        if not root.exists():
            return DatasetValidationReport(
                dataset_name=self.name,
                root=root,
                exists=False,
                is_directory=False,
                layout_checked=True,
                valid=False,
                messages=("STATUS: MISSING", "Configured SCARED root does not exist."),
                status="MISSING",
                protocol="a00_scared_processed_v1",
                contract_valid=False,
            )
        if not root.is_dir():
            return DatasetValidationReport(
                dataset_name=self.name,
                root=root,
                exists=True,
                is_directory=False,
                layout_checked=True,
                valid=False,
                messages=("STATUS: INVALID", "Configured SCARED root is not a directory."),
                status="INVALID",
                protocol="a00_scared_processed_v1",
                contract_valid=False,
            )
        dataset_ids = DEFAULT_A01_SCARED_ROLES.dataset_ids
        present = tuple(dataset_id for dataset_id in dataset_ids if (root / dataset_id).is_dir())
        missing = tuple(dataset_id for dataset_id in dataset_ids if dataset_id not in present)
        sample_count = 0
        protocol_count = 0
        issues: list[str] = []
        for dataset_id in present:
            for keyframe in sorted((root / dataset_id).glob("keyframe_*")):
                data_dir = keyframe / "data" if (keyframe / "data").is_dir() else keyframe
                frame_dir = data_dir / "frame_data"
                frames = tuple(frame_dir.glob("*.json")) if frame_dir.is_dir() else ()
                sample_count += len(frames)
                if (data_dir / "a00_protocol.json").is_file():
                    protocol_count += 1
                if not frames:
                    issues.append(f"{dataset_id}/{keyframe.name}: missing frame_data/*.json")
        if missing:
            issues.append("missing active folds: " + ", ".join(missing))
        if sample_count == 0:
            issues.append("no processed frame_data/*.json samples found")
        if protocol_count == 0:
            issues.append(
                "no a00_protocol.json found; calibration units/direction and rectification cannot be proven"
            )
        valid = not issues
        status = "USABLE" if valid else ("INCOMPLETE" if sample_count else "INVALID")
        messages = (
            f"STATUS: {status}",
            f"found {len(present)}/7 active folds and {sample_count} frame records",
            *issues,
        )
        return DatasetValidationReport(
            dataset_name=self.name,
            root=root,
            exists=True,
            is_directory=True,
            layout_checked=True,
            valid=valid,
            messages=messages,
            status=status,
            protocol="a00_scared_processed_v1",
            sequence_count=len(present),
            sample_count=sample_count,
            stereo_available=sample_count > 0,
            calibration_status="USABLE" if protocol_count else "INCOMPLETE",
            contract_valid=valid,
        )

    def enumerate_sequences(self, root: Path) -> Sequence[str]:
        root = Path(root)
        if not root.is_dir():
            raise ValueError("SCARED root must be an existing directory")
        return tuple(
            dataset_id
            for dataset_id in DEFAULT_A01_SCARED_ROLES.dataset_ids
            if (root / dataset_id).is_dir()
        )

    def inspect_metadata(self, root: Path, sequence_id: str) -> Mapping[str, object]:
        if sequence_id not in DEFAULT_A01_SCARED_ROLES.dataset_ids:
            raise KeyError(
                "active SCARED metadata accepts only dataset_1 through dataset_7; "
                "SCARED-C and dataset_9 are not active"
            )
        sequence_root = Path(root) / sequence_id
        if not sequence_root.is_dir():
            raise KeyError(f"unknown SCARED sequence: {sequence_id}")
        keyframes: list[dict[str, object]] = []
        for keyframe in sorted(path for path in sequence_root.glob("keyframe_*") if path.is_dir()):
            data_dir = keyframe / "data" if (keyframe / "data").is_dir() else keyframe
            frames = tuple(sorted((data_dir / "frame_data").glob("*.json")))
            keyframes.append(
                {
                    "keyframe_id": keyframe.name,
                    "frame_count": len(frames),
                    "has_a00_protocol": (data_dir / "a00_protocol.json").is_file(),
                }
            )
        return {
            "dataset_id": sequence_id,
            "protocol": "a00_scared_processed_v1",
            "keyframes": keyframes,
        }


__all__ = [
    "A01Manifest",
    "ScaredA00Adapter",
    "ScaredA00Dataset",
    "ScaredA00ProtocolError",
    "ScaredA00Sample",
    "ScaredA00SampleError",
    "ScaredAdapter",
    "ScaredSupervision",
    "build_a01_manifest",
    "collate_scared_a00",
    "load_a01_manifest",
]
