"""Dataset-specific layout handling for the SCARED-C development protocol."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from reliable_endo_gs.data.index import (
    SCARED_C_PROTOCOL,
    SampleIndex,
    build_scared_c_index,
)
from reliable_endo_gs.data.validation import DatasetValidationReport


@dataclass(frozen=True, slots=True)
class ScaredCAdapter:
    """Recognize SCARED-C static keyframe assets without loading at construction."""

    name: str = "scared_c"
    protocol: str = SCARED_C_PROTOCOL
    layout_validation_status: str = "implemented"

    def validate_root(self, root: Path) -> DatasetValidationReport:
        """Validate layout and decode each complete record when explicitly requested."""

        if not root.exists():
            return DatasetValidationReport(
                dataset_name=self.name,
                root=root,
                exists=False,
                is_directory=False,
                layout_checked=True,
                valid=False,
                messages=("STATUS: MISSING",),
                status="MISSING",
                protocol=self.protocol,
                calibration_status="MISSING",
                depth_status="MISSING",
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
                messages=("STATUS: INVALID", "Root is not a directory."),
                status="INVALID",
                protocol=self.protocol,
                calibration_status="INVALID",
                depth_status="INVALID",
                contract_valid=False,
            )

        try:
            index = build_scared_c_index(root, strict=False, protocol=self.protocol)
        except (OSError, TypeError, ValueError) as error:
            return DatasetValidationReport(
                dataset_name=self.name,
                root=root,
                exists=True,
                is_directory=True,
                layout_checked=True,
                valid=False,
                messages=(
                    "STATUS: INVALID",
                    f"Index construction failed ({type(error).__name__}).",
                ),
                status="INVALID",
                protocol=self.protocol,
                calibration_status="INVALID",
                depth_status="INVALID",
                contract_valid=False,
            )

        if index.issues:
            incomplete_markers = ("missing", "no dataset_", "no keyframe_")
            status = (
                "INCOMPLETE"
                if any(
                    any(marker in issue.casefold() for marker in incomplete_markers)
                    for issue in index.issues
                )
                else "INVALID"
            )
            return DatasetValidationReport(
                dataset_name=self.name,
                root=root,
                exists=True,
                is_directory=True,
                layout_checked=True,
                valid=False,
                messages=(f"STATUS: {status}", *index.issues),
                status=status,
                protocol=self.protocol,
                sequence_count=len(index.sequence_ids),
                sample_count=len(index.records),
                stereo_available=False,
                calibration_status="INCOMPLETE" if status == "INCOMPLETE" else "INVALID",
                depth_status="INCOMPLETE" if status == "INCOMPLETE" else "INVALID",
                contract_valid=False,
                index_hash=index.index_hash if index.records else None,
            )
        if not index.records:
            return DatasetValidationReport(
                dataset_name=self.name,
                root=root,
                exists=True,
                is_directory=True,
                layout_checked=True,
                valid=False,
                messages=("STATUS: INCOMPLETE", "No complete SCARED-C keyframe records found."),
                status="INCOMPLETE",
                protocol=self.protocol,
                sequence_count=0,
                sample_count=0,
                stereo_available=False,
                calibration_status="INCOMPLETE",
                depth_status="INCOMPLETE",
                contract_valid=False,
            )

        try:
            from reliable_endo_gs.data.loaders import load_scared_c_sample

            for record in index.records:
                load_scared_c_sample(record)
        except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
            return DatasetValidationReport(
                dataset_name=self.name,
                root=root,
                exists=True,
                is_directory=True,
                layout_checked=True,
                valid=False,
                messages=(
                    "STATUS: INVALID",
                    f"Contract validation failed ({type(error).__name__}).",
                ),
                status="INVALID",
                protocol=self.protocol,
                sequence_count=len(index.sequence_ids),
                sample_count=len(index.records),
                stereo_available=False,
                calibration_status="INVALID",
                depth_status="INVALID",
                contract_valid=False,
                index_hash=index.index_hash,
            )

        return DatasetValidationReport(
            dataset_name=self.name,
            root=root,
            exists=True,
            is_directory=True,
            layout_checked=True,
            valid=True,
            messages=(
                "STATUS: USABLE",
                f"Validated {len(index.records)} complete static keyframe record(s).",
            ),
            status="USABLE",
            protocol=self.protocol,
            sequence_count=len(index.sequence_ids),
            sample_count=len(index.records),
            stereo_available=True,
            calibration_status="USABLE",
            depth_status="USABLE",
            contract_valid=True,
            index_hash=index.index_hash,
        )

    def enumerate_sequences(self, root: Path) -> Sequence[str]:
        """Return deterministic dataset directory names, without decoding tensors."""

        if not root.is_dir():
            raise ValueError("SCARED-C root must be an existing directory")
        return tuple(
            path.name
            for path in sorted(
                (
                    candidate
                    for candidate in root.iterdir()
                    if candidate.is_dir() and candidate.name.casefold().startswith("dataset_")
                ),
                key=lambda path: (path.name.casefold(), path.name),
            )
        )

    def enumerate_samples(self, root: Path) -> SampleIndex:
        """Return the deterministic static-keyframe index for ``root``."""

        return build_scared_c_index(root, strict=True, protocol=self.protocol)

    def inspect_metadata(self, root: Path, sequence_id: str) -> Mapping[str, object]:
        """Read frame-log metadata for one known sequence without exposing paths."""

        if not sequence_id or "/" in sequence_id or "\\" in sequence_id:
            raise ValueError("sequence_id must be a non-empty logical identifier")
        sequence_root = root / sequence_id
        if not sequence_root.is_dir():
            raise KeyError(f"unknown SCARED-C sequence: {sequence_id}")
        keyframes: list[dict[str, object]] = []
        for keyframe_root in sorted(
            (
                path
                for path in sequence_root.iterdir()
                if path.is_dir() and path.name.casefold().startswith("keyframe_")
            ),
            key=lambda path: (path.name.casefold(), path.name),
        ):
            frame_log = keyframe_root / "frame_log.json"
            if not frame_log.is_file():
                continue
            try:
                decoded = json.loads(frame_log.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid frame log for {keyframe_root.name}: {error}") from error
            if not isinstance(decoded, dict):
                raise ValueError(f"frame log for {keyframe_root.name} must be an object")
            included = decoded.get("included_frames", [])
            keyframes.append(
                {
                    "keyframe_id": keyframe_root.name,
                    "frame_log_key": decoded.get("key"),
                    "total_frames_on_disk": decoded.get("total_frames_on_disk"),
                    "included_count": len(included) if isinstance(included, list) else None,
                    "has_frame_data_archive": (
                        keyframe_root / "data" / "frame_data.tar.gz"
                    ).is_file(),
                }
            )
        return {
            "dataset_id": self.name,
            "protocol": self.protocol,
            "sequence_id": sequence_id,
            "keyframes": keyframes,
        }
