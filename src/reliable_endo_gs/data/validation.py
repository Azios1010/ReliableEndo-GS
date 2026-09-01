"""Structured basic validation for external dataset roots."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetValidationReport:
    """Result of explicit dataset-root validation."""

    dataset_name: str
    root: Path
    exists: bool
    is_directory: bool
    layout_checked: bool
    valid: bool
    messages: tuple[str, ...]
    status: str = ""
    protocol: str | None = None
    sequence_count: int | None = None
    sample_count: int | None = None
    stereo_available: bool | None = None
    calibration_status: str | None = None
    depth_status: str | None = None
    contract_valid: bool | None = None
    index_hash: str | None = None

    def to_dict(self, *, redact_paths: bool = True) -> dict[str, object]:
        """Return a JSON-compatible report, redacting external paths by default."""

        return {
            "dataset_name": self.dataset_name,
            "root": "<REDACTED_ABSOLUTE_PATH>" if redact_paths else self.root.as_posix(),
            "exists": self.exists,
            "is_directory": self.is_directory,
            "layout_checked": self.layout_checked,
            "valid": self.valid,
            "messages": list(self.messages),
            "status": self.status or ("USABLE" if self.valid else "INVALID"),
            "protocol": self.protocol,
            "sequence_count": self.sequence_count,
            "sample_count": self.sample_count,
            "stereo_available": self.stereo_available,
            "calibration_status": self.calibration_status,
            "depth_status": self.depth_status,
            "contract_valid": self.contract_valid,
            "index_hash": self.index_hash,
        }


def validate_dataset_root_basic(dataset_name: str, root: Path) -> DatasetValidationReport:
    """Validate only presence and directory type, without layout assumptions."""

    if not root.exists():
        return DatasetValidationReport(
            dataset_name=dataset_name,
            root=root,
            exists=False,
            is_directory=False,
            layout_checked=False,
            valid=False,
            messages=("DATASET NOT PRESENT: configured root does not exist.",),
            status="MISSING",
        )

    if not root.is_dir():
        return DatasetValidationReport(
            dataset_name=dataset_name,
            root=root,
            exists=True,
            is_directory=False,
            layout_checked=False,
            valid=False,
            messages=("DATASET PRESENT BUT INVALID: configured root is not a directory.",),
            status="INVALID",
        )

    return DatasetValidationReport(
        dataset_name=dataset_name,
        root=root,
        exists=True,
        is_directory=True,
        layout_checked=False,
        valid=True,
        messages=(
            "DATASET PRESENT, BASIC VALIDATION PASSED.",
            "DETAILED LAYOUT VALIDATION NOT IMPLEMENTED YET.",
        ),
        status="USABLE",
    )
