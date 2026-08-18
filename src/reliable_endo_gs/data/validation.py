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

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report."""

        return {
            "dataset_name": self.dataset_name,
            "root": self.root.as_posix(),
            "exists": self.exists,
            "is_directory": self.is_directory,
            "layout_checked": self.layout_checked,
            "valid": self.valid,
            "messages": list(self.messages),
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
    )
