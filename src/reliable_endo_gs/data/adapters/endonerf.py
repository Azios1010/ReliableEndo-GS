"""EndoNeRF dataset identity and validation scaffold."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from reliable_endo_gs.data.validation import (
    DatasetValidationReport,
    validate_dataset_root_basic,
)


@dataclass(frozen=True)
class EndoNerfAdapter:
    """EndoNeRF adapter scaffold with no assumed internal layout."""

    name: str = "endonerf"
    layout_validation_status: str = "not_implemented"

    def validate_root(self, root: Path) -> DatasetValidationReport:
        return validate_dataset_root_basic(self.name, root)

    def enumerate_sequences(self, root: Path) -> Sequence[str]:
        raise NotImplementedError("EndoNeRF sequence layout has not been authoritatively verified")

    def inspect_metadata(self, root: Path, sequence_id: str) -> Mapping[str, object]:
        raise NotImplementedError("EndoNeRF metadata layout has not been authoritatively verified")
