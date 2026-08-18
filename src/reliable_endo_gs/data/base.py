"""Non-tensor dataset adapter contracts."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from reliable_endo_gs.data.validation import DatasetValidationReport


class DatasetAdapter(Protocol):
    """Describe dataset identity and metadata inspection boundaries.

    Implementations must not access the filesystem during construction or
    decode scientific data as part of this contract.
    """

    @property
    def name(self) -> str:
        """Return the normalized dataset name."""

    @property
    def layout_validation_status(self) -> str:
        """Describe whether detailed layout validation is implemented."""

    def validate_root(self, root: Path) -> DatasetValidationReport:
        """Validate the external root at explicitly requested validation time."""

    def enumerate_sequences(self, root: Path) -> Sequence[str]:
        """Return sequence identifiers when authoritative layout support exists."""

    def inspect_metadata(self, root: Path, sequence_id: str) -> Mapping[str, object]:
        """Inspect non-tensor metadata for a known sequence."""
