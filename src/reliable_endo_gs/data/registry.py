"""Deterministic, explicit dataset adapter registry."""

from collections.abc import Callable

from reliable_endo_gs.data.adapters import (
    C3vdAdapter,
    EndoNerfAdapter,
    ScaredAdapter,
    ScaredCAdapter,
)
from reliable_endo_gs.data.base import DatasetAdapter


class DatasetRegistryError(LookupError):
    """Raised when a dataset adapter name is unknown."""


def normalize_dataset_name(name: str) -> str:
    """Normalize case and separators for registry lookup."""

    normalized = "".join(character for character in name.casefold() if character.isalnum())
    if not normalized:
        raise DatasetRegistryError("Dataset name must contain at least one letter or digit")
    return normalized


_ADAPTER_FACTORIES: dict[str, Callable[[], DatasetAdapter]] = {
    "scared": ScaredAdapter,
    "scaredc": ScaredCAdapter,
    "endonerf": EndoNerfAdapter,
    "c3vd": C3vdAdapter,
}


def list_dataset_names() -> tuple[str, ...]:
    """Return registered dataset names in deterministic order."""

    return tuple(sorted(_ADAPTER_FACTORIES))


def get_dataset_adapter(name: str) -> DatasetAdapter:
    """Return a fresh adapter for a known normalized dataset name."""

    normalized = normalize_dataset_name(name)
    factory = _ADAPTER_FACTORIES.get(normalized)
    if factory is None:
        available = ", ".join(list_dataset_names())
        raise DatasetRegistryError(f"Unknown dataset {name!r}; available adapters: {available}")
    return factory()
