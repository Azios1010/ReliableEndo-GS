"""Tests for the explicit dataset adapter registry."""

import pytest

from reliable_endo_gs.data.adapters import C3vdAdapter, EndoNerfAdapter, ScaredAdapter
from reliable_endo_gs.data.registry import DatasetRegistryError, get_dataset_adapter


@pytest.mark.parametrize(
    ("name", "adapter_type", "canonical_name"),
    [
        ("SCARED", ScaredAdapter, "scared"),
        ("Endo-NeRF", EndoNerfAdapter, "endonerf"),
        ("c3_vd", C3vdAdapter, "c3vd"),
    ],
)
def test_known_dataset_resolves(name: str, adapter_type: type[object], canonical_name: str) -> None:
    adapter = get_dataset_adapter(name)

    assert isinstance(adapter, adapter_type)
    assert adapter.name == canonical_name
    assert adapter.layout_validation_status == "not_implemented"


def test_unknown_dataset_fails_clearly() -> None:
    with pytest.raises(DatasetRegistryError, match="Unknown dataset"):
        get_dataset_adapter("unknown")
